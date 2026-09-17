//! A small kernel that keeps the GPU clocked up while the host reads
//! experts from disk.
//!
//! The SoC lowers the GPU clock when the GPU idles between short bursts of
//! work, and a streamed decode is exactly that pattern: forty cuts a step,
//! each a wait on the host while it seats a layer's experts. llama.cpp's
//! expert store measured the kernels of a step running 20% slower under it
//! and took the whole loss back with a 16 MiB scale kernel kept in flight on
//! its own queue during the gaps (`TAG_METAL_HEATER`). This is that. The
//! tier says when: it arms this as a cut's wait returns and as a fire's last
//! frame lands, and only when the segment before read from disk, which is
//! the pool's own knowledge and no one else's. A frame pauses it as it
//! commits, because real work reaching the queue is the device's business.
//!
//! Measured on Qwen3.6-35B-A3B at 40 cuts a step: with a pool that misses
//! 244 times a step (85 ms of reads), it takes the step from 119 to 112 ms
//! and the frames' own device time from 24.0 to 18.4 ms. With a pool that
//! misses nothing, the gaps are 0.03 ms and its kernels are simply in the
//! way: the same frames' device time goes from 9.7 to 24.1 ms and the step
//! doubles. So it is off unless asked for, and even then it only fires when
//! the last segment read from disk.
//!
//! `PIE_METAL_HEATER=<MiB>` turns it on (`on` for 16 MiB; `0` or `off` for
//! the default, off), `PIE_METAL_HEATER_INFLIGHT` how many are queued
//! ahead (2).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

pub const ENV: &str = "PIE_METAL_HEATER";
pub const INFLIGHT_ENV: &str = "PIE_METAL_HEATER_INFLIGHT";
const DEFAULT_MIB: u64 = 16;
const DEFAULT_INFLIGHT: usize = 2;

struct Shared {
    armed: AtomicBool,
    stop: AtomicBool,
    lock: Mutex<()>,
    wake: Condvar,
}

static SHARED: OnceLock<Arc<Shared>> = OnceLock::new();

/// What the environment asks for: `(MiB, in flight)`, or `None` for off,
/// which is what an unset `PIE_METAL_HEATER` means.
#[must_use]
pub(crate) fn wanted() -> Option<(u64, usize)> {
    let mib = match std::env::var(ENV) {
        Ok(word) => match word.trim().to_ascii_lowercase().as_str() {
            "0" | "off" | "false" | "no" => return None,
            "on" | "auto" => DEFAULT_MIB,
            other => other.parse::<u64>().ok().filter(|&n| n > 0)?,
        },
        Err(_) => return None,
    };
    let inflight = std::env::var(INFLIGHT_ENV)
        .ok()
        .and_then(|word| word.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_INFLIGHT);
    Some((mib, inflight))
}

/// Let the heater run until the next commit.
pub fn arm() {
    if let Some(shared) = SHARED.get()
        && !shared.armed.swap(true, Ordering::AcqRel)
    {
        let _guard = shared
            .lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        shared.wake.notify_one();
    }
}

/// Real work is about to be committed: stop queueing kernels.
pub fn pause() {
    if let Some(shared) = SHARED.get() {
        shared.armed.store(false, Ordering::Release);
    }
}

#[must_use]
pub fn running() -> bool {
    SHARED.get().is_some()
}

#[cfg(target_vendor = "apple")]
mod apple {
    use super::*;
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
    };

    const SOURCE: &str = "#include <metal_stdlib>\n\
        using namespace metal;\n\
        kernel void pie_heater(device float* x [[buffer(0)]], \
        uint i [[thread_position_in_grid]]) { x[i] = x[i] * 0.999f + 1.0f; }\n";

    pub(super) struct Kit {
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        elements: usize,
    }

    // SAFETY: the queue, buffer and pipeline are used from the heater thread
    // only, once built; Metal documents all three thread-safe for this.
    unsafe impl Send for Kit {}

    impl Kit {
        pub(super) fn open(
            device: &ProtocolObject<dyn MTLDevice>,
            mib: u64,
        ) -> std::result::Result<Kit, String> {
            let queue = device
                .newCommandQueue()
                .ok_or_else(|| "the device would not open a second command queue".to_string())?;
            let bytes = usize::try_from(mib << 20).map_err(|_| "too many MiB".to_string())?;
            let buffer = device
                .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| format!("the device declined {mib} MiB for the heater"))?;
            let source = crate::device::ctx::nsstring(SOURCE);
            let library = device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|error| error.localizedDescription().to_string())?;
            let name = crate::device::ctx::nsstring("pie_heater");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "the heater kernel did not compile to an entrypoint".to_string())?;
            let pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| error.localizedDescription().to_string())?;
            Ok(Kit {
                queue,
                buffer,
                pipeline,
                elements: bytes / 4,
            })
        }

        /// One kernel, committed. Returns the command buffer to wait on.
        pub(super) fn fire(&self) -> Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
            let buffer = self.queue.commandBuffer()?;
            let encoder = buffer.computeCommandEncoder()?;
            encoder.setComputePipelineState(&self.pipeline);
            // SAFETY: a live buffer bound at index 0, which the kernel reads
            // and writes within its own length.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&self.buffer), 0, 0);
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: self.elements,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            buffer.commit();
            Some(buffer)
        }
    }

    pub(super) fn run(kit: Kit, shared: Arc<Shared>, inflight: usize) {
        loop {
            {
                let mut guard = shared
                    .lock
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                while !shared.armed.load(Ordering::Acquire) && !shared.stop.load(Ordering::Acquire)
                {
                    guard = shared
                        .wake
                        .wait(guard)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
            }
            if shared.stop.load(Ordering::Acquire) {
                return;
            }
            let mut queued: std::collections::VecDeque<_> =
                std::collections::VecDeque::with_capacity(inflight);
            while shared.armed.load(Ordering::Acquire) && !shared.stop.load(Ordering::Acquire) {
                while queued.len() < inflight {
                    match kit.fire() {
                        Some(buffer) => queued.push_back(buffer),
                        None => {
                            shared.stop.store(true, Ordering::Release);
                            break;
                        }
                    }
                }
                if let Some(buffer) = queued.pop_front() {
                    buffer.waitUntilCompleted();
                }
            }
            for buffer in queued.drain(..) {
                buffer.waitUntilCompleted();
            }
        }
    }
}

/// Start the heater thread for this process, once. Says what it did.
pub fn start(device: &super::Context, wanted: Option<(u64, usize)>) -> String {
    let Some((mib, inflight)) = wanted else {
        return format!("heater off (`{ENV}=<MiB>` holds the clock up through long reads)");
    };
    if SHARED.get().is_some() {
        return "heater already running".to_string();
    }
    #[cfg(target_vendor = "apple")]
    {
        let kit = match apple::Kit::open(device.device(), mib) {
            Ok(kit) => kit,
            Err(why) => return format!("heater not started: {why}"),
        };
        let shared = Arc::new(Shared {
            armed: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            lock: Mutex::new(()),
            wake: Condvar::new(),
        });
        let worker = Arc::clone(&shared);
        let spawned = std::thread::Builder::new()
            .name("pie-metal-heater".to_string())
            .spawn(move || apple::run(kit, worker, inflight));
        match spawned {
            Ok(_) => {
                let _ = SHARED.set(shared);
                format!("heater on: {mib} MiB x {inflight} in flight during the cuts")
            }
            Err(why) => format!("heater not started: {why}"),
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (device, mib, inflight);
        "heater unavailable off Apple".to_string()
    }
}
