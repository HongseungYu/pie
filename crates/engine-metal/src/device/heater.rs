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
//! frame lands, and — under the `reads` policy — only when the segment
//! before read from disk, which is the pool's own knowledge and no one
//! else's. A frame pauses it as it commits, because real work reaching the
//! queue is the device's business.
//!
//! Measured on Qwen3.6-35B-A3B at 40 cuts a step: with a pool that misses
//! 244 times a step (85 ms of reads), it takes the step from 119 to 112 ms
//! and the frames' own device time from 24.0 to 18.4 ms. With a pool that
//! misses nothing, the gaps are 0.03 ms and its kernels are simply in the
//! way: the same frames' device time goes from 9.7 to 24.1 ms and the step
//! doubles. So it is off unless asked for, and even then it only fires when
//! the last segment read from disk.
//!
//! On Qwen3.8-Flash-Next (48 cuts a step) that was not enough: the frames'
//! device time still climbed from 30.6 ms at no misses to 43 ms at 264,
//! heater on. Two things the `reads` rule leaves out: the ~9 ms gap between
//! one fire's last frame and the next fire's first, which arms nothing
//! unless layer 47 happened to read; and the two 16 MiB kernels in flight
//! at `pause()`, which run beside the real frame (~0.25 ms a missing
//! layer). Hence the knobs below, so the policy and the kernel's shape can
//! be swept without a rebuild.
//!
//! `PIE_METAL_HEATER=on` turns it on as the ALU kernel at every gap, the
//! setting measured best below; `PIE_METAL_HEATER=<MiB>` asks for the scale
//! kernel of that size, armed after a read (`0` or `off` for the default,
//! off). `PIE_METAL_HEATER_INFLIGHT` says how many are queued ahead (2).
//! `PIE_METAL_HEATER_ARM=reads|always` says when the tier arms it (`reads`:
//! after a segment that read from disk; `always`: at every cut wait and
//! every fire tail). `PIE_METAL_HEATER_KERNEL=mem|alu` picks the
//! kernel: `mem` scales the buffer (`x = 0.999 x + 1`, bandwidth-bound,
//! wide); `alu` runs `PIE_METAL_HEATER_ALU_THREADS` (1024) threads through
//! `PIE_METAL_HEATER_ALU_ITERS` (8192) dependent FMAs each — narrow and
//! compute-bound, so the device is never idle yet a kernel still in flight
//! when a frame commits takes a sliver of it; `spin` is the same chain but
//! one dispatch an armed window, polling a flag the host clears at `pause()`
//! (`_ALU_ITERS` chunks of 256 at most), so the host commits almost nothing
//! and a short kernel does not have to be a frequent one. `PIE_METAL_HEATER_LOG=<path>`
//! appends one line per armed window: when it opened, how many kernels ran,
//! their summed and mean device time — fixed work, so the mean is a direct
//! reading of the clock in the gaps.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

pub const ENV: &str = "PIE_METAL_HEATER";
pub const INFLIGHT_ENV: &str = "PIE_METAL_HEATER_INFLIGHT";
pub const ARM_ENV: &str = "PIE_METAL_HEATER_ARM";
pub const KERNEL_ENV: &str = "PIE_METAL_HEATER_KERNEL";
pub const ALU_THREADS_ENV: &str = "PIE_METAL_HEATER_ALU_THREADS";
pub const ALU_ITERS_ENV: &str = "PIE_METAL_HEATER_ALU_ITERS";
pub const LOG_ENV: &str = "PIE_METAL_HEATER_LOG";
const DEFAULT_MIB: u64 = 16;
const DEFAULT_INFLIGHT: usize = 2;
const DEFAULT_ALU_THREADS: u64 = 1024;
const DEFAULT_ALU_ITERS: u32 = 8192;

/// What the heater kernel does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// Scale a buffer of this many MiB: wide and bandwidth-bound.
    Mem { mib: u64 },
    /// This many threads, each through this many dependent FMAs: narrow and
    /// compute-bound.
    Alu { threads: u64, iters: u32 },
    /// This many threads spinning on FMAs until the host clears a flag or
    /// `iters` chunks of 256 have passed: one dispatch an armed window, so
    /// the host commits almost nothing and a frame waits on nothing longer
    /// than one chunk.
    Spin { threads: u64, iters: u32 },
}

/// Everything the environment says about the heater.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub kernel: Kernel,
    pub inflight: usize,
    /// Arm at every cut wait and fire tail, not only after a disk read.
    pub always: bool,
    pub log: Option<std::path::PathBuf>,
}

struct Shared {
    armed: AtomicBool,
    stop: AtomicBool,
    always: bool,
    lock: Mutex<()>,
    wake: Condvar,
    /// The spin kernel's flag, in a shared buffer: 1 while it should keep
    /// spinning, 0 to make it return. Null for the other kernels.
    flag: std::sync::atomic::AtomicPtr<std::sync::atomic::AtomicU32>,
}

impl Shared {
    fn flag(&self, value: u32) {
        let flag = self.flag.load(Ordering::Acquire);
        if !flag.is_null() {
            // SAFETY: the pointer names the first word of the heater's own
            // shared buffer, which the kit keeps alive for the thread's life.
            unsafe { (*flag).store(value, Ordering::Release) };
        }
    }
}

static SHARED: OnceLock<Arc<Shared>> = OnceLock::new();

/// What the environment asks for, or `None` for off, which is what an unset
/// `PIE_METAL_HEATER` means.
#[must_use]
pub(crate) fn wanted() -> Option<Config> {
    let (mib, on_word) = match std::env::var(ENV) {
        Ok(word) => match word.trim().to_ascii_lowercase().as_str() {
            "0" | "off" | "false" | "no" => return None,
            "on" | "auto" => (DEFAULT_MIB, true),
            other => (other.parse::<u64>().ok().filter(|&n| n > 0)?, false),
        },
        Err(_) => return None,
    };
    let number = |env: &str| {
        std::env::var(env)
            .ok()
            .and_then(|word| word.trim().parse::<u64>().ok())
            .filter(|&n| n > 0)
    };
    let inflight = number(INFLIGHT_ENV).map_or(DEFAULT_INFLIGHT, |n| n as usize);
    let threads = number(ALU_THREADS_ENV).unwrap_or(DEFAULT_ALU_THREADS);
    let iters = number(ALU_ITERS_ENV).map_or(DEFAULT_ALU_ITERS, |n| n.min(u64::from(u32::MAX)) as u32);
    // `on` is the setting the step-model effort settled on (issue 02 of
    // .scratch/decode-step-model): the narrow ALU kernel at every gap holds
    // the cut frames at 29.7 ms with no misses and 31.5 at 331, where the
    // 16 MiB scale after a read held 30.1 and 43.4. A MiB count still asks
    // for the scale kernel, and `_KERNEL` / `_ARM` say otherwise explicitly.
    let kernel = match std::env::var(KERNEL_ENV)
        .map(|word| word.trim().to_ascii_lowercase())
        .as_deref()
    {
        Ok("alu") => Kernel::Alu { threads, iters },
        Ok("spin") => Kernel::Spin { threads, iters },
        Ok("mem") => Kernel::Mem { mib },
        _ if on_word => Kernel::Alu { threads, iters },
        _ => Kernel::Mem { mib },
    };
    let always = match std::env::var(ARM_ENV)
        .map(|word| word.trim().to_ascii_lowercase())
        .as_deref()
    {
        Ok("always") => true,
        Ok("reads") => false,
        _ => on_word,
    };
    Some(Config {
        kernel,
        inflight,
        always,
        log: std::env::var_os(LOG_ENV).map(std::path::PathBuf::from),
    })
}

/// Let the heater run until the next commit.
pub fn arm() {
    if let Some(shared) = SHARED.get()
        && !shared.armed.swap(true, Ordering::AcqRel)
    {
        shared.flag(1);
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
        shared.flag(0);
    }
}

#[must_use]
pub fn running() -> bool {
    SHARED.get().is_some()
}

/// Whether the tier should arm this at every gap, not only after a read.
#[must_use]
pub fn always() -> bool {
    SHARED.get().is_some_and(|shared| shared.always)
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
        uint i [[thread_position_in_grid]]) { x[i] = x[i] * 0.999f + 1.0f; }\n\
        kernel void pie_heater_alu(device float* x [[buffer(0)]], \
        constant uint& iters [[buffer(1)]], uint i [[thread_position_in_grid]]) { \
        float a = x[i] + 1.0f; float b = a * 0.5f + 1.0f; \
        for (uint k = 0; k < iters; k++) { a = fma(a, 0.999f, b); b = fma(b, 0.998f, a); } \
        x[i] = a + b; }\n\
        kernel void pie_heater_spin(device float* x [[buffer(0)]], \
        constant uint& chunks [[buffer(1)]], device atomic_uint* flag [[buffer(2)]], \
        uint i [[thread_position_in_grid]]) { \
        float a = x[i] + 1.0f; float b = a * 0.5f + 1.0f; \
        for (uint c = 0; c < chunks; c++) { \
        if (atomic_load_explicit(flag, memory_order_relaxed) == 0) break; \
        for (uint k = 0; k < 256; k++) { a = fma(a, 0.999f, b); b = fma(b, 0.998f, a); } } \
        x[i] = a + b; }\n";

    pub(super) struct Kit {
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        /// The spin kernel's flag word, when there is one.
        flag: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        elements: usize,
        iters: Option<u32>,
    }

    // SAFETY: the queue, buffer and pipeline are used from the heater thread
    // only, once built; Metal documents all three thread-safe for this.
    unsafe impl Send for Kit {}

    impl Kit {
        pub(super) fn open(
            device: &ProtocolObject<dyn MTLDevice>,
            kernel: Kernel,
        ) -> std::result::Result<Kit, String> {
            let queue = device
                .newCommandQueue()
                .ok_or_else(|| "the device would not open a second command queue".to_string())?;
            let (bytes, entry, iters) = match kernel {
                Kernel::Mem { mib } => (mib << 20, "pie_heater", None),
                Kernel::Alu { threads, iters } => (threads * 4, "pie_heater_alu", Some(iters)),
                Kernel::Spin { threads, iters } => (threads * 4, "pie_heater_spin", Some(iters)),
            };
            let flag = match kernel {
                Kernel::Spin { .. } => Some(
                    device
                        .newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
                        .ok_or_else(|| "the device declined 4 bytes for the heater's flag".to_string())?,
                ),
                _ => None,
            };
            let bytes = usize::try_from(bytes).map_err(|_| "too many bytes".to_string())?;
            let buffer = device
                .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| format!("the device declined {bytes} bytes for the heater"))?;
            let source = crate::device::ctx::nsstring(SOURCE);
            let library = device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|error| error.localizedDescription().to_string())?;
            let name = crate::device::ctx::nsstring(entry);
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "the heater kernel did not compile to an entrypoint".to_string())?;
            let pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| error.localizedDescription().to_string())?;
            Ok(Kit {
                queue,
                buffer,
                flag,
                pipeline,
                elements: bytes / 4,
                iters,
            })
        }

        /// The flag word of the spin kernel, host-writable, or null.
        pub(super) fn flag_word(&self) -> *mut std::sync::atomic::AtomicU32 {
            match &self.flag {
                Some(flag) => flag.contents().as_ptr().cast(),
                None => std::ptr::null_mut(),
            }
        }

        /// One kernel, committed. Returns the command buffer to wait on.
        pub(super) fn fire(&self) -> Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
            let buffer = self.queue.commandBuffer()?;
            let encoder = buffer.computeCommandEncoder()?;
            encoder.setComputePipelineState(&self.pipeline);
            // SAFETY: a live buffer bound at index 0, which the kernel reads
            // and writes within its own length; the iteration count is a
            // 4-byte constant copied at encode time.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&self.buffer), 0, 0);
                if let Some(iters) = self.iters {
                    let iters: u32 = iters;
                    encoder.setBytes_length_atIndex(
                        std::ptr::NonNull::from(&iters).cast(),
                        std::mem::size_of::<u32>(),
                        1,
                    );
                }
                if let Some(flag) = &self.flag {
                    encoder.setBuffer_offset_atIndex(Some(flag), 0, 2);
                }
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: self.elements,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.elements.min(256),
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            buffer.commit();
            Some(buffer)
        }
    }

    /// Device time of a completed command buffer, in milliseconds.
    fn span_ms(buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> f64 {
        (buffer.GPUEndTime() - buffer.GPUStartTime()) * 1e3
    }

    pub(super) fn run(
        kit: Kit,
        shared: Arc<Shared>,
        inflight: usize,
        mut log: Option<std::io::BufWriter<std::fs::File>>,
    ) {
        use std::io::Write;
        let began = std::time::Instant::now();
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
            let opened = began.elapsed().as_secs_f64() * 1e3;
            let (mut kernels, mut busy, mut least, mut most) = (0u64, 0.0f64, f64::MAX, 0.0f64);
            let mut note = |buffer: &ProtocolObject<dyn MTLCommandBuffer>| {
                let ms = span_ms(buffer);
                kernels += 1;
                busy += ms;
                least = least.min(ms);
                most = most.max(ms);
            };
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
                    note(&buffer);
                }
            }
            for buffer in queued.drain(..) {
                buffer.waitUntilCompleted();
                note(&buffer);
            }
            if let Some(log) = log.as_mut()
                && kernels > 0
            {
                let _ = writeln!(
                    log,
                    "{opened:.3},{:.3},{kernels},{busy:.3},{:.4},{least:.4},{most:.4}",
                    began.elapsed().as_secs_f64() * 1e3,
                    busy / kernels as f64,
                );
            }
        }
    }
}

/// Start the heater thread for this process, once. Says what it did.
pub fn start(device: &super::Context, wanted: Option<Config>) -> String {
    let Some(config) = wanted else {
        return format!("heater off (`{ENV}=<MiB>` holds the clock up through long reads)");
    };
    if SHARED.get().is_some() {
        return "heater already running".to_string();
    }
    #[cfg(target_vendor = "apple")]
    {
        let kit = match apple::Kit::open(device.device(), config.kernel) {
            Ok(kit) => kit,
            Err(why) => return format!("heater not started: {why}"),
        };
        let log = config.log.as_ref().and_then(|path| {
            use std::io::Write;
            let mut file = std::io::BufWriter::new(std::fs::File::create(path).ok()?);
            let _ = writeln!(file, "opened_ms,closed_ms,kernels,busy_ms,mean_ms,min_ms,max_ms");
            Some(file)
        });
        let shared = Arc::new(Shared {
            armed: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            always: config.always,
            lock: Mutex::new(()),
            wake: Condvar::new(),
            flag: std::sync::atomic::AtomicPtr::new(kit.flag_word()),
        });
        let worker = Arc::clone(&shared);
        let inflight = config.inflight;
        let spawned = std::thread::Builder::new()
            .name("pie-metal-heater".to_string())
            .spawn(move || apple::run(kit, worker, inflight, log));
        match spawned {
            Ok(_) => {
                let _ = SHARED.set(shared);
                let what = match config.kernel {
                    Kernel::Mem { mib } => format!("{mib} MiB scale"),
                    Kernel::Alu { threads, iters } => format!("{threads} threads x {iters} FMAs"),
                    Kernel::Spin { threads, iters } => {
                        format!("{threads} threads spinning up to {iters} x 256 FMAs")
                    }
                };
                format!(
                    "heater on: {what} x {inflight} in flight, armed {}{}",
                    if config.always { "at every gap" } else { "after a read" },
                    config
                        .log
                        .as_ref()
                        .map_or(String::new(), |path| format!(", logged to {}", path.display())),
                )
            }
            Err(why) => format!("heater not started: {why}"),
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (device, config);
        "heater unavailable off Apple".to_string()
    }
}
