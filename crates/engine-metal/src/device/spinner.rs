//! A thread that does nothing but keep one CPU core busy, so the host side
//! of a streamed decode does not pay to wake up.
//!
//! A miss call spawns sixteen reader threads (`FileWriter::pread`), and the
//! cost of that depends on how long the host has been idle: measured with
//! `scratch/qwen38-profile/tools/ssd_gap.c` on this box, one expert costs
//! 0.71 ms when calls are back to back, 1.01 ms after a 2 ms gap, 1.35 ms
//! after 10 ms and 2.34 ms after 100 ms. That is what a pool that rarely
//! misses sees (`copy_ms = 0.83 + 0.41 m` at 11204 seats against `0.35 +
//! 0.36 m` at 4000, issue 16), and no constant per-call cost can describe
//! it. Keeping the SSD awake with a small read every millisecond does not
//! flatten it; one thread spinning on a core does: 0.53-0.58 ms at every
//! gap up to 30 ms. So the wake-up is the CPU's (thread creation and
//! scheduling on cores that have gone to sleep), and this is the cure.
//!
//! `PIE_METAL_CPU_HEATER=1` turns it on; it spins from the load to the
//! process's end. Off by default: it costs a core.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

pub const ENV: &str = "PIE_METAL_CPU_HEATER";

static STARTED: OnceLock<()> = OnceLock::new();

/// Whether the environment asks for it.
#[must_use]
pub(crate) fn wanted() -> bool {
    matches!(
        std::env::var(ENV).as_deref().map(str::trim),
        Ok("1" | "on" | "true" | "yes")
    )
}

/// Start the spinning thread for this process, once. Says what it did.
pub fn start(wanted: bool) -> String {
    if !wanted {
        return format!("cpu heater off (`{ENV}=1` keeps the host awake between reads)");
    }
    if STARTED.get().is_some() {
        return "cpu heater already running".to_string();
    }
    static STOP: AtomicBool = AtomicBool::new(false);
    let spawned = std::thread::Builder::new()
        .name("pie-metal-cpu-heater".to_string())
        .spawn(|| {
            let mut x = 1.0f64;
            while !STOP.load(Ordering::Relaxed) {
                // Real arithmetic, not a spin hint: the core must stay awake,
                // and a `yield` might let it doze.
                x = std::hint::black_box(x).mul_add(1.000_001, 0.5);
            }
        });
    match spawned {
        Ok(_) => {
            let _ = STARTED.set(());
            "cpu heater on: one thread spinning for the load's life".to_string()
        }
        Err(why) => format!("cpu heater not started: {why}"),
    }
}
