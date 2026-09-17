// New on this branch: what the pool reports and logs.

use super::*;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Prediction {
    pub total: u64,
    pub covered: [u64; 4],
    pub misses: u64,
    pub saved: [u64; 4],
    pub prefetched: u64,
}

/// The pool's tally, for the log and the shutdown line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheReport {
    pub policy: String,
    pub slots: u32,
    pub pairs: u64,
    pub per_slot: u64,
    pub resident: u64,
    pub distinct: u64,
    pub hits: u64,
    pub misses: u64,
    pub bytes_read: u64,
    pub segments: u64,
    pub copy_ns: u64,
    pub wait_ns: u64,
    pub gpu_ns: u64,
    pub ring_fills: u64,
    pub ring_copied: u64,
    pub ring_read: u64,
    pub ring_bytes: u64,
    pub ring_wait_ns: u64,
}

impl std::fmt::Display for CacheReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let looked = self.hits + self.misses;
        let rate = if looked == 0 {
            0.0
        } else {
            self.hits as f64 * 100.0 / looked as f64
        };
        write!(
            f,
            "expert-cache: {} seats x {:.2} MiB ({:.2} GiB) over {} (layer, expert) pairs \
             [{}]; {} resident, {} distinct ever seated; {} hits / {} misses ({rate:.1}% hit) \
             across {} segments; {:.3} GiB read from disk ({:.1} ms copying, {:.1} ms waiting \
             on the device, {:.1} ms of device time in the cut frames)",
            self.slots,
            self.per_slot as f64 / (1u64 << 20) as f64,
            gib(u64::from(self.slots) * self.per_slot),
            self.pairs,
            self.policy,
            self.resident,
            self.distinct,
            self.hits,
            self.misses,
            self.segments,
            gib(self.bytes_read),
            self.copy_ns as f64 / 1e6,
            self.wait_ns as f64 / 1e6,
            self.gpu_ns as f64 / 1e6,
        )?;
        if self.ring_fills > 0 {
            write!(
                f,
                "; prefill ring: {} fills, {} experts copied from seats, {} read ({:.3} GiB), \
                 {:.1} ms waited for a fill",
                self.ring_fills,
                self.ring_copied,
                self.ring_read,
                gib(self.ring_bytes),
                self.ring_wait_ns as f64 / 1e6,
            )?;
        }
        Ok(())
    }
}

/// One fire's slice of the tally, appended to `PIE_EXPERT_CACHE_LOG` as CSV.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FireRecord {
    pub rows: u32,
    pub cuts: u64,
    pub copies: u64,
    pub hits: u64,
    pub misses: u64,
    pub bytes_read: u64,
    pub cut_ms: f64,
    pub copy_ms: f64,
    pub wait_ms: f64,
    pub walk_ms: f64,
    /// Device time of this fire's cut frames (every frame but the last).
    pub gpu_cut_ms: f64,
    /// Device time of final frames that landed since the previous record:
    /// the fire before this one, in steady decode.
    pub gpu_tail_ms: f64,
}

type FireLog = std::sync::Mutex<(u64, std::io::BufWriter<std::fs::File>)>;

static LOG: std::sync::OnceLock<Option<FireLog>> = std::sync::OnceLock::new();

/// Open the per-fire CSV the load asked for (`PIE_EXPERT_CACHE_LOG`), once.
/// Until this runs, and when it is handed nothing, `log_fire` writes nowhere.
pub(super) fn open_log(path: Option<&std::path::Path>) {
    let _ = LOG.get_or_init(|| {
        use std::io::Write;
        let mut file = std::io::BufWriter::new(std::fs::File::create(path?).ok()?);
        let _ = writeln!(
            file,
            "seq,rows,cuts,copies,hits,misses,bytes_read,cut_ms,copy_ms,wait_ms,walk_ms,\
             gpu_cut_ms,gpu_tail_ms"
        );
        let _ = file.flush();
        Some(std::sync::Mutex::new((0, file)))
    });
}

fn fire_log() -> Option<&'static FireLog> {
    LOG.get()?.as_ref()
}

pub fn log_fire(record: &FireRecord) {
    use std::io::Write;
    let Some(log) = fire_log() else {
        return;
    };
    let mut log = log
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (seq, file) = &mut *log;
    let _ = writeln!(
        file,
        "{seq},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        record.rows,
        record.cuts,
        record.copies,
        record.hits,
        record.misses,
        record.bytes_read,
        record.cut_ms,
        record.copy_ms,
        record.wait_ms,
        record.walk_ms,
        record.gpu_cut_ms,
        record.gpu_tail_ms,
    );
    let _ = file.flush();
    *seq += 1;
}
