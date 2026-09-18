// New on this branch: what the pool reports and logs.

use super::plan::gib;

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
    /// Device time of final frames that landed since the previous record
    /// closed: the fire before this one, in steady decode.
    pub gpu_tail_ms: f64,
    /// Host time of the n-gram row gather at the hasher's cut: the join of
    /// a prefetch, and any rows read there and then.
    pub ple_ms: f64,
    /// N-gram rows this fire read from the artifact, and how many of them
    /// the prefetch had not already landed (read at the cut instead).
    pub ple_reads: u64,
    pub ple_prefetch_misses: u64,
    /// Wall clock as this fire opened, ms since the load's first fire. The
    /// step a decode takes is the gap between two of these, which is the
    /// only way to read a step off one request rather than by subtracting
    /// one request's wall time from another's.
    pub t_ms: f64,
}

/// One cut's slice: a layer's own misses, and what they cost it.
#[derive(Debug, Clone, Copy, Default)]
pub struct CutRecord {
    pub fire: u64,
    pub group: u32,
    pub rows: u32,
    pub misses: u64,
    pub hits: u64,
    pub copies: u64,
    pub bytes_read: u64,
    /// Host time this cut spent reading its misses.
    pub copy_ms: f64,
    /// Host time waiting on the device for the frame this cut closed.
    pub wait_ms: f64,
    /// Device time that frame reported.
    pub gpu_ms: f64,
    /// The whole cut, reading and seating together.
    pub cut_ms: f64,
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
             gpu_cut_ms,gpu_tail_ms,t_ms,ple_ms,ple_reads,ple_prefetch_misses"
        );
        let _ = file.flush();
        Some(std::sync::Mutex::new((0, file)))
    });
}

fn fire_log() -> Option<&'static FireLog> {
    LOG.get()?.as_ref()
}

type CutLog = std::sync::Mutex<std::io::BufWriter<std::fs::File>>;

static CUTS: std::sync::OnceLock<Option<CutLog>> = std::sync::OnceLock::new();

/// Open the per-cut CSV the load asked for (`PIE_EXPERT_CACHE_CUT_LOG`).
pub(super) fn open_cuts(path: Option<&std::path::Path>) {
    let _ = CUTS.get_or_init(|| {
        use std::io::Write;
        let mut file = std::io::BufWriter::new(std::fs::File::create(path?).ok()?);
        let _ = writeln!(
            file,
            "fire,group,rows,misses,hits,copies,bytes_read,copy_ms,wait_ms,gpu_ms,cut_ms"
        );
        let _ = file.flush();
        Some(std::sync::Mutex::new(file))
    });
}

pub(super) fn log_cut(record: &CutRecord) {
    use std::io::Write;
    let Some(log) = CUTS.get().and_then(Option::as_ref) else {
        return;
    };
    let mut file = log
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let _ = writeln!(
        file,
        "{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3}",
        record.fire,
        record.group,
        record.rows,
        record.misses,
        record.hits,
        record.copies,
        record.bytes_read,
        record.copy_ms,
        record.wait_ms,
        record.gpu_ms,
        record.cut_ms,
    );
    // Flushed with the fire's own record, not here: a flush a cut was 48
    // syscalls a step in the measured window.
}

/// Push the cut rows of a fire to disk beside its record.
fn flush_cuts() {
    use std::io::Write;
    if let Some(log) = CUTS.get().and_then(Option::as_ref) {
        let _ = log
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .flush();
    }
}

pub fn log_fire(record: &FireRecord) {
    use std::io::Write;
    flush_cuts();
    let Some(log) = fire_log() else {
        return;
    };
    let mut log = log
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (seq, file) = &mut *log;
    let _ = writeln!(
        file,
        "{seq},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{}",
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
        record.t_ms,
        record.ple_ms,
        record.ple_reads,
        record.ple_prefetch_misses,
    );
    let _ = file.flush();
    *seq += 1;
}

/// What the pool has done, counted as it goes, and one fire's slice of it.
/// The tier owns this: nothing outside reads a counter in order to subtract
/// it from another one later.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Mark {
    swaps: u64,
    segments: u64,
    hits: u64,
    misses: u64,
    bytes_read: u64,
    cut_ns: u64,
    copy_ns: u64,
    wait_ns: u64,
    gpu_ns: u64,
    tail_ns: u64,
    rows_ns: u64,
}

#[derive(Debug, Default)]
pub(super) struct Tally {
    pub(super) swaps: u64,
    pub(super) segments: u64,
    pub(super) hits: u64,
    pub(super) misses: u64,
    pub(super) bytes_read: u64,
    pub(super) distinct: u64,
    pub(super) cut_ns: u64,
    pub(super) copy_ns: u64,
    pub(super) wait_ns: u64,
    pub(super) gpu_ns: u64,
    /// Device time of the fires' final frames, summed as they land.
    pub(super) tail_ns: u64,
    /// Host time of the n-gram row gathers.
    pub(super) rows_ns: u64,
    /// This fire's n-gram row reads, and how many the prefetch missed.
    pub(super) rows_read: u64,
    pub(super) rows_missed: u64,
    /// When the load's first fire opened, and when this one did.
    first: Option<std::time::Instant>,
    at_ms: f64,
    pub(super) prediction: Prediction,
    fires: u64,
    report_every: u64,
    at: Mark,
    /// The counters as the cut before this one closed.
    cut_at: Mark,
}

impl Tally {
    pub(super) fn new(report_every: u64) -> Tally {
        Tally {
            report_every,
            ..Tally::default()
        }
    }

    fn mark(&self) -> Mark {
        Mark {
            swaps: self.swaps,
            segments: self.segments,
            hits: self.hits,
            misses: self.misses,
            bytes_read: self.bytes_read,
            cut_ns: self.cut_ns,
            copy_ns: self.copy_ns,
            wait_ns: self.wait_ns,
            gpu_ns: self.gpu_ns,
            tail_ns: self.tail_ns,
            rows_ns: self.rows_ns,
        }
    }

    /// A fire opens: when it opened is what says how long the one before it
    /// took. The counters' mark stays where the last close left it, so a
    /// final frame that lands between one fire's close and the next one's
    /// open — the readout's wait — is counted by the record that follows,
    /// rather than by none.
    pub(super) fn open(&mut self) {
        self.at_ms = self
            .first
            .get_or_insert_with(std::time::Instant::now)
            .elapsed()
            .as_secs_f64()
            * 1e3;
    }

    /// A fire closes: its own slice, and one more fire on the count.
    pub(super) fn close(&mut self, rows: u32, walk: std::time::Duration) -> FireRecord {
        let at = self.at;
        self.at = self.mark();
        self.fires += 1;
        FireRecord {
            rows,
            cuts: self.segments - at.segments,
            copies: self.swaps - at.swaps,
            hits: self.hits - at.hits,
            misses: self.misses - at.misses,
            bytes_read: self.bytes_read - at.bytes_read,
            cut_ms: (self.cut_ns - at.cut_ns) as f64 / 1e6,
            copy_ms: (self.copy_ns - at.copy_ns) as f64 / 1e6,
            wait_ms: (self.wait_ns - at.wait_ns) as f64 / 1e6,
            walk_ms: walk.as_secs_f64() * 1e3,
            gpu_cut_ms: (self.gpu_ns - at.gpu_ns) as f64 / 1e6,
            gpu_tail_ms: (self.tail_ns - at.tail_ns) as f64 / 1e6,
            t_ms: self.at_ms,
            ple_ms: (self.rows_ns - at.rows_ns) as f64 / 1e6,
            ple_reads: std::mem::take(&mut self.rows_read),
            ple_prefetch_misses: std::mem::take(&mut self.rows_missed),
        }
    }

    pub(super) fn fires(&self) -> u64 {
        self.fires
    }

    /// One cut closes: its own slice of the counters since the cut before.
    pub(super) fn close_cut(&mut self, group: u32, rows: u32, cut_ms: f64) -> CutRecord {
        let at = self.cut_at;
        self.cut_at = self.mark();
        CutRecord {
            fire: self.fires,
            group,
            rows,
            misses: self.misses - at.misses,
            hits: self.hits - at.hits,
            copies: self.swaps - at.swaps,
            bytes_read: self.bytes_read - at.bytes_read,
            copy_ms: (self.copy_ns - at.copy_ns) as f64 / 1e6,
            wait_ms: (self.wait_ns - at.wait_ns) as f64 / 1e6,
            gpu_ms: (self.gpu_ns - at.gpu_ns) as f64 / 1e6,
            cut_ms,
        }
    }

    /// Whether this fire is one the pool says a word about.
    pub(super) fn due(&self) -> bool {
        self.report_every > 0 && self.fires.is_multiple_of(self.report_every)
    }
}
