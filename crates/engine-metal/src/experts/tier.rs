// Replaced main's: the tier that seats a segment's experts in the pool.

use std::collections::BTreeMap;

use kernels_metal::Tensor;
use model_exec::fire::MaskSpan;
use model_ir::ValueId;

use crate::device::{Buffer, Handles};
use crate::error::{Fault, Result};
use crate::weight_store::Store;

use super::pool::{Band, Group, Hold, Pool};
use super::source::Bytes;
use super::tally;
use super::{CACHE_ENV, CacheReport, GroupResidency, Plan, Source, uncached};

/// Two whole layers of seats, borrowed from the pool for one prefill fire.
/// A fire in which some lane has `min_rows` rows or more names nearly every
/// expert of every layer, so its routed matmuls run over a half holding the
/// whole layer while the next layer fills the other: experts the pool holds
/// are copied from their seats, the rest are read. The fire takes the two
/// halves at its start and gives them back at its end holding the last two
/// layers it read, so the damage to the working set is those `2 * experts`
/// seats whatever the layer count. After llama.cpp's expert store
/// (`TAG_MOE_STORE_PREFILL`) and FreeToken's prefill.
#[derive(Debug)]
struct Ring {
    experts: u32,
    /// The seats this fire borrowed, `2 * experts` of them: half `h` holds
    /// `seats[h * experts ..][.. experts]`, one seat per expert of the layer
    /// that half was filled with. Empty between prefill fires.
    seats: Vec<u32>,
    holds: [Option<usize>; 2],
    filling: Option<(usize, usize, std::thread::JoinHandle<Result<()>>)>,
    fills: u64,
    copied: u64,
    read: u64,
    bytes: u64,
    lookups: u64,
    wait_ns: u64,
}

type Job = (u64, u64, u64);

/// What a cut decided: the copies a segment's seats are waiting on, and the
/// `(group, seat, expert)` each one makes resident once it has landed.
#[derive(Debug, Default)]
struct Seating {
    jobs: Vec<Job>,
    landing: Vec<(usize, u32, u32)>,
}

#[derive(Debug)]
pub struct Tier {
    store: Store,
    source: Source,
    groups: Vec<Group>,
    pool: Pool,
    ring: Option<Ring>,
    of_routes: BTreeMap<u32, usize>,
    tally: tally::Tally,
    threads: usize,
    /// What the read in flight will make resident once it lands.
    landing: Vec<(usize, u32, u32)>,
    /// Rows in one lane from which a fire is a prefill one and takes the
    /// ring; 0 when this load has no ring.
    prefill_min: u32,
    /// Whether the fire open now is a prefill one, holding the ring.
    whole: bool,
    per_slot: u64,
    pairs: u64,
    policy: String,
    hint_of: BTreeMap<u32, ValueId>,
    predicted: Vec<Option<Vec<Vec<u32>>>>,
    /// Where this load's seat tables start in the store, and how far apart
    /// they are: group `g`'s table sits at `tables.0 + g * tables.1`.
    tables: (u64, u64),
    inflight: Option<std::thread::JoinHandle<Result<()>>>,
    file: Option<std::sync::Arc<std::fs::File>>,
    prefetch: bool,
    prefetch_k: usize,
    blocked: bool,
    dump: Option<std::io::BufWriter<std::fs::File>>,
}

const PREFETCH_K: usize = 4;

pub const PREDICTION_PREFIXES: [usize; 4] = [6, 8, 12, 16];

/// Threads the seat copies run on. llama.cpp's expert store measured 16
/// ahead of 8 on this class of SSD for whole-expert requests.
const SEAT_THREADS: usize = 16;

impl Tier {
    /// Say where group `at`'s experts sit, so the matmuls that follow this
    /// cut read the seats the tier landed them in. `seat_of` answers for a
    /// pool segment; a ring segment reads the half it was filled into, one
    /// borrowed seat per expert. An expert this segment does not route to is written as seat
    /// zero and never read: only the experts a row names are looked up.
    fn say_seats(&mut self, at: usize, half: Option<&[u32]>) -> Result<()> {
        let experts = self.groups[at].experts as usize;
        let mut table = Vec::with_capacity(experts * 4);
        for expert in 0..experts {
            let seat = match half {
                Some(half) => half.get(expert).copied().unwrap_or(0),
                None => self.groups[at].seat_of[expert].unwrap_or(0),
            };
            table.extend_from_slice(&seat.to_le_bytes());
        }
        let (base, stride) = self.tables;
        self.store.write(base + at as u64 * stride, &table)
    }

    /// `tables` is where this load's seat tables start in the store: one of
    /// `experts` `u32`s per group, `seat_table_stride` apart, saying where
    /// each expert sits. Every cut rewrites its group's before the matmuls
    /// read it.
    pub fn open(
        plan: &Plan,
        store: &Store,
        source: Source,
        offsets: &[u64],
        tables: u64,
    ) -> Result<Tier> {
        tally::open_log(plan.knobs.log.as_deref());
        let mut tier = Tier {
            store: store.clone(),
            source,
            groups: Vec::with_capacity(plan.groups.len()),
            pool: Pool::new(plan.slots),
            ring: (plan.ring > 0).then(|| Ring {
                seats: Vec::new(),
                experts: plan.groups.first().map_or(0, |group| group.experts),
                holds: [None, None],
                filling: None,
                fills: 0,
                copied: 0,
                read: 0,
                bytes: 0,
                lookups: 0,
                wait_ns: 0,
            }),
            of_routes: BTreeMap::new(),
            tally: tally::Tally::new(plan.report_every()),
            per_slot: plan.per_slot,
            pairs: plan.pairs,
            policy: plan.policy.clone(),
            hint_of: plan
                .groups
                .iter()
                .filter_map(|group| group.hint.map(|hint| (group.routes.0, hint)))
                .collect(),
            predicted: vec![None; plan.groups.len()],
            tables: (tables, plan.seat_table_stride()),
            inflight: None,
            file: None,
            prefetch: crate::diag::on().route_prefetch,
            dump: crate::diag::on()
                .route_dump
                .as_ref()
                .and_then(|path| std::fs::File::create(path).ok())
                .map(std::io::BufWriter::new),
            prefetch_k: crate::diag::on().prefetch_k.unwrap_or(PREFETCH_K),
            threads: crate::diag::on().seat_threads.unwrap_or(SEAT_THREADS),
            blocked: false,
            landing: Vec::new(),
            prefill_min: if plan.ring > 0 { plan.prefill_min } else { 0 },
            whole: false,
        };
        for (at, group) in plan.groups.iter().enumerate() {
            let bands = group
                .bands
                .iter()
                .map(|&band| {
                    let band = &plan.bands[band];
                    let from = tier.source.at(band.param).ok_or_else(|| {
                        Fault::Residency(format!(
                            "the seat source states no offset for band `{}` (param {}), \
                             which this plan streams — the residency plan and the bytes \
                             behind it were not built from each other",
                            band.name, band.param,
                        ))
                    })?;
                    Ok(Band {
                        name: band.name.clone(),
                        at: offsets[band.param],
                        from,
                        stride: band.stride,
                    })
                })
                .collect::<Result<Vec<Band>>>()?;
            tier.of_routes.insert(group.routes.0, at);
            tier.groups.push(Group {
                experts: group.experts,
                bands,
                seat_of: vec![None; group.experts as usize],
                ever: vec![false; group.experts as usize],
            });
        }
        // Every seat starts empty: an expert is read from disk the first time
        // a router names it, and not before.
        tier.file = tier
            .source
            .file()
            .and_then(|file| file.try_clone().ok())
            .map(std::sync::Arc::new);
        if let (Some(file), Bytes::Artifact(_)) = (&tier.file, &tier.source.bytes)
            && plan.uncached()
            && !uncached(file)
        {
            // The seats ARE the copy the engine reads; a second copy of every
            // expert in the page cache would only crowd out the pool.
            eprintln!(
                "engine-metal: F_NOCACHE on the artifact was refused; expert reads go \
                 through the page cache"
            );
        }
        tier.source.settle();
        Ok(tier)
    }

    /// Seat this segment's routed experts and rewrite its routing vector to
    /// seat numbers. `whole` says the fire is a prefill one (some lane has
    /// the ring's row count or more), which takes the ring when there is one.
    #[allow(clippy::too_many_arguments)]
    /// Open a fire over these lane row counts and say whether it is a
    /// prefill one. A prefill fire names nearly every expert of every layer,
    /// so it borrows two whole layers of seats from the pool and runs its
    /// routed matmuls over those, leaving the rest of the pool as the decode
    /// left it; the damage to the working set is those `2 * experts` seats,
    /// whatever the layer count.
    pub fn begin_fire(&mut self, lane_rows: &[u32]) -> Result<bool> {
        // The fire before this one is over: take its last prefetch, so the
        // seats it is landing in are not held against this fire's borrow. A
        // fire the shell refused after staging never ended; end it now,
        // rather than borrow twice.
        self.join_inflight()?;
        self.return_ring()?;
        let whole = self.prefill_min > 0
            && lane_rows.iter().any(|&rows| rows >= self.prefill_min)
            && self.ring.is_some();
        self.whole = whole;
        if whole {
            self.borrow_ring()?;
        }
        self.tally.open();
        Ok(whole)
    }

    /// Close the fire: the ring goes back to the pool, and the pool says
    /// what this fire cost it — a line in the CSV the load asked for, a
    /// trace line under `tier-trace`, and every so often the whole report.
    pub fn end_fire(&mut self, rows: u32, walk: std::time::Duration) -> Result<()> {
        self.return_ring()?;
        let record = self.tally.close(rows, walk);
        tally::log_fire(&record);
        if crate::diag::on().tier_trace {
            eprintln!(
                "tier: fire of {} row(s): {} seat copies over {} cuts, {} hits / {} \
                 misses, {:.1} MiB read; cuts {:.1} ms (copies {:.1} ms, waiting on \
                 the device {:.1} ms, {:.1} ms of device time); walk {:.1} ms",
                record.rows,
                record.copies,
                record.cuts,
                record.hits,
                record.misses,
                record.bytes_read as f64 / (1u64 << 20) as f64,
                record.cut_ms,
                record.copy_ms,
                record.wait_ms,
                record.gpu_cut_ms,
                record.walk_ms,
            );
        }
        if self.tally.due() {
            eprintln!(
                "{}; {} fires, final frames {:.1} ms on the device",
                self.report(),
                self.tally.fires(),
                self.tally.tail_ns as f64 / 1e6
            );
        }
        Ok(())
    }

    /// The borrowed seats go back to the pool holding the last two layers
    /// they were filled with, at the cold end of the LRU, where a decode may
    /// hit them but an eviction takes them first.
    fn return_ring(&mut self) -> Result<()> {
        if !self.whole {
            return Ok(());
        }
        self.whole = false;
        self.ring_join(None)?;
        let Some(ring) = self.ring.as_mut() else {
            return Ok(());
        };
        let experts = ring.experts as usize;
        let seats = std::mem::take(&mut ring.seats);
        let holds = std::mem::take(&mut ring.holds);
        if seats.len() < 2 * experts {
            return Ok(());
        }
        for (half, held) in holds.iter().enumerate() {
            for (expert, &seat) in seats[half * experts..][..experts].iter().enumerate() {
                self.pool.hold(seat, Hold::Free);
                match held {
                    // the half holds this layer, expert by expert
                    Some(group) => {
                        self.assign(*group, seat, expert as u32);
                        self.pool.unlink(seat);
                        self.pool.push_front(seat);
                    }
                    // never filled: nothing sits in it
                    None => self.pool.strip(seat),
                }
            }
        }
        Ok(())
    }

    /// Take `2 * experts` seats for this fire's ring: the ones never used
    /// first, then the coldest, exactly as a miss would.
    fn borrow_ring(&mut self) -> Result<()> {
        let experts = self.ring.as_ref().map_or(0, |ring| ring.experts);
        let mut seats = Vec::with_capacity(2 * experts as usize);
        for _ in 0..2 * experts {
            match self.take(0) {
                Ok(seat) => {
                    self.pool.hold(seat, Hold::Fire);
                    seats.push(seat);
                }
                // Give back what this borrow took before refusing, or the
                // pool loses those seats for the rest of the load.
                Err(why) => {
                    for seat in seats {
                        self.pool.strip(seat);
                    }
                    self.whole = false;
                    return Err(why);
                }
            }
        }
        if let Some(ring) = self.ring.as_mut() {
            ring.seats = seats;
            ring.holds = [None, None];
        }
        Ok(())
    }

    pub fn segment(
        &mut self,
        arena: &mut Buffer,
        handles: &Handles,
        routes: ValueId,
        rect: Tensor,
        hint: Option<Tensor>,
        span: MaskSpan,
    ) -> Result<()> {
        let Some(&at) = self.of_routes.get(&routes.0) else {
            return Ok(());
        };
        let started = std::time::Instant::now();
        let out = self.segment_at(at, arena, handles, routes, rect, hint, span);
        self.tally.cut_ns += started.elapsed().as_nanos() as u64;
        out
    }

    #[must_use]
    pub fn hint_for(&self, routes: ValueId) -> Option<ValueId> {
        self.hint_of.get(&routes.0).copied()
    }

    fn read_rows(
        arena: &mut Buffer,
        handles: &Handles,
        rect: Tensor,
        span: MaskSpan,
        what: &str,
    ) -> Result<Vec<Vec<i32>>> {
        let width = usize::try_from(rect.width).unwrap_or(usize::MAX);
        let row = handles.get(rect.buf).ok_or_else(|| Fault::Unbound {
            what: format!(
                "handle {}, {what}, which this fire minted no row for",
                rect.buf
            ),
        })?;
        let first = row.offset() + u64::from(span.row_offset) * rect.width as u64 * 4;
        let mut raw = vec![0u8; span.rows as usize * width * 4];
        arena.read(first, &mut raw)?;
        Ok(raw
            .chunks_exact(width * 4)
            .map(|row| {
                row.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|e| i32::from_le_bytes([e[0], e[1], e[2], e[3]]))
                    .collect()
            })
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    fn predict(
        &mut self,
        at: usize,
        arena: &mut Buffer,
        handles: &Handles,
        routes: ValueId,
        rect: Tensor,
        hint: Option<Tensor>,
        span: MaskSpan,
    ) -> Result<()> {
        if let Some(predicted) = self.predicted[at].take() {
            let truth = Self::read_rows(arena, handles, rect, span, "a routing vector")?;
            for (row, ranked) in truth.iter().zip(predicted.iter()) {
                for &id in row {
                    if id < 0 {
                        continue;
                    }
                    let expert = id as u32;
                    if expert >= self.groups[at].experts {
                        continue;
                    }
                    let seated = self.groups[at].seat_of[expert as usize].is_some();
                    self.tally.prediction.total += 1;
                    if !seated {
                        self.tally.prediction.misses += 1;
                    }
                    for (i, &k) in PREDICTION_PREFIXES.iter().enumerate() {
                        if ranked.iter().take(k).any(|&p| p == expert) {
                            self.tally.prediction.covered[i] += 1;
                            if !seated {
                                self.tally.prediction.saved[i] += 1;
                            }
                        }
                    }
                }
            }
        }
        if let (Some(hint), true) = (hint, at + 1 < self.predicted.len()) {
            let _ = routes;
            let rows = Self::read_rows(arena, handles, hint, span, "a route prediction")?;
            self.predicted[at + 1] = Some(
                rows.into_iter()
                    .map(|row| {
                        row.into_iter()
                            .filter(|&id| id >= 0)
                            .map(|id| id as u32)
                            .collect()
                    })
                    .collect(),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn segment_at(
        &mut self,
        at: usize,
        arena: &mut Buffer,
        handles: &Handles,
        routes: ValueId,
        rect: Tensor,
        hint: Option<Tensor>,
        span: MaskSpan,
    ) -> Result<()> {
        self.join_inflight()?;
        // The cut that brought us here waited for the previous segment's
        // matmuls, so nothing on the device reads those seats any more.
        self.pool.release(Hold::Segment);
        self.tally.segments += 1;
        if self.whole {
            return self.ring_at(at, arena, handles, routes, rect, span);
        }
        self.segment_rows(at, arena, handles, routes, rect, hint, span)
    }

    /// The routing vector's rows as raw bytes, and where they live.
    fn routing_bytes(
        arena: &mut Buffer,
        handles: &Handles,
        routes: ValueId,
        rect: Tensor,
        span: MaskSpan,
    ) -> Result<(u64, Vec<u8>)> {
        let width = u64::from(rect.width);
        let base = handles
            .get(rect.buf)
            .ok_or_else(|| Fault::Unbound {
                what: format!(
                    "handle {}, the routing vector of value {}, which this fire minted no \
                     row for",
                    rect.buf, routes.0
                ),
            })?
            .offset();
        let first = base + u64::from(span.row_offset) * width * 4;
        let count = usize::try_from(u64::from(span.rows) * width).unwrap_or(usize::MAX);
        let mut raw = vec![0u8; count * 4];
        arena.read(first, &mut raw)?;
        Ok((first, raw))
    }

    fn dump_rows(&mut self, at: usize, raw: &[u8], width: usize) {
        if let Some(dump) = &mut self.dump {
            use std::io::Write;
            for row in raw.chunks_exact(width * 4) {
                let ids: Vec<String> = row
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|e| i32::from_le_bytes([e[0], e[1], e[2], e[3]]).to_string())
                    .collect();
                let _ = writeln!(dump, "{at}\t{}", ids.join(" "));
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn segment_rows(
        &mut self,
        at: usize,
        arena: &mut Buffer,
        handles: &Handles,
        routes: ValueId,
        rect: Tensor,
        hint: Option<Tensor>,
        span: MaskSpan,
    ) -> Result<()> {
        if span.rows > 0 && (hint.is_some() || self.predicted[at].is_some()) {
            self.predict(at, arena, handles, routes, rect, hint, span)?;
        }
        if span.rows == 0 {
            return Ok(());
        }
        let (_, raw) = Self::routing_bytes(arena, handles, routes, rect, span)?;
        self.dump_rows(at, &raw, rect.width as usize);
        let seating = self.decide(at, &raw)?;
        self.land(seating)?;
        self.say_seats(at, None)?;
        if self.prefetch
            && let Some(rows) = self.predicted.get(at + 1).cloned().flatten()
        {
            self.prefetch(at + 1, &rows)?;
        }
        Ok(())
    }

    /// Where every expert this segment routes to will sit, and the copies
    /// that must land before its matmuls read them. Deciding moves no bytes
    /// and makes no expert resident: it takes seats and holds them, so a
    /// second row naming the same expert finds the same answer and no two
    /// experts are given one seat.
    fn decide(&mut self, at: usize, raw: &[u8]) -> Result<Seating> {
        let mut seating = Seating::default();
        let mut taken: BTreeMap<u32, u32> = BTreeMap::new();
        for entry in raw.as_chunks::<4>().0 {
            let id = i32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]);
            if id < 0 {
                continue;
            }
            let expert = id as u32;
            if expert >= self.groups[at].experts {
                return Err(Fault::Residency(format!(
                    "a routing vector names expert {expert} and `{}` declares {} of them; \
                     a seat cannot be found for an expert the router does not have. This \
                     is a routing vector read at the wrong instant — the segment cut ran \
                     against bytes some other segment wrote.",
                    self.groups[at].bands[0].name, self.groups[at].experts
                )));
            }
            if let Some(&seat) = taken.get(&expert) {
                self.pool.bump(seat);
                continue;
            }
            if let Some(seat) = self.groups[at].seat_of[expert as usize] {
                self.tally.hits += 1;
                self.pool.bump(seat);
                self.pool.hold(seat, Hold::Segment);
                taken.insert(expert, seat);
                continue;
            }
            self.tally.misses += 1;
            let seat = self.take(at)?;
            self.pool.hold(seat, Hold::Segment);
            self.jobs(at, seat, expert, &mut seating.jobs);
            seating.landing.push((at, seat, expert));
            taken.insert(expert, seat);
        }
        Ok(seating)
    }

    /// Move the bytes this seating asked for, then say the experts are
    /// resident. Nothing is resident before its bytes are there: a read that
    /// refuses leaves an empty seat, not a false hit.
    fn land(&mut self, seating: Seating) -> Result<()> {
        self.blocked = !seating.jobs.is_empty();
        if seating.jobs.is_empty() {
            return Ok(());
        }
        let started = std::time::Instant::now();
        match self.source.file() {
            Some(file) => self
                .store
                .write_from_file(file, &seating.jobs, self.threads)?,
            None => {
                for &(into, from, len) in &seating.jobs {
                    let from = usize::try_from(from).unwrap_or(usize::MAX);
                    let len = usize::try_from(len).unwrap_or(usize::MAX);
                    let source = self.source.get(from, len).ok_or_else(|| Fault::Ceiling {
                        what: "bytes of the seat source",
                        need: (from + len) as u64,
                        have: self.source.len(),
                    })?;
                    self.store.write(into, source)?;
                }
            }
        }
        self.tally.swaps += seating.jobs.len() as u64;
        self.tally.bytes_read += seating.jobs.iter().map(|&(_, _, len)| len).sum::<u64>();
        self.tally.copy_ns += started.elapsed().as_nanos() as u64;
        for (at, seat, expert) in seating.landing {
            self.assign(at, seat, expert);
        }
        Ok(())
    }

    /// A prefill segment: the layer's matmuls run over the ring half that
    /// holds the whole layer, and the next layer's half fills meanwhile.
    fn ring_at(
        &mut self,
        at: usize,
        arena: &mut Buffer,
        handles: &Handles,
        routes: ValueId,
        rect: Tensor,
        span: MaskSpan,
    ) -> Result<()> {
        let half = self.ring_ready(at)?;
        let (seats, experts) = {
            let ring = self.ring.as_ref().expect("the caller checked");
            let experts = ring.experts as usize;
            (
                ring.seats[half * experts..][..experts].to_vec(),
                ring.experts,
            )
        };
        self.say_seats(at, Some(&seats))?;
        if span.rows > 0 {
            let (_, raw) = Self::routing_bytes(arena, handles, routes, rect, span)?;
            self.dump_rows(at, &raw, rect.width as usize);
            let mut lookups = 0u64;
            for entry in raw.as_chunks::<4>().0 {
                let id = i32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]);
                if id < 0 {
                    continue;
                }
                let expert = id as u32;
                if expert >= experts {
                    return Err(Fault::Residency(format!(
                        "a routing vector names expert {expert} and `{}` declares {} of them; \
                         the prefill ring holds no such expert.",
                        self.groups[at].bands[0].name, experts
                    )));
                }
                lookups += 1;
            }
            if let Some(ring) = self.ring.as_mut() {
                ring.lookups += lookups;
            }
        }
        if at + 1 < self.groups.len() && !self.ring_has(at + 1) {
            self.ring_fill(at + 1, 1 - half)?;
        }
        Ok(())
    }

    fn ring_has(&self, group: usize) -> bool {
        self.ring.as_ref().is_some_and(|ring| {
            ring.holds.contains(&Some(group))
                || ring.filling.as_ref().is_some_and(|(g, _, _)| *g == group)
        })
    }

    /// Join the fill in flight, if any, and note which half it landed in.
    /// `wanted` is the group whose cut is waiting on it, when one is: only
    /// that wait is the fire's to account for.
    fn ring_join(&mut self, wanted: Option<usize>) -> Result<()> {
        let Some(ring) = self.ring.as_mut() else {
            return Ok(());
        };
        let Some((group, half, handle)) = ring.filling.take() else {
            return Ok(());
        };
        let started = std::time::Instant::now();
        let landed = handle.join().unwrap_or_else(|_| {
            Err(Fault::Residency(
                "the prefill ring's fill thread panicked".to_string(),
            ))
        });
        if wanted == Some(group) {
            ring.wait_ns += started.elapsed().as_nanos() as u64;
        }
        landed?;
        ring.holds[half] = Some(group);
        Ok(())
    }

    /// The half holding group `at`, filling it now if no fill was ahead.
    fn ring_ready(&mut self, at: usize) -> Result<usize> {
        self.ring_join(Some(at))?;
        let held = self
            .ring
            .as_ref()
            .and_then(|ring| ring.holds.iter().position(|g| *g == Some(at)));
        if let Some(half) = held {
            return Ok(half);
        }
        // Nothing was ahead of this layer (the first store layer of a fire):
        // fill and wait. Either half is free — the cut waited for the device —
        // but keep the one the previous layer used if it is still wanted.
        let half = usize::from(
            self.ring
                .as_ref()
                .is_some_and(|ring| at > 0 && ring.holds[0] == Some(at - 1)),
        );
        self.ring_fill(at, half)?;
        self.ring_join(Some(at))?;
        Ok(half)
    }

    /// Fill `half` of the ring with every expert of `group`, in the
    /// background: seats the pool holds are copied, the rest are read.
    fn ring_fill(&mut self, group: usize, half: usize) -> Result<()> {
        let Some(file) = self.file.clone() else {
            return Err(Fault::Residency(
                "the prefill ring reads whole layers from a file, and this tier's seat \
                 source is not one"
                    .to_string(),
            ));
        };
        let (seats, experts) = {
            let ring = self.ring.as_ref().expect("the caller checked");
            let experts = ring.experts as usize;
            (
                ring.seats[half * experts..][..experts].to_vec(),
                ring.experts,
            )
        };
        let mut copies: Vec<Job> = Vec::new();
        let mut reads: Vec<Job> = Vec::new();
        let (mut copied, mut read) = (0u64, 0u64);
        {
            let held = &self.groups[group];
            for expert in 0..experts.min(held.experts) {
                let dst = seats[expert as usize];
                match held.seat_of[expert as usize] {
                    Some(seat) => {
                        copied += 1;
                        for band in &held.bands {
                            copies.push((
                                band.at + u64::from(dst) * band.stride,
                                band.at + u64::from(seat) * band.stride,
                                band.stride,
                            ));
                        }
                    }
                    None => {
                        read += 1;
                        for band in &held.bands {
                            reads.push((
                                band.at + u64::from(dst) * band.stride,
                                band.from + u64::from(expert) * band.stride,
                                band.stride,
                            ));
                        }
                    }
                }
            }
        }
        let copiers = self.store.copiers(&copies)?;
        let writers = self.store.file_writers(&reads)?;
        let threads = self.threads;
        let bytes: u64 = reads.iter().map(|&(_, _, len)| len).sum();
        self.tally.bytes_read += bytes;
        self.tally.swaps += (copies.len() + reads.len()) as u64;
        if let Some(ring) = self.ring.as_mut() {
            ring.fills += 1;
            ring.copied += copied;
            ring.read += read;
            ring.bytes += bytes;
            ring.holds[half] = None;
        }
        let handle = std::thread::spawn(move || {
            for (copier, jobs) in &copiers {
                copier.copy(jobs, threads)?;
            }
            for (writer, jobs) in &writers {
                writer.pread(&file, jobs, threads)?;
            }
            Ok(())
        });
        if let Some(ring) = self.ring.as_mut() {
            ring.filling = Some((group, half, handle));
        }
        Ok(())
    }

    /// The copy jobs that land `expert` of group `at` in `seat`: one per band.
    fn jobs(&self, at: usize, seat: u32, expert: u32, into: &mut Vec<Job>) {
        for band in &self.groups[at].bands {
            into.push((
                band.at + u64::from(seat) * band.stride,
                band.from + u64::from(expert) * band.stride,
                band.stride,
            ));
        }
    }

    /// Read this seating's bytes on a thread of its own. The seats are held
    /// until `join_inflight` takes the read and makes them resident.
    fn spawn(&mut self, seating: Seating) -> Result<()> {
        let Some(file) = self.file.clone() else {
            return Ok(());
        };
        if seating.jobs.is_empty() {
            return Ok(());
        }
        self.tally.swaps += seating.jobs.len() as u64;
        self.tally.bytes_read += seating.jobs.iter().map(|&(_, _, len)| len).sum::<u64>();
        let writers = self.store.file_writers(&seating.jobs)?;
        self.landing = seating.landing;
        let threads = self.threads;
        self.inflight = Some(std::thread::spawn(move || {
            for (writer, jobs) in &writers {
                writer.pread(&file, jobs, threads)?;
            }
            Ok(())
        }));
        Ok(())
    }

    fn prefetch(&mut self, at: usize, rows: &[Vec<u32>]) -> Result<()> {
        if self.file.is_none() {
            return Ok(());
        }
        // The segment that just ran holds the seats the device is about to
        // read, so a prefetch takes seats around them.
        let mut wanted: Vec<u32> = Vec::new();
        for row in rows {
            for &expert in row.iter().take(self.prefetch_k) {
                if expert < self.groups[at].experts && !wanted.contains(&expert) {
                    wanted.push(expert);
                }
            }
        }
        let mut seating = Seating::default();
        for expert in wanted {
            if let Some(seat) = self.groups[at].seat_of[expert as usize] {
                self.pool.bump(seat);
                continue;
            }
            let Ok(seat) = self.take(at) else {
                break;
            };
            // Held until the copy lands: the seat is not resident yet, and
            // the cut that joins the copy is what makes it so.
            self.pool.hold(seat, Hold::Inflight);
            self.jobs(at, seat, expert, &mut seating.jobs);
            seating.landing.push((at, seat, expert));
            self.tally.prediction.prefetched += 1;
        }
        self.spawn(seating)
    }

    /// Take the read in flight, if any, and make what it landed resident.
    /// A read that refused leaves its seats empty and free.
    fn join_inflight(&mut self) -> Result<()> {
        let Some(handle) = self.inflight.take() else {
            return Ok(());
        };
        let landed = handle.join().unwrap_or_else(|_| {
            Err(Fault::Residency(
                "the route prefetch thread panicked".to_string(),
            ))
        });
        for (at, seat, expert) in std::mem::take(&mut self.landing) {
            self.pool.hold(seat, Hold::Free);
            if landed.is_ok() {
                self.assign(at, seat, expert);
            } else {
                self.pool.strip(seat);
            }
        }
        landed
    }

    /// A seat for a new expert: the LRU's victim, emptied of whoever sat
    /// there. The caller holds it and says what lands in it.
    fn take(&mut self, at: usize) -> Result<u32> {
        let seat = self.pool.victim().ok_or_else(|| {
            Fault::Residency(format!(
                "one segment of this fire routes to more than {} distinct experts of \
                 `{}`, and the shared pool seats {}: every seat is held by a matmul \
                 this same segment will run, by a copy landing in it, or by this \
                 fire's prefill ring. Every expert one segment reads must be resident \
                 at once — raise `{CACHE_ENV}` or `device_weight_budget`, or fire \
                 fewer tokens per step.",
                self.pool.slots, self.groups[at].bands[0].name, self.pool.slots
            ))
        })?;
        if let Some((group, held)) = self.pool.in_seat[seat as usize].take()
            && self.groups[group as usize].seat_of[held as usize] == Some(seat)
        {
            self.groups[group as usize].seat_of[held as usize] = None;
        }
        self.pool.bump(seat);
        Ok(seat)
    }

    /// Say that `expert` of group `at` is resident in `seat`, its bytes
    /// already there. An expert sits in one seat: the one it held before,
    /// if any, goes back to the pool empty.
    fn assign(&mut self, at: usize, seat: u32, expert: u32) {
        if let Some(old) = self.groups[at].seat_of[expert as usize].replace(seat)
            && old != seat
        {
            self.pool.strip(old);
        }
        self.pool.in_seat[seat as usize] = Some((at as u32, expert));
        if !self.groups[at].ever[expert as usize] {
            self.groups[at].ever[expert as usize] = true;
            self.tally.distinct += 1;
        }
    }

    #[must_use]
    pub fn residency(&self) -> Vec<GroupResidency> {
        self.groups
            .iter()
            .enumerate()
            .map(|(at, group)| {
                let mut in_seat = vec![None; self.pool.slots as usize];
                for (seat, held) in self.pool.in_seat.iter().enumerate() {
                    if let Some((of, expert)) = *held
                        && of as usize == at
                    {
                        in_seat[seat] = Some(expert);
                    }
                }
                GroupResidency {
                    name: group.bands[0].name.clone(),
                    experts: group.experts,
                    slots: self.pool.slots,
                    in_seat,
                }
            })
            .collect()
    }

    #[must_use]
    pub fn report(&self) -> CacheReport {
        let ring = self.ring.as_ref();
        CacheReport {
            policy: self.policy.clone(),
            slots: self.pool.slots,
            pairs: self.pairs,
            per_slot: self.per_slot,
            resident: self.pool.resident(),
            distinct: self.tally.distinct,
            hits: self.tally.hits,
            misses: self.tally.misses,
            bytes_read: self.tally.bytes_read,
            segments: self.tally.segments,
            copy_ns: self.tally.copy_ns,
            wait_ns: self.tally.wait_ns,
            gpu_ns: self.tally.gpu_ns,
            ring_fills: ring.map_or(0, |r| r.fills),
            ring_copied: ring.map_or(0, |r| r.copied),
            ring_read: ring.map_or(0, |r| r.read),
            ring_bytes: ring.map_or(0, |r| r.bytes),
            ring_wait_ns: ring.map_or(0, |r| r.wait_ns),
        }
    }

    #[must_use]
    pub fn motion(&self) -> (u64, u64) {
        (self.tally.swaps, self.tally.segments)
    }

    /// The device is idle until the next commit. If the last segment read
    /// from disk this one probably will too, and the wait is long enough
    /// that the clock falls: hold it up. If it did not, the gap is tens of
    /// microseconds and a heater kernel would only be in the way.
    fn heat(&self) {
        if self.blocked {
            crate::device::heater::arm();
        }
    }

    /// A cut waited this long on the device.
    pub fn note_wait(&mut self, ns: u64) {
        self.tally.wait_ns += ns;
        self.heat();
    }

    /// Device time of the fire's final frame, as it lands.
    pub fn note_tail(&mut self, ns: u64) {
        self.tally.tail_ns += ns;
        self.heat();
    }

    /// Device time of a cut frame, as the command buffer reports it.
    pub fn note_gpu(&mut self, ns: u64) {
        self.tally.gpu_ns += ns;
    }

    #[must_use]
    pub fn source(&self) -> Option<(u64, u64)> {
        self.source.backing()
    }

    #[must_use]
    pub fn source_kind(&self) -> &'static str {
        self.source.kind()
    }
}

impl Drop for Tier {
    fn drop(&mut self) {
        let _ = self.join_inflight();
        if let Some(ring) = self.ring.as_mut()
            && let Some((_, _, handle)) = ring.filling.take()
        {
            let _ = handle.join();
        }
        if self.tally.segments > 0 {
            eprintln!("{}", self.report());
        }
    }
}
