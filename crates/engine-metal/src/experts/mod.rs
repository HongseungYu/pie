// Replaced main's: the shared LRU pool and the tier that seats experts in it.

use std::collections::BTreeMap;

use kernels_metal::Tensor;
use model_exec::fire::MaskSpan;
use model_ir::ValueId;

use crate::device::{Buffer, Handles};
use crate::error::{Fault, Result};
use crate::weight_store::Store;

mod plan;
mod source;
mod tally;
mod trace;

use plan::gib;
pub use plan::{
    BandPlan, CACHE_ENV, HEADROOM_ENV, LOG_ENV, NOCACHE_ENV, PREFILL_ENV, Plan, Policy, REPORT_ENV,
    RegionPlan, free_ram, nocache, parse_bytes, policy, prefill_min, report_every, uncached,
};
use source::Bytes;
pub use source::Source;
pub use tally::{CacheReport, FireRecord, Prediction, log_fire};
use trace::found;
pub use trace::{Attachments, GroupPlan, GroupResidency, cuts, fan_out, pass_group};

#[derive(Clone, Debug)]
struct Passing {
    row_offset: u32,
    rows: u32,
    ids: Vec<i32>,
    groups: Vec<Vec<u32>>,
}

#[derive(Debug)]
struct Band {
    name: String,
    at: u64,
    from: u64,
    stride: u64,
}

/// One router's mixture: its bands and where each of its experts sits in
/// the shared pool, if anywhere.
#[derive(Debug)]
struct Group {
    experts: u32,
    bands: Vec<Band>,
    seat_of: Vec<Option<u32>>,
    ever: Vec<bool>,
}

const NONE: u32 = u32::MAX;

/// The shared LRU pool: one seat holds one `(group, expert)` of any group.
/// The seats in use sit on a doubly linked list ordered by recency, least
/// recently used at the head, so a victim is the head rather than a scan
/// over every seat (the scan cost llama.cpp's expert store 4 ms a step at
/// 11k seats). Seats never used yet wait on `free` and go first.
#[derive(Debug)]
struct Pool {
    slots: u32,
    in_seat: Vec<Option<(u32, u32)>>,
    pinned: Vec<bool>,
    prev: Vec<u32>,
    next: Vec<u32>,
    linked: Vec<bool>,
    head: u32,
    tail: u32,
    free: Vec<u32>,
}

impl Pool {
    fn new(slots: u32) -> Pool {
        Pool {
            slots,
            in_seat: vec![None; slots as usize],
            pinned: vec![false; slots as usize],
            prev: vec![NONE; slots as usize],
            next: vec![NONE; slots as usize],
            linked: vec![false; slots as usize],
            head: NONE,
            tail: NONE,
            free: (0..slots).rev().collect(),
        }
    }

    fn unlink(&mut self, seat: u32) {
        let s = seat as usize;
        if !self.linked[s] {
            return;
        }
        let (p, n) = (self.prev[s], self.next[s]);
        if p == NONE {
            self.head = n;
        } else {
            self.next[p as usize] = n;
        }
        if n == NONE {
            self.tail = p;
        } else {
            self.prev[n as usize] = p;
        }
        self.prev[s] = NONE;
        self.next[s] = NONE;
        self.linked[s] = false;
    }

    fn push_back(&mut self, seat: u32) {
        let s = seat as usize;
        debug_assert!(!self.linked[s], "a seat is on the list once");
        self.prev[s] = self.tail;
        self.next[s] = NONE;
        if self.tail == NONE {
            self.head = seat;
        } else {
            self.next[self.tail as usize] = seat;
        }
        self.tail = seat;
        self.linked[s] = true;
    }

    /// Most recently used, now.
    fn bump(&mut self, seat: u32) {
        self.unlink(seat);
        self.push_back(seat);
    }

    /// A seat to take: one never used, else the least recently used seat
    /// no one has pinned this segment.
    fn victim(&mut self) -> Option<u32> {
        if let Some(seat) = self.free.pop() {
            return Some(seat);
        }
        let mut at = self.head;
        while at != NONE {
            if !self.pinned[at as usize] {
                return Some(at);
            }
            at = self.next[at as usize];
        }
        None
    }

    fn resident(&self) -> u64 {
        self.in_seat.iter().filter(|held| held.is_some()).count() as u64
    }
}

/// Two full-layer buffers past the pool's seats, for prefill. A fire in
/// which some lane has `min_rows` rows or more names nearly every expert of
/// every layer, so its routed matmuls run over a buffer holding the whole
/// layer (seat = `base + half * experts + expert`, the LRU untouched), and
/// the next layer's buffer fills while this one computes: experts the pool
/// holds are copied from their seats, the rest are read. After llama.cpp's
/// expert store (`TAG_MOE_STORE_PREFILL`) and FreeToken's prefill.
#[derive(Debug)]
struct Ring {
    base: u32,
    experts: u32,
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

#[derive(Debug)]
pub struct Tier {
    store: Store,
    source: Source,
    groups: Vec<Group>,
    pool: Pool,
    ring: Option<Ring>,
    of_routes: BTreeMap<u32, usize>,
    swaps: u64,
    segments: u64,
    threads: usize,
    pending: Vec<(usize, u32, u32)>,
    hits: u64,
    misses: u64,
    bytes_read: u64,
    distinct: u64,
    per_slot: u64,
    pairs: u64,
    policy: String,
    cut_ns: u64,
    copy_ns: u64,
    wait_ns: u64,
    gpu_ns: u64,
    hint_of: BTreeMap<u32, ValueId>,
    predicted: Vec<Option<Vec<Vec<u32>>>>,
    passing: Vec<Option<Passing>>,
    /// Per group: the first seat of the ring half its routes were last
    /// rewritten to, `None` when they were seated into the pool.
    on_ring: Vec<Option<u32>>,
    inflight: Option<std::thread::JoinHandle<Result<()>>>,
    file: Option<std::sync::Arc<std::fs::File>>,
    prefetch: bool,
    prefetch_k: usize,
    blocked: bool,
    dump: Option<std::io::BufWriter<std::fs::File>>,
    prediction: Prediction,
}

const PREFETCH_K: usize = 4;

pub const PREDICTION_PREFIXES: [usize; 4] = [6, 8, 12, 16];

/// Threads the seat copies run on. llama.cpp's expert store measured 16
/// ahead of 8 on this class of SSD for whole-expert requests.
const SEAT_THREADS: usize = 16;

impl Tier {
    /// The ids a group's routing vector names once its cut has run, as
    /// `(count, first)`: `None` when this tier does not seat the group (the
    /// router's own expert ids); `experts` seats from the ring half's first
    /// seat after `ring_at`; else any of the pool's seats plus the ring's
    /// (`pass_at` and `segment_rows` seat wherever the LRU lands them).
    #[must_use]
    pub fn route_ids(&self, routes: ValueId, experts: u32) -> Option<(u32, u32)> {
        let at = *self.of_routes.get(&routes.0)?;
        Some(match self.on_ring[at] {
            Some(first) => (experts, first),
            None => (
                self.pool.slots + self.ring.as_ref().map_or(0, |ring| 2 * ring.experts),
                0,
            ),
        })
    }

    pub fn open(plan: &Plan, store: &Store, source: Source, offsets: &[u64]) -> Result<Tier> {
        let mut tier = Tier {
            store: store.clone(),
            source,
            groups: Vec::with_capacity(plan.groups.len()),
            pool: Pool::new(plan.slots),
            ring: (plan.ring > 0).then(|| Ring {
                base: plan.slots,
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
            swaps: 0,
            segments: 0,
            hits: 0,
            misses: 0,
            bytes_read: 0,
            distinct: 0,
            per_slot: plan.per_slot,
            pairs: plan.pairs,
            policy: plan.policy.clone(),
            cut_ns: 0,
            copy_ns: 0,
            wait_ns: 0,
            gpu_ns: 0,
            hint_of: plan
                .groups
                .iter()
                .filter_map(|group| group.hint.map(|hint| (group.routes.0, hint)))
                .collect(),
            predicted: vec![None; plan.groups.len()],
            passing: vec![None; plan.groups.len()],
            on_ring: vec![None; plan.groups.len()],
            prediction: Prediction::default(),
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
            pending: Vec::new(),
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
            && nocache()
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
    pub fn segment(
        &mut self,
        arena: &mut Buffer,
        handles: &Handles,
        routes: ValueId,
        rect: Tensor,
        hint: Option<Tensor>,
        span: MaskSpan,
        pass: (u32, u32),
        whole: bool,
    ) -> Result<u32> {
        let Some(&at) = self.of_routes.get(&routes.0) else {
            return Ok(1);
        };
        let started = std::time::Instant::now();
        let out = self.segment_at(at, arena, handles, routes, rect, hint, span, pass, whole);
        self.cut_ns += started.elapsed().as_nanos() as u64;
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
                    self.prediction.total += 1;
                    if !seated {
                        self.prediction.misses += 1;
                    }
                    for (i, &k) in PREDICTION_PREFIXES.iter().enumerate() {
                        if ranked.iter().take(k).any(|&p| p == expert) {
                            self.prediction.covered[i] += 1;
                            if !seated {
                                self.prediction.saved[i] += 1;
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
        pass: (u32, u32),
        whole: bool,
    ) -> Result<u32> {
        self.join_inflight()?;
        self.on_ring[at] = None;
        if whole && self.ring.is_some() {
            return self.ring_at(at, arena, handles, routes, rect, span);
        }
        // a fill a refused fire left in flight copies from seats this segment evicts
        self.ring_join(usize::MAX)?;
        if pass.1 > 1 {
            return self.pass_at(at, arena, handles, routes, rect, span, pass);
        }
        self.segment_rows(at, arena, handles, routes, rect, hint, span)?;
        Ok(1)
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
        // The cut that brought us here waited for the previous segment's
        // matmuls, so nothing on the device reads a seat any more: every pin
        // is released and this segment pins only what it reads.
        self.pool.pinned.fill(false);
        self.segments += 1;
        if span.rows == 0 {
            return Ok(());
        }
        let (first, mut raw) = Self::routing_bytes(arena, handles, routes, rect, span)?;
        self.dump_rows(at, &raw, rect.width as usize);
        for entry in raw.as_chunks_mut::<4>().0 {
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
            let seat = self.seat(at, expert)?;
            entry.copy_from_slice(&(seat as i32).to_le_bytes());
        }
        self.flush()?;
        arena.write(first, &raw)?;
        if self.prefetch
            && let Some(rows) = self.predicted.get(at + 1).cloned().flatten()
        {
            self.prefetch(at + 1, &rows)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn pass_at(
        &mut self,
        at: usize,
        arena: &mut Buffer,
        handles: &Handles,
        routes: ValueId,
        rect: Tensor,
        span: MaskSpan,
        (pass, passes): (u32, u32),
    ) -> Result<u32> {
        self.pool.pinned.fill(false);
        self.segments += 1;
        if span.rows == 0 {
            return Ok(0);
        }
        let width = u64::from(rect.width);
        let count = usize::try_from(u64::from(span.rows) * width).unwrap_or(usize::MAX);
        let fresh = pass == 0
            || self.passing[at]
                .as_ref()
                .is_none_or(|p| p.row_offset != span.row_offset || p.rows != span.rows);
        let first = if fresh {
            let (first, raw) = Self::routing_bytes(arena, handles, routes, rect, span)?;
            self.dump_rows(at, &raw, width as usize);
            let ids: Vec<i32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|e| i32::from_le_bytes([e[0], e[1], e[2], e[3]]))
                .collect();
            let mut order: Vec<u32> = Vec::new();
            for &id in &ids {
                if id < 0 {
                    continue;
                }
                let expert = id as u32;
                if expert >= self.groups[at].experts {
                    return Err(Fault::Residency(format!(
                        "a routing vector names expert {expert} and `{}` declares {} of them; \
                         a seat cannot be found for an expert the router does not have.",
                        self.groups[at].bands[0].name, self.groups[at].experts
                    )));
                }
                if !order.contains(&expert) {
                    order.push(expert);
                }
            }
            let seat_of = &self.groups[at].seat_of;
            let (mut leading, trailing): (Vec<u32>, Vec<u32>) = order
                .into_iter()
                .partition(|&e| seat_of[e as usize].is_some());
            leading.extend(trailing);
            let order = leading;
            let seats = pass_group(self.pool.slots) as usize;
            let groups = order.chunks(seats).map(<[u32]>::to_vec).collect();
            self.passing[at] = Some(Passing {
                row_offset: span.row_offset,
                rows: span.rows,
                ids,
                groups,
            });
            first
        } else {
            let row = handles.get(rect.buf).ok_or_else(|| Fault::Unbound {
                what: format!(
                    "handle {}, the routing vector of value {}, which this fire minted no \
                     row for",
                    rect.buf, routes.0
                ),
            })?;
            row.offset() + u64::from(span.row_offset) * width * 4
        };
        let (ids, group, next) = {
            let state = self.passing[at].as_ref().expect("stated just above");
            (
                state.ids.clone(),
                state.groups.get(pass as usize).cloned().unwrap_or_default(),
                state.groups.get(pass as usize + 1).cloned(),
            )
        };
        let _ = passes;
        let mut seat_of: BTreeMap<u32, i32> = BTreeMap::new();
        for &expert in &group {
            let seat = self.seat(at, expert)?;
            seat_of.insert(expert, seat as i32);
        }
        let mut raw = Vec::with_capacity(count * 4);
        let mut assigned = 0usize;
        for id in ids {
            let entry = if id < 0 {
                -1
            } else {
                seat_of.get(&(id as u32)).copied().unwrap_or(-1)
            };
            if entry >= 0 {
                assigned += 1;
            }
            raw.extend_from_slice(&entry.to_le_bytes());
        }
        if crate::diag::on().cut_trace {
            let seats: Vec<i32> = seat_of.values().copied().collect();
            eprintln!(
                "pass {pass} of {passes} on group {at}: group of {} experts (seats {:?}), {assigned} of {count} entries assigned, groups {}",
                group.len(),
                seats,
                self.passing[at].as_ref().map_or(0, |p| p.groups.len())
            );
        }
        self.flush()?;
        arena.write(first, &raw)?;
        if self.prefetch
            && let Some(next) = next
        {
            self.prefetch_group(at, &next)?;
        }
        Ok(self.passing[at]
            .as_ref()
            .map_or(0, |p| p.groups.len() as u32))
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
    ) -> Result<u32> {
        self.pool.pinned.fill(false);
        self.segments += 1;
        let half = self.ring_ready(at)?;
        let (base, experts) = {
            let ring = self.ring.as_ref().expect("the caller checked");
            (ring.base + half as u32 * ring.experts, ring.experts)
        };
        self.on_ring[at] = Some(base);
        if span.rows > 0 {
            let (first, mut raw) = Self::routing_bytes(arena, handles, routes, rect, span)?;
            self.dump_rows(at, &raw, rect.width as usize);
            let mut lookups = 0u64;
            for entry in raw.as_chunks_mut::<4>().0 {
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
                entry.copy_from_slice(&((base + expert) as i32).to_le_bytes());
                lookups += 1;
            }
            arena.write(first, &raw)?;
            if let Some(ring) = self.ring.as_mut() {
                ring.lookups += lookups;
            }
        }
        if at + 1 < self.groups.len() && !self.ring_has(at + 1) {
            self.ring_fill(at + 1, 1 - half)?;
        }
        Ok(1)
    }

    fn ring_has(&self, group: usize) -> bool {
        self.ring.as_ref().is_some_and(|ring| {
            ring.holds.contains(&Some(group))
                || ring.filling.as_ref().is_some_and(|(g, _, _)| *g == group)
        })
    }

    /// Join the fill in flight, if any, and note which half it landed in.
    fn ring_join(&mut self, wanted: usize) -> Result<()> {
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
        if group == wanted {
            ring.wait_ns += started.elapsed().as_nanos() as u64;
        }
        landed?;
        ring.holds[half] = Some(group);
        Ok(())
    }

    /// The half holding group `at`, filling it now if no fill was ahead.
    fn ring_ready(&mut self, at: usize) -> Result<usize> {
        self.ring_join(at)?;
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
        self.ring_join(at)?;
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
        let (base, experts) = {
            let ring = self.ring.as_ref().expect("the caller checked");
            (ring.base + half as u32 * ring.experts, ring.experts)
        };
        let mut copies: Vec<Job> = Vec::new();
        let mut reads: Vec<Job> = Vec::new();
        let (mut copied, mut read) = (0u64, 0u64);
        {
            let held = &self.groups[group];
            for expert in 0..experts.min(held.experts) {
                let dst = base + expert;
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
        self.bytes_read += bytes;
        self.swaps += (copies.len() + reads.len()) as u64;
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

    fn spawn(&mut self, jobs: Vec<Job>) -> Result<()> {
        let Some(file) = self.file.clone() else {
            return Ok(());
        };
        if jobs.is_empty() {
            return Ok(());
        }
        self.swaps += jobs.len() as u64;
        self.bytes_read += jobs.iter().map(|&(_, _, len)| len).sum::<u64>();
        let writers = self.store.file_writers(&jobs)?;
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
        // Pins stay: the segment that just ran pinned the seats the device is
        // about to read, and a prefetch may only evict around them.
        let mut wanted: Vec<u32> = Vec::new();
        for row in rows {
            for &expert in row.iter().take(self.prefetch_k) {
                if expert < self.groups[at].experts && !wanted.contains(&expert) {
                    wanted.push(expert);
                }
            }
        }
        let mut jobs: Vec<Job> = Vec::new();
        for expert in wanted {
            if let Some(seat) = self.groups[at].seat_of[expert as usize] {
                self.pool.bump(seat);
                continue;
            }
            let Ok(seat) = self.place(at, expert) else {
                break;
            };
            self.jobs(at, seat, expert, &mut jobs);
            self.prediction.prefetched += 1;
        }
        self.spawn(jobs)
    }

    fn prefetch_group(&mut self, at: usize, experts: &[u32]) -> Result<()> {
        if self.file.is_none() {
            return Ok(());
        }
        let mut jobs: Vec<Job> = Vec::new();
        for &expert in experts {
            if self.groups[at].seat_of[expert as usize].is_some() {
                continue;
            }
            let Ok(seat) = self.place(at, expert) else {
                break;
            };
            self.jobs(at, seat, expert, &mut jobs);
            self.prediction.prefetched += 1;
        }
        self.spawn(jobs)
    }

    fn join_inflight(&mut self) -> Result<()> {
        match self.inflight.take() {
            Some(handle) => handle.join().unwrap_or_else(|_| {
                Err(Fault::Residency(
                    "the route prefetch thread panicked".to_string(),
                ))
            }),
            None => Ok(()),
        }
    }

    fn touch(&mut self, seat: u32) {
        self.pool.bump(seat);
        self.pool.pinned[seat as usize] = true;
    }

    /// Give `expert` of group `at` a seat, evicting whoever held it. The
    /// caller issues the copy.
    fn place(&mut self, at: usize, expert: u32) -> Result<u32> {
        let seat = self.evict(at)?;
        if let Some((group, held)) = self.pool.in_seat[seat as usize].take() {
            self.groups[group as usize].seat_of[held as usize] = None;
        }
        self.pool.in_seat[seat as usize] = Some((at as u32, expert));
        self.groups[at].seat_of[expert as usize] = Some(seat);
        if !self.groups[at].ever[expert as usize] {
            self.groups[at].ever[expert as usize] = true;
            self.distinct += 1;
        }
        self.touch(seat);
        Ok(seat)
    }

    fn seat(&mut self, at: usize, expert: u32) -> Result<u32> {
        if let Some(seat) = self.groups[at].seat_of[expert as usize] {
            if !self.pool.pinned[seat as usize] {
                self.hits += 1;
            }
            self.touch(seat);
            return Ok(seat);
        }
        self.misses += 1;
        let seat = self.place(at, expert)?;
        self.pending.push((at, seat, expert));
        Ok(seat)
    }

    fn flush(&mut self) -> Result<()> {
        self.blocked = !self.pending.is_empty();
        if self.pending.is_empty() {
            return Ok(());
        }
        let started = std::time::Instant::now();
        let pending = std::mem::take(&mut self.pending);
        let mut jobs: Vec<Job> = Vec::with_capacity(pending.len() * 3);
        for &(at, seat, expert) in &pending {
            self.jobs(at, seat, expert, &mut jobs);
        }
        match self.source.file() {
            Some(file) => self.store.write_from_file(file, &jobs, self.threads)?,
            None => {
                for &(into, from, len) in &jobs {
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
        self.swaps += jobs.len() as u64;
        self.bytes_read += jobs.iter().map(|&(_, _, len)| len).sum::<u64>();
        self.copy_ns += started.elapsed().as_nanos() as u64;
        Ok(())
    }

    /// A seat to reuse: never used, else the least recently used one no one
    /// has pinned this segment.
    fn evict(&mut self, at: usize) -> Result<u32> {
        self.pool.victim().ok_or_else(|| {
            Fault::Residency(format!(
                "one segment of this fire routes to more than {} distinct experts of \
                 `{}`, and the shared pool seats {}: every seat is pinned by a matmul \
                 this same segment will run, so no seat can be reused. Every expert one \
                 segment reads must be resident at once — raise `{CACHE_ENV}` or \
                 `device_weight_budget`, or fire fewer tokens per step.",
                self.pool.slots, self.groups[at].bands[0].name, self.pool.slots
            ))
        })
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
            distinct: self.distinct,
            hits: self.hits,
            misses: self.misses,
            bytes_read: self.bytes_read,
            segments: self.segments,
            copy_ns: self.copy_ns,
            wait_ns: self.wait_ns,
            gpu_ns: self.gpu_ns,
            ring_fills: ring.map_or(0, |r| r.fills),
            ring_copied: ring.map_or(0, |r| r.copied),
            ring_read: ring.map_or(0, |r| r.read),
            ring_bytes: ring.map_or(0, |r| r.bytes),
            ring_wait_ns: ring.map_or(0, |r| r.wait_ns),
        }
    }

    #[must_use]
    pub fn motion(&self) -> (u64, u64) {
        (self.swaps, self.segments)
    }

    #[must_use]
    pub fn hits(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    #[must_use]
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    #[must_use]
    pub fn prediction(&self) -> Prediction {
        self.prediction
    }

    #[must_use]
    pub fn host_time(&self) -> (u64, u64, u64) {
        (self.cut_ns, self.copy_ns, self.wait_ns)
    }

    pub fn note_wait(&mut self, ns: u64) {
        self.wait_ns += ns;
    }

    /// Whether the last segment read from disk, so the next cut's host phase
    /// is likely to be long. The heater asks before it fires.
    #[must_use]
    pub fn blocking(&self) -> bool {
        self.blocked
    }

    /// Device time of a cut frame, as the command buffer reports it.
    pub fn note_gpu(&mut self, ns: u64) {
        self.gpu_ns += ns;
    }

    #[must_use]
    pub fn gpu_ns(&self) -> u64 {
        self.gpu_ns
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
        if self.segments > 0 {
            eprintln!("{}", self.report());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_parse() {
        assert_eq!(parse_bytes("4GiB"), Some(4 << 30));
        assert_eq!(parse_bytes("4G"), Some(4 << 30));
        assert_eq!(parse_bytes(" 512 MiB "), Some(512 << 20));
        assert_eq!(parse_bytes("1024"), Some(1024));
        assert_eq!(parse_bytes("1.5GiB"), Some(3 << 29));
        assert_eq!(parse_bytes("lots"), None);
    }

    #[test]
    fn free_ram_reads_on_apple() {
        if cfg!(target_vendor = "apple") {
            assert!(free_ram().is_some_and(|bytes| bytes > 0));
        }
    }

    #[test]
    fn the_pool_evicts_least_recently_used_first() {
        let mut pool = Pool::new(3);
        // Never-used seats go first, in order.
        assert_eq!(pool.victim(), Some(0));
        pool.bump(0);
        assert_eq!(pool.victim(), Some(1));
        pool.bump(1);
        assert_eq!(pool.victim(), Some(2));
        pool.bump(2);
        // Now the list decides: 0 is the oldest.
        assert_eq!(pool.victim(), Some(0));
        pool.bump(0);
        assert_eq!(pool.victim(), Some(1));
        // A pinned head is skipped.
        pool.pinned[1] = true;
        assert_eq!(pool.victim(), Some(2));
        pool.pinned[2] = true;
        pool.pinned[0] = true;
        assert_eq!(pool.victim(), None);
        // Unlinking the tail and the middle keeps the list whole.
        pool.pinned.fill(false);
        pool.unlink(0);
        pool.unlink(2);
        assert_eq!(pool.head, 1);
        assert_eq!(pool.tail, 1);
        pool.unlink(1);
        assert_eq!(pool.head, NONE);
        assert_eq!(pool.tail, NONE);
    }
}
