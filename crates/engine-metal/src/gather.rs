use std::collections::{BTreeMap, BTreeSet, HashMap};

use model_compiler::CompiledModel;
use model_exec::fire::MaskSpan;
use model_ir::ops::{Attention, Layout};
use model_ir::{Def, Operation, Trace, ValueId};

use kernels_metal::attn::ple::reference;

use crate::device::Handles;
use crate::device::alloc::Buffer;
use crate::error::{Fault, Result};
use crate::experts::{Attachments, Source};
use crate::weight_store::Store;
use kernels_metal::Tensor;

/// The n-gram hasher's constants and where its state lives, so the host
/// can name a fire's rows before the device does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hasher {
    pub eos: i32,
    pub mults: Vec<u64>,
    pub primes: Vec<u64>,
    pub offsets: Vec<u64>,
    pub heads_per_ngram: usize,
    /// The recurrent cache row holding each slot's last `span` token ids.
    pub state_row: usize,
}

impl Hasher {
    fn span(&self) -> usize {
        self.mults.len().saturating_sub(1)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Table {
    pub name: String,
    pub params: Vec<usize>,
    pub rows: u64,
    pub seats: u32,
    pub strides: Vec<u64>,
    pub hasher: Option<Hasher>,
    host_of: BTreeMap<usize, u64>,
    host_bytes: u64,
    device_bytes: u64,
}

impl Table {
    #[must_use]
    pub fn stored(&self) -> u64 {
        self.strides
            .iter()
            .map(|stride| (self.rows * stride).next_multiple_of(crate::weights::ALIGN))
            .sum()
    }

    #[must_use]
    pub fn slab(&self) -> u64 {
        self.device_bytes
    }
}

#[derive(Debug, Clone, Default)]
pub struct Plan {
    table: Option<Table>,
}

impl Plan {
    pub fn of(
        trace: &Trace,
        planes: &Attachments,
        budget: Option<u64>,
        max_tokens: u32,
    ) -> Result<Plan> {
        let Some(budget) = budget else {
            return Ok(Plan::default());
        };
        let bytes = crate::weights::plane_bytes(trace)?;
        let full: u64 = bytes
            .iter()
            .map(|plane| plane.next_multiple_of(crate::weights::ALIGN))
            .sum();
        if budget >= full {
            return Ok(Plan::default());
        }

        let mut heads: Option<usize> = None;
        let mut hasher: Option<Hasher> = None;
        for node in &trace.nodes {
            let Operation::Attention(op) = &node.op else {
                continue;
            };
            let (primes, state, eos, mults, offsets, heads_per_ngram) = match op {
                Attention::PleNgramIds {
                    primes,
                    state,
                    eos,
                    mults,
                    offsets,
                    heads_per_ngram,
                    ..
                }
                | Attention::PleNgramIdsChunked {
                    primes,
                    state,
                    eos,
                    mults,
                    offsets,
                    heads_per_ngram,
                    ..
                } => (primes, state, eos, mults, offsets, heads_per_ngram),
                _ => continue,
            };
            heads = Some(heads.map_or(primes.len(), |held: usize| held.max(primes.len())));
            if hasher.is_none()
                && let Some(Def::Cache(row)) = trace.values.get(state.0 as usize).map(|v| &v.def)
            {
                hasher = Some(Hasher {
                    eos: i32::try_from(*eos).unwrap_or(i32::MAX),
                    mults: mults.clone(),
                    primes: primes.clone(),
                    offsets: offsets.clone(),
                    heads_per_ngram: *heads_per_ngram as usize,
                    state_row: *row as usize,
                });
            }
        }

        let mut found: Option<Table> = None;
        for node in &trace.nodes {
            let Operation::Layout(Layout::EmbedConcat { table, .. }) = &node.op else {
                continue;
            };
            let codes = weight_of(trace, *table)?;
            if let Some(held) = &found {
                if held.params.first() == Some(&codes) {
                    continue;
                }
                return Err(Fault::Param {
                    name: trace.params[codes].name.clone(),
                    why: "is a second gathered table in one plan; this class holds one, \
                          because one is what exists — a second wants the group vocabulary \
                          the routed tier has and this one deliberately does not",
                });
            }
            let Some(heads) = heads else {
                return Err(Fault::Param {
                    name: trace.params[codes].name.clone(),
                    why: "is read by a concatenating gather in a plan that carries no PLE \
                          hasher; the gathered class is the hasher's demand shape and there \
                          is no static demand to serve without one",
                });
            };
            let mut params = vec![codes];
            params.extend(planes.get(&codes).into_iter().flatten().copied());
            let rows = trace.params[codes].shape.first().copied().unwrap_or(0);
            if rows == 0 {
                return Err(Fault::Param {
                    name: trace.params[codes].name.clone(),
                    why: "declares no rows, and a row slab over a table with no rows has no \
                          stride to seat",
                });
            }
            let mut strides = Vec::with_capacity(params.len());
            for &at in &params {
                let plane_rows = trace.params[at].shape.first().copied().unwrap_or(0);
                if plane_rows != rows {
                    return Err(Fault::Param {
                        name: trace.params[at].name.clone(),
                        why: "is a companion plane of a gathered table whose leading axis is \
                              not the table's; a seat is one row of every plane at the same \
                              index, and two row counts make that untrue",
                    });
                }
                strides.push(bytes[at] / rows);
            }
            let seats = u64::from(max_tokens).saturating_mul(heads as u64).min(rows);
            let seats = u32::try_from(seats).unwrap_or(u32::MAX).max(1);
            let mut host_of = BTreeMap::new();
            let mut host_bytes = 0u64;
            let mut device_bytes = 0u64;
            for (at, &param) in params.iter().enumerate() {
                host_of.insert(param, host_bytes);
                host_bytes += bytes[param];
                device_bytes +=
                    (u64::from(seats) * strides[at]).next_multiple_of(crate::weights::ALIGN);
            }
            found = Some(Table {
                name: trace.params[codes].name.clone(),
                params,
                rows,
                seats,
                strides,
                hasher: hasher.clone(),
                host_of,
                host_bytes,
                device_bytes,
            });
        }

        let Some(table) = found else {
            return Ok(Plan::default());
        };
        let gathered: BTreeSet<usize> = table.params.iter().copied().collect();
        let rest: u64 = bytes
            .iter()
            .enumerate()
            .filter(|(at, _)| !gathered.contains(at))
            .map(|(_, plane)| plane.next_multiple_of(crate::weights::ALIGN))
            .sum();
        if budget < table.device_bytes {
            return Err(Fault::Residency(format!(
                "`device_weight_budget` is {budget} bytes and `{}` alone is {} of them, so \
                 it is held CPU-side and read through a row slab of {} seats — which is \
                 {} bytes, and the budget does not hold even that. The slab is sized by the \
                 fire's row ceiling and not by the table's {} rows, so this number does not \
                 shrink: raise the budget past it, or lower `max_tokens`.",
                table.name, bytes[table.params[0]], table.seats, table.device_bytes, table.rows,
            )));
        }
        let _ = rest;
        Ok(Plan { table: Some(table) })
    }

    #[must_use]
    pub fn gathers(&self) -> bool {
        self.table.is_some()
    }

    #[must_use]
    pub fn table(&self) -> Option<&Table> {
        self.table.as_ref()
    }

    #[must_use]
    pub fn resident(&self, param: usize) -> Option<u32> {
        let table = self.table.as_ref()?;
        table.params.contains(&param).then_some(table.seats)
    }

    #[must_use]
    pub fn host_at(&self, param: usize) -> Option<u64> {
        self.table.as_ref()?.host_of.get(&param).copied()
    }

    #[must_use]
    pub fn params(&self) -> BTreeSet<usize> {
        self.table
            .as_ref()
            .map(|t| t.params.iter().copied().collect())
            .unwrap_or_default()
    }

    #[must_use]
    pub fn device_demand(&self) -> u64 {
        self.table.as_ref().map_or(0, |t| t.device_bytes)
    }

    #[must_use]
    pub fn source_bytes(&self) -> u64 {
        self.table.as_ref().map_or(0, |t| t.host_bytes)
    }

    #[must_use]
    pub fn host_bands(&self) -> BTreeMap<usize, u64> {
        self.table
            .as_ref()
            .map(|t| t.host_of.clone())
            .unwrap_or_default()
    }

    #[must_use]
    pub fn vocab(&self, param: usize) -> Option<u32> {
        let table = self.table.as_ref()?;
        (table.params.first() == Some(&param)).then_some(table.seats)
    }
}

fn weight_of(trace: &Trace, id: ValueId) -> Result<usize> {
    match trace.values.get(id.0 as usize).map(|decl| &decl.def) {
        Some(Def::Weight(w)) => Ok(*w as usize),
        _ => Err(Fault::Param {
            name: format!("value {}", id.0),
            why: "is read as a concatenating gather's table and is not a weight; a gathered \
                  plane is a `Def::Weight` row and nothing else resolves there",
        }),
    }
}

pub fn cuts(trace: &Trace, compiled: &CompiledModel) -> Result<Vec<Option<ValueId>>> {
    let mut out = Vec::with_capacity(compiled.template().len());
    for (at, region) in compiled.template().iter().enumerate() {
        let mut here: Option<ValueId> = None;
        for node in region.nodes.clone() {
            let Some(node) = trace.nodes.get(node as usize) else {
                continue;
            };
            let Operation::Attention(op) = &node.op else {
                continue;
            };
            let ids = match op {
                Attention::PleNgramIds { ngram_ids, .. }
                | Attention::PleNgramIdsChunked { ngram_ids, .. } => *ngram_ids,
                _ => continue,
            };
            if let Some(first) = here {
                if first != ids {
                    return Err(Fault::Residency(format!(
                        "region {at} holds two n-gram hashers landing different id vectors \
                         (values {} and {}), and a gathered load cuts its command buffer \
                         after EACH one — a single cut behind both would seat the first \
                         arm's rows and then read the second arm's raw table ids as seats. \
                         Raise `device_weight_budget` to hold the table whole, or bake an \
                         artifact whose regions carry one hasher each.",
                        first.0, ids.0
                    )));
                }
                continue;
            }
            here = Some(ids);
        }
        out.push(here);
    }
    Ok(out)
}

#[derive(Debug)]
struct Band {
    at: u64,
    from: u64,
    stride: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Residency {
    pub name: String,
    pub rows: u64,
    pub seats: u32,
    pub demanded: u32,
}

/// Threads a batch of row reads runs on: a row is one uncached block, and
/// sixteen of them in flight is what the expert tier found the drive likes.
const ROW_THREADS: usize = 16;

/// The rows of a gathered table a fire needs, seated in a device slab.
///
/// Rows come off the artifact by uncached `pread` at `from + row * stride`
/// (the offsets are known from the load), so their cost does not depend on
/// the page cache; the old path, out of the artifact's mapping, stays
/// behind `PIE_PLE_SOURCE=mmap`. With a hasher known, the host names a
/// fire's rows as the fire opens — the same window and the same slot state
/// the kernel will read — and lands them on a thread of their own while
/// the first frames run; the hasher's cut joins that read, and any id the
/// device names that the host did not is read there and then, counted as a
/// prefetch miss. A wrong guess costs a read, never a wrong row.
#[derive(Debug)]
pub struct Slab {
    store: Store,
    source: Source,
    file: Option<std::sync::Arc<std::fs::File>>,
    pread: bool,
    prefetch: bool,
    hasher: Option<Hasher>,
    bands: Vec<Band>,
    rows: u64,
    seats: u32,
    name: String,
    seat_of: HashMap<i32, u32>,
    /// Rows a prefetch is landing, resident once the read is joined.
    pending: HashMap<i32, u32>,
    inflight: Option<std::thread::JoinHandle<Result<u64>>>,
    in_seat: Vec<i32>,
    next: u32,
    fires: u64,
    copies: u64,
    /// Rows read from the artifact, and how many of them the hasher's cut
    /// had to read because no prefetch had landed them; both for the load,
    /// with the fire's own slice marked at `fire`.
    reads: u64,
    missed: u64,
    reads_at: u64,
    missed_at: u64,
    /// Host time spent inside the pread itself (prefetch thread and cut alike),
    /// apart from the seating and the join around it.
    read_ns: u64,
    read_ns_at: u64,
}

impl Slab {
    pub fn open(
        plan: &Plan,
        store: &Store,
        source: Source,
        offsets: &[u64],
        pread: bool,
        prefetch: bool,
    ) -> Result<Slab> {
        let table = plan.table.as_ref().ok_or_else(|| Fault::Param {
            name: "the gathered table".to_string(),
            why: "is opened over a plan that gathers nothing",
        })?;
        let mut bands = Vec::with_capacity(table.params.len());
        for (at, &param) in table.params.iter().enumerate() {
            let seat0 = offsets.get(param).copied().ok_or_else(|| Fault::Param {
                name: format!("param {param}"),
                why: "is a gathered plane the store laid down no offset for",
            })?;
            let from = source.at(param).ok_or_else(|| Fault::Param {
                name: format!("param {param}"),
                why: "is a gathered plane the CPU-side source states no row-0 offset for",
            })?;
            bands.push(Band {
                at: seat0,
                from,
                stride: table.strides[at],
            });
        }
        let mut source = source;
        source.settle();
        let file = source
            .file()
            .and_then(|file| file.try_clone().ok())
            .map(std::sync::Arc::new);
        let pread = pread && file.is_some();
        Ok(Slab {
            store: store.clone(),
            source,
            file,
            pread,
            prefetch: prefetch && pread && table.hasher.is_some(),
            hasher: table.hasher.clone(),
            bands,
            rows: table.rows,
            seats: table.seats,
            name: table.name.clone(),
            seat_of: HashMap::new(),
            pending: HashMap::new(),
            inflight: None,
            in_seat: vec![-1; table.seats as usize],
            next: 0,
            fires: 0,
            copies: 0,
            reads: 0,
            missed: 0,
            reads_at: 0,
            missed_at: 0,
            read_ns: 0,
            read_ns_at: 0,
        })
    }

    /// One line for the load log.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "n-gram rows: {} seats of `{}`, read by {}{}",
            self.seats,
            self.name,
            if self.pread { "uncached pread" } else { "the artifact's mapping" },
            if self.prefetch { ", prefetched as the fire opens" } else { "" },
        )
    }

    /// A fire opens: every seat is free again, and the fire's counters start.
    pub fn fire(&mut self) {
        let _ = self.join();
        self.seat_of.clear();
        self.pending.clear();
        self.in_seat.fill(-1);
        self.next = 0;
        self.fires += 1;
        self.reads_at = self.reads;
        self.missed_at = self.missed;
        self.read_ns_at = self.read_ns;
    }

    /// This fire's rows read, how many the prefetch missed, and the host time
    /// the reads themselves took.
    #[must_use]
    pub fn this_fire(&self) -> (u64, u64, f64) {
        (
            self.reads - self.reads_at,
            self.missed - self.missed_at,
            (self.read_ns - self.read_ns_at) as f64 / 1e6,
        )
    }

    /// Name and land the rows this fire will ask for, before the device
    /// does: `lanes` is each lane's slot and its token ids in fire order,
    /// `state` reads a slot's first cells of recurrent cache row `row` —
    /// the window the kernel hashes over. Does nothing without a hasher,
    /// a file, or the knob.
    pub fn prefetch(
        &mut self,
        lanes: &[(u32, &[u32])],
        state: impl Fn(usize, u32, usize) -> Option<Vec<i32>>,
    ) -> Result<()> {
        if !self.prefetch {
            return Ok(());
        }
        let Some(hasher) = self.hasher.clone() else {
            return Ok(());
        };
        let Some(file) = self.file.clone() else {
            return Ok(());
        };
        let h = reference::Hash {
            eos: hasher.eos,
            mults: &hasher.mults,
            primes: &hasher.primes,
            offsets: &hasher.offsets,
            heads_per_ngram: hasher.heads_per_ngram,
        };
        let span = hasher.span();
        let mut jobs: Vec<(u64, u64, u64)> = Vec::new();
        for &(slot, tokens) in lanes {
            if tokens.is_empty() {
                continue;
            }
            // A copy of the slot's window: the device's own state moves
            // when the kernel runs, not here.
            let Some(mut cells) = state(hasher.state_row, slot, span) else {
                continue;
            };
            let ids: Vec<i32> = tokens
                .iter()
                .map(|&t| i32::try_from(t).unwrap_or(hasher.eos))
                .collect();
            for id in reference::walk(&h, &ids, &mut cells) {
                if id < 0 || u64::from(id.unsigned_abs()) >= self.rows {
                    continue;
                }
                if self.seat_of.contains_key(&id) || self.pending.contains_key(&id) {
                    continue;
                }
                let Some(seat) = self.take() else {
                    break;
                };
                self.jobs(seat, id, &mut jobs);
                self.pending.insert(id, seat);
                self.in_seat[seat as usize] = id;
            }
        }
        if jobs.is_empty() {
            return Ok(());
        }
        self.reads += jobs.len() as u64 / self.bands.len().max(1) as u64;
        let writers = self.store.file_writers(&jobs)?;
        self.inflight = Some(std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let landed = crate::device::alloc::FileWriter::pread_many(&file, &writers, ROW_THREADS);
            landed.map(|()| started.elapsed().as_nanos() as u64)
        }));
        Ok(())
    }

    /// Take the prefetch in flight, if any: its rows are resident from here.
    fn join(&mut self) -> Result<()> {
        let Some(handle) = self.inflight.take() else {
            return Ok(());
        };
        let landed = handle.join().unwrap_or_else(|_| {
            Err(Fault::Residency(
                "the n-gram row prefetch thread panicked".to_string(),
            ))
        });
        let pending = std::mem::take(&mut self.pending);
        match landed {
            Ok(read_ns) => {
                self.read_ns += read_ns;
                self.copies += pending.len() as u64 * self.bands.len() as u64;
                self.seat_of.extend(pending);
                Ok(())
            }
            Err(why) => {
                // Nothing landed: the seats go back, and the cut reads them.
                for (_, seat) in pending {
                    self.in_seat[seat as usize] = -1;
                }
                Err(why)
            }
        }
    }

    /// A free seat, or none when the fire has named more distinct rows
    /// than the slab holds.
    fn take(&mut self) -> Option<u32> {
        if self.next >= self.seats {
            return None;
        }
        let seat = self.next;
        self.next += 1;
        Some(seat)
    }

    /// The read jobs that land `row` in `seat`, one per band.
    fn jobs(&self, seat: u32, row: i32, into: &mut Vec<(u64, u64, u64)>) {
        for band in &self.bands {
            into.push((
                band.at + u64::from(seat) * band.stride,
                band.from + u64::from(row.unsigned_abs()) * band.stride,
                band.stride,
            ));
        }
    }

    pub fn segment(
        &mut self,
        arena: &mut Buffer,
        handles: &Handles,
        ids: ValueId,
        rect: Tensor,
        span: MaskSpan,
    ) -> Result<()> {
        if span.rows == 0 {
            return Ok(());
        }
        let width = u64::from(rect.width);
        let base = {
            let row = handles.get(rect.buf).ok_or_else(|| Fault::Unbound {
                what: format!(
                    "handle {}, the n-gram id vector of value {}, which this fire minted no \
                     row for",
                    rect.buf, ids.0
                ),
            })?;
            row.offset()
        };
        let first = base + u64::from(span.row_offset) * width * 4;
        let count = usize::try_from(u64::from(span.rows) * width).unwrap_or(usize::MAX);
        let mut raw = vec![0u8; count * 4];
        arena.read(first, &mut raw)?;
        // The prefetch's rows first: what it landed is resident now, and
        // what the device names beyond that is read here.
        let prefetching = self.inflight.is_some();
        self.join()?;
        let mut jobs: Vec<(u64, u64, u64)> = Vec::new();
        let mut landing: Vec<(i32, u32)> = Vec::new();
        for entry in raw.as_chunks_mut::<4>().0 {
            let id = i32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]);
            if id < 0 {
                continue;
            }
            if u64::from(id.unsigned_abs()) >= self.rows {
                entry.copy_from_slice(&(self.seats as i32).to_le_bytes());
                continue;
            }
            let seat = match self.seat_of.get(&id).copied() {
                Some(seat) => seat,
                None => {
                    let seat = self.take().ok_or_else(|| self.full())?;
                    if self.pread {
                        self.jobs(seat, id, &mut jobs);
                    } else {
                        self.copy(seat, id)?;
                    }
                    self.seat_of.insert(id, seat);
                    self.in_seat[seat as usize] = id;
                    landing.push((id, seat));
                    seat
                }
            };
            entry.copy_from_slice(&(seat as i32).to_le_bytes());
        }
        if !jobs.is_empty() {
            let file = self.file.clone().ok_or_else(|| {
                Fault::Residency("n-gram rows by pread want the artifact's file".to_string())
            })?;
            let started = std::time::Instant::now();
            self.store.write_from_file(&file, &jobs, ROW_THREADS)?;
            self.read_ns += started.elapsed().as_nanos() as u64;
            self.copies += jobs.len() as u64;
        }
        self.reads += landing.len() as u64;
        if prefetching || self.prefetch {
            self.missed += landing.len() as u64;
        }
        arena.write(first, &raw)?;
        Ok(())
    }

    fn full(&self) -> Fault {
        Fault::Residency(format!(
            "this fire demands more than {} distinct rows of `{}`, which is every seat \
             the gathered row slab has. The slab is sized `max_tokens x heads` and a \
             fire may not present more token rows than `max_tokens`, so this is a \
             composition wider than the budget it was planned against: raise \
             `device_weight_budget` to hold the table whole, or lower `max_tokens`.",
            self.seats, self.name,
        ))
    }

    fn copy(&mut self, seat: u32, row: i32) -> Result<()> {
        for band in 0..self.bands.len() {
            let (into, from, stride) = {
                let band = &self.bands[band];
                (
                    band.at + u64::from(seat) * band.stride,
                    band.from + u64::from(row.unsigned_abs()) * band.stride,
                    band.stride,
                )
            };
            let from = usize::try_from(from).unwrap_or(usize::MAX);
            let len = usize::try_from(stride).unwrap_or(usize::MAX);
            let source = self.source.get(from, len).ok_or_else(|| Fault::Ceiling {
                what: "bytes of the gathered row source",
                need: (from + len) as u64,
                have: self.source.len(),
            })?;
            self.store.write(into, source)?;
            self.copies += 1;
        }
        Ok(())
    }

    #[must_use]
    pub fn residency(&self) -> Residency {
        Residency {
            name: self.name.clone(),
            rows: self.rows,
            seats: self.seats,
            demanded: self.next,
        }
    }

    #[must_use]
    pub fn motion(&self) -> (u64, u64) {
        (self.copies, self.fires)
    }

    #[must_use]
    pub fn source_kind(&self) -> &'static str {
        self.source.kind()
    }

    #[must_use]
    pub fn backing(&self) -> Option<(u64, u64)> {
        self.source.backing()
    }
}
