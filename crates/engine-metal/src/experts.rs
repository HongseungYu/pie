use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use kernels_metal::Tensor;
use model_compiler::CompiledModel;
use model_exec::fire::MaskSpan;
use model_ir::{Def, Linear, Operands, Operation, Trace, ValueId};

use crate::device::{Buffer, Handles};
use crate::error::{Fault, Result};
use crate::host_source::HostSource;
use crate::mapping::Mapping;
use crate::weight_store::Store;

pub type Attachments = BTreeMap<usize, Vec<usize>>;

/// How many routed-expert seats the shared pool holds.
///
/// The pool is ONE LRU over `(layer, expert)` pairs: every group (one router
/// and the bands it indexes) aliases the same `slots × stride` slabs, so a
/// seat can hold any layer's expert and the layers compete for seats by
/// recency rather than each holding a fixed share.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Every expert resident; no tier opens. `PIE_EXPERT_CACHE=off`.
    Off,
    /// The classic `[model] device_weight_budget`: as many seats as fit the
    /// byte budget once the dense planes are paid for.
    Budget(u64),
    /// Exactly this many seats. `PIE_EXPERT_CACHE=<n>`.
    Slots(u32),
    /// Free RAM at load, less a headroom, less the dense planes, all of it
    /// in seats. The default when nothing else is stated.
    Auto { headroom: u64 },
}

pub const CACHE_ENV: &str = "PIE_EXPERT_CACHE";
pub const HEADROOM_ENV: &str = "PIE_EXPERT_CACHE_HEADROOM";
pub const LOG_ENV: &str = "PIE_EXPERT_CACHE_LOG";
pub const NOCACHE_ENV: &str = "PIE_EXPERT_CACHE_NOCACHE";
pub const PREFILL_ENV: &str = "PIE_EXPERT_CACHE_PREFILL";
pub const REPORT_ENV: &str = "PIE_EXPERT_CACHE_REPORT";
/// A row routes to `top_k` of `experts`, so `rows` rows name about
/// `experts * (1 - exp(-rows * top_k / experts))` distinct experts: the
/// union passes 80% of the layer at `rows = 1.6 * experts / top_k`. Below
/// that a whole-layer read moves bytes no matmul asks for, so the ring's
/// own threshold is that, rounded to 2.
const PREFILL_UNION: u32 = 2;
const PREFILL_FLOOR: u32 = 32;
const DEFAULT_REPORT_EVERY: u64 = 256;

/// Rows in one lane from which a fire counts as prefill and runs its routed
/// matmuls over the ring (a whole layer at a time, filled one layer ahead).
/// `PIE_EXPERT_CACHE_PREFILL=<rows>` states it, `=0` drops the ring.
#[must_use]
pub fn prefill_min(experts: u32, top_k: u32) -> u32 {
    let derived = (PREFILL_UNION * experts)
        .div_ceil(top_k.max(1))
        .max(PREFILL_FLOOR);
    match std::env::var(PREFILL_ENV) {
        Ok(word) => match word.trim().to_ascii_lowercase().as_str() {
            "0" | "off" | "false" | "no" => 0,
            other => other.parse::<u32>().unwrap_or(derived),
        },
        Err(_) => derived,
    }
}

/// Fires between one `expert-cache:` line on stderr and the next; 0 leaves
/// only the shutdown line. `PIE_EXPERT_CACHE_REPORT=<n>`.
#[must_use]
pub fn report_every() -> u64 {
    std::env::var(REPORT_ENV)
        .ok()
        .and_then(|word| word.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_REPORT_EVERY)
}

/// Ask the kernel to serve reads through `file` from the disk, not the page
/// cache, and to cache nothing it reads (`F_NOCACHE`). Pages already cached
/// are still used; nothing new is added. True when the kernel agreed.
pub fn uncached(file: &std::fs::File) -> bool {
    #[cfg(target_vendor = "apple")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: fcntl on a live descriptor with an integer argument. With
        // `F_RDAHEAD` off a 32 KiB read stays a 32 KiB read.
        unsafe {
            libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) == 0
                && libc::fcntl(file.as_raw_fd(), libc::F_RDAHEAD, 0) == 0
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = file;
        false
    }
}

/// Whether the load's reads bypass the page cache (`F_NOCACHE`): the planes
/// landed at load and every seat copy after. On unless
/// `PIE_EXPERT_CACHE_NOCACHE=0`.
#[must_use]
pub fn nocache() -> bool {
    !matches!(
        std::env::var(NOCACHE_ENV).as_deref().map(str::trim),
        Ok("0" | "off" | "false" | "no")
    )
}
const DEFAULT_HEADROOM: u64 = 4 << 30;

/// The policy this process runs under: the environment first, then the
/// configured budget, then `Auto`.
pub fn policy(budget: Option<u64>) -> Result<Policy> {
    let headroom = match std::env::var(HEADROOM_ENV) {
        Ok(word) => parse_bytes(&word).ok_or_else(|| {
            Fault::Residency(format!(
                "`{HEADROOM_ENV}={word}` is not a byte count; write `4GiB`, `512MiB`, or a \
                 plain number of bytes"
            ))
        })?,
        Err(_) => DEFAULT_HEADROOM,
    };
    if let Ok(word) = std::env::var(CACHE_ENV) {
        let word = word.trim().to_string();
        return match word.to_ascii_lowercase().as_str() {
            "off" | "false" | "no" | "full" => Ok(Policy::Off),
            "" | "auto" | "on" | "true" | "yes" => Ok(Policy::Auto { headroom }),
            _ => word.parse::<u32>().map(Policy::Slots).map_err(|_| {
                Fault::Residency(format!(
                    "`{CACHE_ENV}={word}` is neither `off`, `auto`, nor a seat count"
                ))
            }),
        };
    }
    Ok(match budget {
        Some(budget) => Policy::Budget(budget),
        None => Policy::Auto { headroom },
    })
}

/// `4GiB`, `4G`, `4GB`, `512MiB`, `512M`, `1024` (bytes).
#[must_use]
pub fn parse_bytes(word: &str) -> Option<u64> {
    let word = word.trim();
    let split = word
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(word.len());
    let (number, unit) = word.split_at(split);
    let number: f64 = number.trim().parse().ok()?;
    let scale: f64 = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "k" | "kb" | "kib" => 1024.0,
        "m" | "mb" | "mib" => 1024.0 * 1024.0,
        "g" | "gb" | "gib" => 1024.0 * 1024.0 * 1024.0,
        "t" | "tb" | "tib" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    if number.is_nan() || number < 0.0 {
        return None;
    }
    Some((number * scale) as u64)
}

/// Host RAM the kernel would hand out without swapping: free, inactive and
/// speculative pages. Mach's own accounting, so it counts what `vm_stat` does.
#[must_use]
pub fn free_ram() -> Option<u64> {
    #[cfg(target_vendor = "apple")]
    {
        // SAFETY: `sysconf` takes a constant; `host_statistics64` fills a
        // `vm_statistics64` sized by the count libc states for it, and the
        // out-count is a local it may shrink.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page <= 0 {
            return None;
        }
        let mut stats: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
        let mut count: libc::mach_msg_type_number_t = libc::HOST_VM_INFO64_COUNT;
        #[allow(deprecated)] // libc's mach shims; `mach2` is not a dependency here.
        let rc = unsafe {
            libc::host_statistics64(
                libc::mach_host_self(),
                libc::HOST_VM_INFO64,
                (&raw mut stats).cast::<libc::integer_t>(),
                &raw mut count,
            )
        };
        if rc != libc::KERN_SUCCESS {
            return None;
        }
        let pages = u64::from(stats.free_count)
            + u64::from(stats.inactive_count)
            + u64::from(stats.speculative_count);
        Some(pages.saturating_mul(page as u64))
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        None
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

#[derive(Debug, Clone, Default)]
pub struct Plan {
    bands: Vec<BandPlan>,
    groups: Vec<GroupPlan>,
    regions: Vec<RegionPlan>,
    alias_of: BTreeMap<usize, usize>,
    resident_of: BTreeMap<usize, u32>,
    host_of: BTreeMap<usize, u64>,
    slots: u32,
    ring: u32,
    prefill_min: u32,
    per_slot: u64,
    pairs: u64,
    policy: String,
    device_bytes: u64,
    host_bytes: u64,
    gathered: crate::gather::Plan,
}

#[derive(Debug, Clone)]
pub struct BandPlan {
    pub param: usize,
    pub name: String,
    pub experts: u32,
    pub slots: u32,
    pub stride: u64,
    pub group: usize,
}

/// One slab of the shared pool: every band that sits at the same position
/// of its group and has the same per-expert stride. All of them alias one
/// `slots × stride` reservation, placed under the `head` param.
#[derive(Debug, Clone)]
pub struct RegionPlan {
    pub position: usize,
    pub stride: u64,
    pub head: usize,
    pub bands: Vec<usize>,
}

#[must_use]
pub fn pass_group(slots: u32) -> u32 {
    if !crate::diag::on().pass_half {
        return slots.max(1);
    }
    (slots / 2).max(1)
}

#[derive(Clone, Debug)]
struct Passing {
    row_offset: u32,
    rows: u32,
    ids: Vec<i32>,
    groups: Vec<Vec<u32>>,
}

#[derive(Debug, Clone)]
pub struct GroupPlan {
    pub routes: ValueId,
    pub experts: u32,
    pub slots: u32,
    pub bands: Vec<usize>,
    pub hint: Option<ValueId>,
}

impl Plan {
    /// A plan under an explicit budget, or full residency when there is
    /// none. Environment-free: this is the arm tests reason about.
    pub fn of(trace: &Trace, planes: &Attachments, budget: Option<u64>) -> Result<Plan> {
        let policy = match budget {
            Some(budget) => Policy::Budget(budget),
            None => Policy::Off,
        };
        Plan::under(trace, planes, policy, crate::gather::Plan::default())
    }

    /// The plan a load runs under: the environment's policy over the
    /// configured budget.
    pub fn beside(
        trace: &Trace,
        planes: &Attachments,
        budget: Option<u64>,
        gathered: crate::gather::Plan,
    ) -> Result<Plan> {
        Plan::under(trace, planes, policy(budget)?, gathered)
    }

    pub fn under(
        trace: &Trace,
        planes: &Attachments,
        policy: Policy,
        gathered: crate::gather::Plan,
    ) -> Result<Plan> {
        let held = gathered.params();
        let bytes = crate::weights::plane_bytes(trace)?;
        let full: u64 = bytes
            .iter()
            .enumerate()
            .filter(|(at, _)| !held.contains(at))
            .map(|(_, plane)| plane.next_multiple_of(crate::weights::ALIGN))
            .sum();
        let whole = |word: &str| Plan {
            device_bytes: full,
            gathered: gathered.clone(),
            policy: word.to_string(),
            ..Plan::default()
        };
        match policy {
            Policy::Off => return Ok(whole("off: every expert resident")),
            Policy::Budget(budget) if budget >= full => {
                return Ok(whole("budget holds the plan whole"));
            }
            Policy::Budget(_) | Policy::Slots(_) | Policy::Auto { .. } => {}
        }

        let (mut bands, mut groups) = found(trace, planes, &bytes)?;
        if bands.is_empty() {
            return match policy {
                Policy::Budget(budget) => Err(Fault::Residency(format!(
                    "`device_weight_budget` is {budget} bytes and this plan's weight table \
                     demands {full}. Nothing in it is a routed-expert bank, so there is no \
                     tier to hold less of: only routed experts stream (their seat is \
                     chosen after the router has run); dense planes do not. Raise the budget, or state `None` for uncapped."
                ))),
                _ => Ok(whole("no routed experts in this plan")),
            };
        }

        // The pool's regions: bands of one position and stride across every
        // group alias one slab. The head is the lowest param of the region,
        // so it is placed before any band that aliases it.
        let mut regions: Vec<RegionPlan> = Vec::new();
        for group in &groups {
            for (position, &band) in group.bands.iter().enumerate() {
                let stride = bands[band].stride;
                let param = bands[band].param;
                let at = match regions
                    .iter()
                    .position(|region| region.position == position && region.stride == stride)
                {
                    Some(at) => at,
                    None => {
                        regions.push(RegionPlan {
                            position,
                            stride,
                            head: param,
                            bands: Vec::new(),
                        });
                        regions.len() - 1
                    }
                };
                regions[at].bands.push(band);
                regions[at].head = regions[at].head.min(param);
            }
        }
        let mut alias_of: BTreeMap<usize, usize> = BTreeMap::new();
        for region in &regions {
            for &band in &region.bands {
                let param = bands[band].param;
                if param != region.head {
                    alias_of.insert(param, region.head);
                }
            }
        }
        let per_slot: u64 = regions.iter().map(|region| region.stride).sum();
        let seats = |n: u32| -> u64 {
            regions
                .iter()
                .map(|region| {
                    (u64::from(n) * region.stride).next_multiple_of(crate::weights::ALIGN)
                })
                .sum()
        };

        let streamed: BTreeSet<usize> = bands.iter().map(|band| band.param).collect();
        let dense: u64 = bytes
            .iter()
            .enumerate()
            .filter(|(at, _)| !streamed.contains(at) && !held.contains(at))
            .map(|(_, plane)| plane.next_multiple_of(crate::weights::ALIGN))
            .sum();
        let fan = groups
            .iter()
            .filter_map(|group| fan_out(trace, group.routes))
            .max()
            .unwrap_or(1)
            .max(1);
        let need = (1..=u32::MAX)
            .find(|&n| pass_group(n) >= fan)
            .unwrap_or(fan);
        let pairs: u64 = groups.iter().map(|group| u64::from(group.experts)).sum();
        let experts = groups[0].experts;
        let prefill_min = prefill_min(experts, fan);
        // The prefill ring: two whole layers of seats past the pool's own.
        let ring = if prefill_min > 0 { 2 * experts } else { 0 };
        let ceiling = u32::try_from(pairs).unwrap_or(u32::MAX).max(need);
        // The most seats, ring included, whose bytes fit `usable`.
        let largest = |usable: u64| -> u32 {
            let mut n = u32::try_from(usable / seats(1).max(1))
                .unwrap_or(u32::MAX)
                .min(ceiling + ring);
            while n > need + ring && seats(n) > usable {
                n -= 1;
            }
            n
        };
        let ring_note = || {
            if ring > 0 {
                format!(
                    " and a prefill ring of {ring} seats ({:.2} GiB; `{PREFILL_ENV}=0` drops \
                     it)",
                    gib(seats(ring))
                )
            } else {
                String::new()
            }
        };

        let (slots, word) = match policy {
            Policy::Off => return Ok(whole("off: every expert resident")),
            Policy::Budget(budget) => {
                let floor = dense + seats(need + ring);
                if budget < floor {
                    return Err(Fault::Residency(format!(
                        "`device_weight_budget` is {budget} bytes; this plan's DENSE planes \
                         demand {dense} resident and its {} routed bands need {need} expert \
                         seats in the shared pool on top (a row routes to {fan} experts and a \
                         pass seats half the pool){}, which is {floor}. Dense planes do not \
                         stream in this build, so the budget cannot be met by holding less. \
                         Raise it to at least {floor}, or state `None`.",
                        bands.len(),
                        ring_note(),
                    )));
                }
                let slots = largest(budget - dense) - ring;
                (slots, format!("budget {:.2} GiB", gib(budget)))
            }
            Policy::Slots(n) => {
                if n < need {
                    return Err(Fault::Residency(format!(
                        "`{CACHE_ENV}={n}` seats fewer experts than one row routes to: a \
                         row reads {fan} experts and a pass seats half the pool, so the pool \
                         needs at least {need} seats"
                    )));
                }
                (n.min(ceiling), format!("{CACHE_ENV}={n}"))
            }
            Policy::Auto { headroom } => {
                let free = free_ram().ok_or_else(|| {
                    Fault::Residency(
                        "the host's free memory could not be read, so the expert pool \
                         cannot be sized from it; state `PIE_EXPERT_CACHE=<seats>` or a \
                         `device_weight_budget`"
                            .to_string(),
                    )
                })?;
                let usable = free.saturating_sub(headroom).saturating_sub(dense);
                let total = largest(usable);
                if seats(total) > usable || total < need + ring {
                    return Err(Fault::Residency(format!(
                        "free RAM is {:.2} GiB; less the {:.2} GiB headroom and the {:.2} \
                         GiB of dense planes, {:.2} GiB is left, and the pool needs at least \
                         {need} seats of {:.2} MiB{}. Lower `{HEADROOM_ENV}`, or state \
                         `{CACHE_ENV}=<seats>` to size the pool by hand.",
                        gib(free),
                        gib(headroom),
                        gib(dense),
                        gib(usable),
                        per_slot as f64 / (1u64 << 20) as f64,
                        ring_note(),
                    )));
                }
                (
                    total - ring,
                    format!(
                        "auto: free {:.2} GiB - headroom {:.2} GiB - dense {:.2} GiB",
                        gib(free),
                        gib(headroom),
                        gib(dense)
                    ),
                )
            }
        };
        debug_assert!(slots >= need, "every arm above proved the floor");

        for band in &mut bands {
            band.slots = slots;
        }
        for group in &mut groups {
            group.slots = slots;
        }
        // The reservation a band's region is placed at: the pool and the ring.
        let resident_of = bands
            .iter()
            .map(|band| (band.param, slots + ring))
            .collect();
        let mut host_bytes = 0u64;
        let mut host_of = BTreeMap::new();
        for band in &bands {
            host_of.insert(band.param, host_bytes);
            host_bytes += u64::from(band.experts) * band.stride;
        }
        Ok(Plan {
            device_bytes: dense + seats(slots + ring),
            bands,
            groups,
            regions,
            alias_of,
            resident_of,
            host_of,
            slots,
            ring,
            prefill_min,
            per_slot,
            pairs,
            policy: word,
            host_bytes,
            gathered,
        })
    }

    #[must_use]
    pub fn gathered(&self) -> &crate::gather::Plan {
        &self.gathered
    }

    #[must_use]
    pub fn streams(&self) -> bool {
        !self.bands.is_empty()
    }

    #[must_use]
    pub fn bands(&self) -> &[BandPlan] {
        &self.bands
    }

    #[must_use]
    pub fn groups(&self) -> &[GroupPlan] {
        &self.groups
    }

    #[must_use]
    pub fn regions(&self) -> &[RegionPlan] {
        &self.regions
    }

    /// Seats in the shared pool.
    #[must_use]
    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// Bytes one seat holds across every region: one expert of every band.
    #[must_use]
    pub fn per_slot_bytes(&self) -> u64 {
        self.per_slot
    }

    /// `(layer, expert)` pairs the pool can be asked for.
    #[must_use]
    pub fn pairs(&self) -> u64 {
        self.pairs
    }

    /// Seats past the pool's own that hold the prefill ring (two whole
    /// layers), or 0 without one.
    #[must_use]
    pub fn ring(&self) -> u32 {
        self.ring
    }

    /// Rows in one lane from which a fire takes the ring.
    #[must_use]
    pub fn prefill_min(&self) -> u32 {
        self.prefill_min
    }

    /// The param whose reservation this streamed param aliases, if it is not
    /// its region's head.
    #[must_use]
    pub fn alias(&self, param: usize) -> Option<usize> {
        self.alias_of.get(&param).copied()
    }

    #[must_use]
    pub fn resident(&self, param: usize) -> Option<u32> {
        self.resident_of.get(&param).copied()
    }

    #[must_use]
    pub fn host_at(&self, param: usize) -> Option<u64> {
        self.host_of.get(&param).copied()
    }

    #[must_use]
    pub fn device_demand(&self) -> u64 {
        self.device_bytes + self.gathered.device_demand()
    }

    #[must_use]
    pub fn host_demand(&self) -> u64 {
        let _ = self.host_bytes;
        0
    }

    #[must_use]
    pub fn source_bytes(&self) -> u64 {
        self.host_bytes
    }

    /// One line for the load log.
    #[must_use]
    pub fn describe(&self) -> String {
        if !self.streams() {
            return format!("expert cache off ({})", self.policy);
        }
        let ring = if self.ring > 0 {
            format!(
                " + a prefill ring of {} seats ({:.2} GiB) from {} rows a lane",
                self.ring,
                gib(u64::from(self.ring) * self.per_slot),
                self.prefill_min
            )
        } else {
            String::new()
        };
        format!(
            "expert cache: {} seats x {:.2} MiB = {:.2} GiB shared by {} groups over {} \
             (layer, expert) pairs{ring} [{}]",
            self.slots,
            self.per_slot as f64 / (1u64 << 20) as f64,
            gib(u64::from(self.slots) * self.per_slot),
            self.groups.len(),
            self.pairs,
            self.policy,
        )
    }
}

fn found(
    trace: &Trace,
    planes: &Attachments,
    bytes: &[u64],
) -> Result<(Vec<BandPlan>, Vec<GroupPlan>)> {
    let mut arity: BTreeMap<u32, u32> = BTreeMap::new();
    let mut hints: BTreeMap<u32, ValueId> = BTreeMap::new();
    let mut order: Vec<ValueId> = Vec::new();
    for node in &trace.nodes {
        let Operation::Linear(op) = &node.op else {
            continue;
        };
        if let Linear::MoeTopkSqrtSoftplus {
            routes,
            hint: Some(hint),
            ..
        }
        | Linear::MoeTopkSigmoid {
            routes,
            hint: Some(hint),
            ..
        } = op
        {
            hints.insert(routes.0, *hint);
        }
        let (routes, experts) = match op {
            Linear::MoeTopkSoftmax {
                routes, experts, ..
            }
            | Linear::MoeTopkSoftmaxScaled {
                routes, experts, ..
            }
            | Linear::MoeTopkSigmoid {
                routes, experts, ..
            }
            | Linear::MoeTopkSqrtSoftplus {
                routes, experts, ..
            }
            | Linear::MoeHashRoute {
                routes, experts, ..
            } => (*routes, *experts),
            _ => continue,
        };
        if arity.insert(routes.0, experts).is_none() {
            order.push(routes);
        }
    }

    let mut of_group: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    let mut routers_of: BTreeMap<usize, BTreeSet<u32>> = BTreeMap::new();
    for node in &trace.nodes {
        let Operation::Linear(op) = &node.op else {
            continue;
        };
        let (routes, indexed) = match op {
            Linear::MoeMatmulSelect { bank, routes, .. }
            | Linear::MoeMatmulSelectQuant { bank, routes, .. } => (*routes, vec![*bank]),
            Linear::MoeMatmulSelectBias {
                bank, bias, routes, ..
            } => (*routes, vec![*bank, *bias]),
            Linear::MoeBiasSum { bias, routes, .. } => (*routes, vec![*bias]),
            _ => continue,
        };
        if !arity.contains_key(&routes.0) {
            return Err(Fault::Residency(format!(
                "value {} is read as a routing vector by `{}` and no router node of this \
                 plan writes it; the expert count a seat is divided out of is the \
                 ROUTER's field, so a routed read whose router this plan does not state \
                 cannot be seated at less than its declared size",
                routes.0,
                op.name(),
            )));
        }
        let seats = of_group.entry(routes.0).or_default();
        for id in indexed {
            let at = weight_of(trace, id)?;
            if !seats.contains(&at) {
                seats.push(at);
            }
            routers_of.entry(at).or_default().insert(routes.0);
            for &plane in planes.get(&at).into_iter().flatten() {
                if !seats.contains(&plane) {
                    seats.push(plane);
                }
                routers_of.entry(plane).or_default().insert(routes.0);
            }
        }
    }
    let shared: BTreeSet<usize> = routers_of
        .iter()
        .filter(|(_, routers)| routers.len() > 1)
        .map(|(&at, _)| at)
        .collect();

    let mut bands: Vec<BandPlan> = Vec::new();
    let mut groups: Vec<GroupPlan> = Vec::new();
    let mut owner: BTreeMap<usize, ValueId> = BTreeMap::new();
    let mut declared: Option<u32> = None;
    for routes in order {
        let Some(mut params) = of_group.remove(&routes.0) else {
            continue;
        };
        params.retain(|at| !shared.contains(at));
        if params.is_empty() {
            continue;
        }
        params.sort_unstable();
        let experts = arity[&routes.0];
        match declared {
            None => declared = Some(experts),
            Some(first) if first != experts => {
                return Err(Fault::Param {
                    name: trace.params[params[0]].name.clone(),
                    why: "is a routed band whose expert count differs from an earlier \
                          group of the same plan; one residency decision covers the plan, \
                          and two arities would make it two decisions",
                });
            }
            Some(_) => {}
        }
        let mut of_this = Vec::with_capacity(params.len());
        for at in params {
            if let Some(other) = owner.insert(at, routes)
                && other != routes
            {
                return Err(Fault::Param {
                    name: trace.params[at].name.clone(),
                    why: "is expert-indexed by two different routing vectors; a seat \
                              number means one group's seat, and a band shared between two \
                              groups would be re-indexed twice",
                });
            }
            let param = &trace.params[at];
            let leading = u32::try_from(param.shape.first().copied().unwrap_or(0)).unwrap_or(0);
            if leading != experts || param.shape.len() < 2 {
                return Err(Fault::Param {
                    name: param.name.clone(),
                    why: "is read as a routed expert band and does not declare \
                          `[experts, ...]` at the router's own expert count; a seat stride \
                          cannot be divided out of it",
                });
            }
            let plane = bytes[at];
            if plane == 0 || !plane.is_multiple_of(u64::from(experts)) {
                return Err(Fault::Param {
                    name: param.name.clone(),
                    why: "is a routed expert band whose bytes do not divide by its expert \
                          count — the experts of one band are not equal, and the seat \
                          arithmetic the tier does would be wrong rather than refused",
                });
            }
            of_this.push(bands.len());
            bands.push(BandPlan {
                param: at,
                name: param.name.clone(),
                experts,
                slots: experts,
                stride: plane / u64::from(experts),
                group: groups.len(),
            });
        }
        groups.push(GroupPlan {
            routes,
            experts,
            slots: experts,
            bands: of_this,
            hint: hints.get(&routes.0).copied(),
        });
    }
    Ok((bands, groups))
}

#[must_use]
pub fn fan_out(trace: &Trace, routes: ValueId) -> Option<u32> {
    trace.nodes.iter().find_map(|node| match &node.op {
        Operation::Linear(Linear::MoeTopkSoftmax {
            routes: r, top_k, ..
        })
        | Operation::Linear(Linear::MoeTopkSoftmaxScaled {
            routes: r, top_k, ..
        })
        | Operation::Linear(Linear::MoeTopkSigmoid {
            routes: r, top_k, ..
        })
        | Operation::Linear(Linear::MoeTopkSqrtSoftplus {
            routes: r, top_k, ..
        })
        | Operation::Linear(Linear::MoeHashRoute {
            routes: r, top_k, ..
        }) if *r == routes => Some(*top_k),
        _ => None,
    })
}

fn weight_of(trace: &Trace, id: ValueId) -> Result<usize> {
    match trace.values.get(id.0 as usize).map(|decl| &decl.def) {
        Some(Def::Weight(w)) => Ok(*w as usize),
        _ => Err(Fault::Param {
            name: format!("value {}", id.0),
            why: "is read at a routed matmul's expert-indexed port and is not a weight; a \
                  band is a `Def::Weight` row and nothing else resolves there",
        }),
    }
}

pub fn cuts(trace: &Trace, compiled: &CompiledModel, plan: &Plan) -> Result<Vec<Option<ValueId>>> {
    let streams = plan.streams();
    let streamed: BTreeSet<u32> = plan.groups.iter().map(|group| group.routes.0).collect();
    let mut out = Vec::with_capacity(compiled.template().len());
    for (at, region) in compiled.template().iter().enumerate() {
        let mut here: Option<ValueId> = None;
        for node in region.nodes.clone() {
            let Some(node) = trace.nodes.get(node as usize) else {
                continue;
            };
            let Operation::Linear(op) = &node.op else {
                continue;
            };
            let routes = match op {
                Linear::MoeTopkSoftmax { routes, .. }
                | Linear::MoeTopkSoftmaxScaled { routes, .. }
                | Linear::MoeTopkSigmoid { routes, .. }
                | Linear::MoeTopkSqrtSoftplus { routes, .. }
                | Linear::MoeHashRoute { routes, .. } => *routes,
                _ => continue,
            };
            if !streamed.contains(&routes.0) {
                continue;
            }
            if let Some(first) = here {
                if first != routes && streams {
                    return Err(Fault::Residency(format!(
                        "region {at} holds two routers (values {} and {}), and a streamed \
                         load cuts its command buffer after EACH one — a single cut behind \
                         both would encode the first mixture's matmuls against seats the \
                         host had not swapped yet. Raise `device_weight_budget` to hold \
                         this plan whole, or bake an artifact whose regions carry one \
                         mixture each.",
                        first.0, routes.0
                    )));
                }
                continue;
            }
            here = Some(routes);
        }
        out.push(here);
    }
    Ok(out)
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupResidency {
    pub name: String,
    pub experts: u32,
    pub slots: u32,
    pub in_seat: Vec<Option<u32>>,
}

#[derive(Debug)]
enum Bytes {
    Landed(HostSource),
    Artifact(Arc<Mapping>),
}

#[derive(Debug)]
pub struct Source {
    bytes: Bytes,
    bands: BTreeMap<usize, u64>,
}

impl Source {
    #[must_use]
    pub fn landed(plan: &Plan, host: HostSource) -> Source {
        Source {
            bytes: Bytes::Landed(host),
            bands: plan.host_of.clone(),
        }
    }

    #[must_use]
    pub fn from_host(host: HostSource, bands: BTreeMap<usize, u64>) -> Source {
        Source {
            bytes: Bytes::Landed(host),
            bands,
        }
    }

    #[must_use]
    pub fn artifact(map: Arc<Mapping>, bands: BTreeMap<usize, u64>) -> Source {
        Source {
            bytes: Bytes::Artifact(map),
            bands,
        }
    }

    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self.bytes {
            Bytes::Landed(_) => "landed",
            Bytes::Artifact(_) => "artifact",
        }
    }

    #[must_use]
    pub fn backing(&self) -> Option<(u64, u64)> {
        match &self.bytes {
            Bytes::Landed(host) => host.backing(),
            Bytes::Artifact(map) => Some((map.backing()?, map.links()?)),
        }
    }

    pub(crate) fn at(&self, param: usize) -> Option<u64> {
        self.bands.get(&param).copied()
    }

    pub(crate) fn file(&self) -> Option<&std::fs::File> {
        match &self.bytes {
            Bytes::Landed(host) => host.file(),
            Bytes::Artifact(map) => Some(map.file()),
        }
    }

    pub(crate) fn get(&self, from: usize, len: usize) -> Option<&[u8]> {
        let all: &[u8] = match &self.bytes {
            Bytes::Landed(host) => host,
            Bytes::Artifact(map) => map,
        };
        all.get(from..from.checked_add(len)?)
    }

    pub(crate) fn len(&self) -> u64 {
        match &self.bytes {
            Bytes::Landed(host) => host.len() as u64,
            Bytes::Artifact(map) => map.len(),
        }
    }

    pub(crate) fn settle(&mut self) {
        match &mut self.bytes {
            Bytes::Landed(host) => host.settle(),
            Bytes::Artifact(_) => {}
        }
    }
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
    inflight: Option<std::thread::JoinHandle<Result<()>>>,
    file: Option<std::sync::Arc<std::fs::File>>,
    prefetch: bool,
    prefetch_k: usize,
    blocked: bool,
    dump: Option<std::io::BufWriter<std::fs::File>>,
    prediction: Prediction,
}

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

fn fire_log() -> Option<&'static FireLog> {
    static LOG: std::sync::OnceLock<Option<FireLog>> = std::sync::OnceLock::new();
    LOG.get_or_init(|| {
        use std::io::Write;
        let path = std::env::var_os(LOG_ENV)?;
        let mut file = std::io::BufWriter::new(std::fs::File::create(&path).ok()?);
        let _ = writeln!(
            file,
            "seq,rows,cuts,copies,hits,misses,bytes_read,cut_ms,copy_ms,wait_ms,walk_ms,\
             gpu_cut_ms,gpu_tail_ms"
        );
        let _ = file.flush();
        Some(std::sync::Mutex::new((0, file)))
    })
    .as_ref()
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

const PREFETCH_K: usize = 4;

pub const PREDICTION_PREFIXES: [usize; 4] = [6, 8, 12, 16];

/// Threads the seat copies run on. llama.cpp's expert store measured 16
/// ahead of 8 on this class of SSD for whole-expert requests.
const SEAT_THREADS: usize = 16;

impl Tier {
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
        if whole && self.ring.is_some() {
            return self.ring_at(at, arena, handles, routes, rect, span);
        }
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
        if span.rows > 0 {
            let (first, mut raw) = Self::routing_bytes(arena, handles, routes, rect, span)?;
            self.dump_rows(at, &raw, rect.width as usize);
            let (base, experts) = {
                let ring = self.ring.as_ref().expect("the caller checked");
                (ring.base + half as u32 * ring.experts, ring.experts)
            };
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
