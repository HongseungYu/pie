// Replaced main's: sizing the shared pool (main sized a fixed share per bank).

use std::collections::{BTreeMap, BTreeSet};

use model_ir::Trace;

use crate::error::{Fault, Result};

use super::*;

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

pub(super) fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

#[derive(Debug, Clone, Default)]
pub struct Plan {
    pub(super) bands: Vec<BandPlan>,
    pub(super) groups: Vec<GroupPlan>,
    pub(super) regions: Vec<RegionPlan>,
    pub(super) alias_of: BTreeMap<usize, usize>,
    pub(super) resident_of: BTreeMap<usize, u32>,
    pub(super) host_of: BTreeMap<usize, u64>,
    pub(super) slots: u32,
    pub(super) ring: u32,
    pub(super) prefill_min: u32,
    pub(super) per_slot: u64,
    pub(super) pairs: u64,
    pub(super) policy: String,
    pub(super) device_bytes: u64,
    pub(super) host_bytes: u64,
    pub(super) gathered: crate::gather::Plan,
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
