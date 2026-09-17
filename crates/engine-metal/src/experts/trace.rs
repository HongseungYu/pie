// Inherited from main unchanged: reading the model trace for routed groups.

use std::collections::{BTreeMap, BTreeSet};

use model_compiler::CompiledModel;
use model_ir::{Def, Linear, Operands, Operation, Trace, ValueId};

use crate::error::{Fault, Result};

use super::*;

pub type Attachments = BTreeMap<usize, Vec<usize>>;

#[derive(Debug, Clone)]
pub struct GroupPlan {
    pub routes: ValueId,
    pub experts: u32,
    pub slots: u32,
    pub bands: Vec<usize>,
    pub hint: Option<ValueId>,
}

pub(super) fn found(
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

pub(super) fn weight_of(trace: &Trace, id: ValueId) -> Result<usize> {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupResidency {
    pub name: String,
    pub experts: u32,
    pub slots: u32,
    pub in_seat: Vec<Option<u32>>,
}
