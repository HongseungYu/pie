# 03 Resolve the branch's knobs once

Status: resolved
Type: task
Blocked by: 01

Scope: the env vars this branch added: `PIE_EXPERT_CACHE`, `_HEADROOM`,
`_NOCACHE`, `_PREFILL`, `_REPORT`, `_LOG`, and the heater's two. Main-era
diag fields (`pass_half`, `route_prefetch`, `route_dump`, `prefetch_k`,
`seat_threads`) stay where they are.

- One value (name and shape open, see issue 08) built in `api.rs` where
  `Plan::beside` is called; passed to `Plan::beside`/`under` and
  `Tier::open`; the heater receives its two values from it.
- `Plan::under` and `Tier::open` read no env and no diag for these knobs.
  `Plan::of`'s "environment-free" comment becomes true.
- `Plan.uncached()` replaces `experts::nocache()` at `serve.rs:543`,
  `weights.rs:858`, `experts.rs` (Tier::open). `report_every` is read once,
  not per fire.

Gate: check/clippy/lib tests.

## Comments

2026-09-17: done. `experts::Knobs` holds the eight knobs this branch added
(policy, prefill override, nocache, report_every, log path, heater) and is
built once by `Knobs::of(budget)`, which `Plan::beside` calls; every
`std::env::var` for them now sits in that one function (heater's two inside
`heater::wanted()`, which it calls). `Plan::under` takes the value, so
`Plan::of`'s "environment-free" comment is finally true: it passes
`Knobs::under(policy)`, every other knob at its default. The plan carries
the value and answers `uncached()`, `report_every()`, `heater()`; those
replaced `experts::nocache()` in serve.rs and weights.rs, the per-fire
`report_every()` read in enqueue (now a Shell field off the plan), and
`heater::wanted()` at the heater's start. The fire CSV's `OnceLock` is
opened by `Tier::open` from the plan instead of reading the env lazily.
Gate: clippy --all-targets clean, lib tests 20 passed.
