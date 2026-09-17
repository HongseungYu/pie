# 03 Resolve the branch's knobs once

Status: needs-triage
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
