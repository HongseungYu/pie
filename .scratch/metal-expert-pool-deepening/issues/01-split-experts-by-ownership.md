# 01 Split experts.rs by ownership (move-only)

Status: needs-triage
Type: task

Move, do not change. `crates/engine-metal/src/experts.rs` becomes
`experts/mod.rs` + files cut along the ownership map in `../spec.md`:

- `experts/trace.rs`: main unchanged. `found`, `fan_out`, `weight_of`, `cuts`,
  `pass_group`, `GroupPlan`, `GroupResidency`, `Attachments`.
- `experts/source.rs`: main unchanged. `Source`, `Bytes`.
- `experts/plan.rs`: replaced. `Plan`, `BandPlan`, `RegionPlan`, `Band`,
  `Plan::{of,beside,under,...}`, and for now `Policy` + env readers +
  `parse_bytes`/`free_ram`/`gib`.
- `experts/tally.rs`: new. `CacheReport`, `FireRecord`, `FireLog`, `fire_log`,
  `log_fire`, `Prediction`.
- `experts/mod.rs` keeps `Group`, `Pool`, `Ring`, `Job`, `Tier`, `mod tests`
  (split later, issue 06).

First line of each file: a one-line comment saying which of the three
(main unchanged / replaced / new) it is.

Gate: `cargo check -p engine-metal`, clippy, `cargo test -p engine-metal --lib`.
`git diff --stat` shows only moves; `pub` surface unchanged.
