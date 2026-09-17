# 06 Split pool and tier

Status: resolved
Type: task
Blocked by: 05

Move-only after 05: `experts/pool.rs` (Pool, Group, holds, LRU tests) and
`experts/tier.rs` (Tier, seating decide/apply, fire hooks). `mod.rs`
re-exports.

Gate: check/clippy/lib tests; diff is moves only.

## Comments

2026-09-17: done. `experts/mod.rs` is 17 lines of module declarations and
re-exports; `pool.rs` holds the seats, their holds, the LRU and its test,
`tier.rs` the ring, the seating and the tier. The knob tests moved to
`plan.rs`, where the knobs are. Every child names what it imports instead
of leaning on `use super::*`, so what a file depends on is visible at its
head. Gate: clippy --all-targets clean, lib tests 20 passed.

| file | lines | what it owns |
|---|---|---|
| `mod.rs` | 17 | the module's face |
| `trace.rs` | 279 | main's: reading the trace for routed groups |
| `source.rs` | 96 | main's: where an expert's bytes come from |
| `plan.rs` | 734 | replaced: the knobs and the pool's size |
| `pool.rs` | 200 | new: seats, holds, the LRU |
| `tally.rs` | 153 | new: what the pool reports |
| `tier.rs` | 1041 | replaced: the ring, seating, the fire |
