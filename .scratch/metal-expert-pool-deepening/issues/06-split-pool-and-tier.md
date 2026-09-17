# 06 Split pool and tier

Status: needs-triage
Type: task
Blocked by: 05

Move-only after 05: `experts/pool.rs` (Pool, Group, holds, LRU tests) and
`experts/tier.rs` (Tier, seating decide/apply, fire hooks). `mod.rs`
re-exports.

Gate: check/clippy/lib tests; diff is moves only.
