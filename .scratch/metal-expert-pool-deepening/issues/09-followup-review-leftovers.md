# 09 Follow-up: review findings outside this plan

Status: needs-triage
Type: task
Blocked by: 06

From the quality review and the deepening scan, not scheduled:

- Tally as its own module; delete the 11 Shell `expert_*`/`gathered_*`
  pass-throughs and the hand-diffed snapshot in `serve.rs` enqueue
  (candidate 4). The dsv4 integration test reads four of them.
- Heater trigger owned by the tier; remove the `OnceLock` global and the
  three `heater::pause()` calls in `device/ctx.rs` commit (candidate 6).
- Dead after the mapping removal: `Weights::windows()` (constant 0),
  `Shell::weight_windows`, `mapping::cut`/`ceiling`, `Buffer::mapped`/`window`.
- `copier`/`copiers`/`FileWriter::copy` duplicate the writer trio in
  `device/alloc.rs` / `weight_store.rs`.
- Routing-vector decode + its error string copy-pasted (collapses in 05).
- `ctx.rs` `commit` vs `commit_timed`.
- Name the seam `Tier` and `gather::Slab` share (candidate 5) if a fake
  seater is ever wanted.
