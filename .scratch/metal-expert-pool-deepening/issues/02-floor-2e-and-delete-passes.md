# 02 Floor 2E and delete Metal's pass feeding

Status: needs-triage
Type: task
Blocked by: 01

One commit, because deleting passes without the floor lets a small pool hit
`evict()`'s "every seat pinned" fault.

- `Plan::under`: `need` = 2E when prefill is enabled, E when
  `PIE_EXPERT_CACHE_PREFILL=0`. `Policy::Slots(n)` below the floor and
  `Policy::Budget` below `dense + seats(floor)` are refused. The ring is no
  longer extra seats past the pool (`slots + ring` -> `slots`); see ADR-0002.
  Until issue 05 lands the ring still lives at `pool.slots..` as today, so
  keep `ring` accounted inside `slots`, not beside it.
- Delete: `Tier::pass_at`, `Passing`, `Tier.passing`, `pass_group`,
  `Tier::segment`'s `pass` argument, serve.rs `run_passes`, `Cuts.groups`
  Cell and the `Sink::fire` launch skip, diag `pass_half` and
  `expert_passes` (+ their tests), the `(u32, u32)` pass tuple in encode.rs.
- Delete the `slots / top_k` derivation of `run_caps`; keep the explicit
  `stream_rows_per_cut` tuning (0 = uncapped).
- Leave model-exec `Descriptor.run_passes`/`pass_spans`/`tail` and
  vulkan/wgpu `Window.pass/passes` alone (issue 07).

Gate: check/clippy/lib tests, then one harness decode vs RESULTS.md.
