# 02 Floor 2E and delete Metal's pass feeding

Status: claimed
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

## Comments

2026-09-17: code landed. The pool's floor is now one layer's experts (E,
the max over groups) in every policy arm, replacing the pass-derived
`pass_group(n) >= fan` search; the ring is still 2E seats beside the pool
until issue 05 folds it in, so the total floor with prefill enabled is
E + 2E today and becomes 2E at 05. Deleted: `Tier::pass_at`, `Passing`,
`Tier.passing`, `prefetch_group` (only pass_at called it), `pass_group`,
the `pass` tuple through `Cuts`/`Sink::across`/`Tier::segment`,
`Cuts.groups`, the `Sink::fire` launch skip, `At.tail` and Metal's
`Encode::tail` override, `Window.pass/passes`, `Windows::of`'s
`run_passes`, serve.rs `run_passes` and the `slots / top_k` cap
derivation (only `stream_rows_per_cut` caps now), diag `pass-half` and
`expert-passes`. `Tier::segment` returns `()` (the pass count had no
reader left). 48 insertions, 299 deletions. Gate: clippy --all-targets
clean, lib tests 20 passed; harness decode pending the machine being
free.
