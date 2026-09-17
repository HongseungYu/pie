# ADR-0002: The prefill ring is borrowed from the pool; the pool floor is 2E

Date: 2026-09-17
Status: accepted

## Context

A prefill fire names nearly every expert of every layer, so the tier serves
it from two whole-layer buffers filled alternately (one half computes while
the next layer lands in the other). Those halves were 2E seats reserved
past the pool's own, sized into the plan beside `slots`. The pool itself was
allowed to be smaller than one layer's experts, which is why a segment could
run in several expert-major passes (`pass_at`, `run_passes`, `Sink::fire`
launch skipping) and why row caps were derived from `slots / top_k`.

## Decision

The ring is not reserved. At the start of a whole-mode fire the tier borrows
2E seats from the pool (never-used seats first, then the LRU-coldest), holds
them for the fire, and at the end returns them as valid `(layer, expert)`
residents at the cold end of the LRU. The pool's floor is 2E seats when
prefill is enabled and E when it is disabled (`PIE_EXPERT_CACHE_PREFILL=0`);
plans below the floor are refused.

With the floor at or above E, one segment can never route to more distinct
experts than fit, so multi-pass segments are impossible: Metal's pass
mechanism and the `slots / top_k` row cap are deleted. The explicit
`stream_rows_per_cut` tuning remains.

## Consequences

- Working-set damage from a prefill is fixed at 2E seats regardless of the
  layer count; between prefills decode has those 2E seats too.
- Steady-state cost equals the reserved design; the last two layers' full
  expert sets are free hits for the next decode.
- Holds need three lifetimes (segment, fire, inflight); a per-seat hold kind
  replaces the boolean pin.
- The shared pass plumbing in model-exec and the vulkan/wgpu window is fed
  by no engine after this and is removed in a follow-up.
- Configurations with fewer than 2E seats are no longer supported with
  prefill enabled; a review should not reintroduce passes to support them.
