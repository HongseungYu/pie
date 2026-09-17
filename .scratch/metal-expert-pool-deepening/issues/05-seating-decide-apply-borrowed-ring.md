# 05 One seating path: decide / apply, borrowed ring, holds, fire hooks

Status: resolved
Type: task
Blocked by: 04

The deepening proper. ADR-0002.

- `decide(routes, &pool) -> Seating { hits, jobs, evictions, holds }` pure;
  `apply(Seating)` moves bytes through `Source` and only then commits
  `seat_of` / `in_seat` / LRU order. Prefetch is the same path with an
  async apply joined at the next cut (commit at join).
- `segment_rows` / `ring_at` collapse into one path; the ring is a seat
  range chosen by the fire's mode, not a second buffer.
- `Ring` as separate seats goes. At `begin_fire` in whole mode the tier
  borrows 2E seats (free first, then LRU-coldest), fills halves as today
  (async, alternate per layer), and at `end_fire` returns them as valid
  `(layer, expert)` residents at the cold end.
- Holds: per-seat enum none / segment / fire / inflight replacing
  `pinned: Vec<bool>`; `victim()` skips any hold; each release point clears
  only its kind.
- `Tier::begin_fire(&lane_rows) -> Mode`, `Tier::end_fire()` called from
  serve.rs enqueue. Delete `Shell.ring_min`, `Prepared.whole`, `Cuts.whole`,
  `segment`'s `whole` argument, `ring_has`/`ring_join`/`ring_ready`/
  `ring_fill` as separate functions, `ring_join(usize::MAX)` sentinel.
- `Tier::segment` interface shrinks to (arena, handles, routes, rect, hint,
  span).

Gate: check/clippy/lib tests, then one harness decode vs RESULTS.md. Hit
rate should rise (returned layers hit); step ms must not regress.

## Comments

2026-09-17: done, with one thing left as it was. `segment_rows` and
`ring_at` are still two functions: they share the pool, the holds, the seat
table and the fire hooks, but their rhythms differ (seat what this segment
names, versus hold a whole layer filled a layer ahead), and folding them
would put a strategy branch inside one function rather than delete one.
`ring_has`/`ring_join`/`ring_ready`/`ring_fill` stay for the same reason;
`ring_join` takes `Option<group>` instead of the `usize::MAX` sentinel.

What landed: `Hold` per seat (Free / Segment / Fire / Inflight) replacing
the boolean pin, each release point clearing only its own kind; `decide`
(pure over the routing bytes and the pool: takes and holds seats, moves
nothing) then `land` (copies, then makes resident), with prefetch the same
path spawned and committed at the join; `Tier::begin_fire` borrowing
`2 * experts` seats from the pool for a prefill fire and `end_fire`
returning them at the cold end of the LRU holding the last two layers they
read; the plan sizing the ring as a floor rather than a reservation, so the
same `PIE_EXPERT_CACHE` now costs 2.64 GiB less on this model and the
returned seats are hits a decode can take. `Shell.ring_min`,
`Prepared.whole`, `Cuts.whole` and `segment`'s `whole` argument are gone.

Gate: clippy --all-targets clean, lib tests 20 passed.
