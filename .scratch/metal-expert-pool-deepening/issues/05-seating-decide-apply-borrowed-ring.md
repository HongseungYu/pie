# 05 One seating path: decide / apply, borrowed ring, holds, fire hooks

Status: claimed
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
