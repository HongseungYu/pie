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

2026-09-17, harness gate (clean mode, 4000 seats, heater on, 1k prompt,
1024 teacher-forced tokens): every forced-token assertion passed.

Timing, corrected 2026-09-17 (see issue 13). What this run actually shows:
`T(1024)` 132.42 s against issue 04's 133.35 s and issue 02's 133.30 s, so
no measurable change end to end; decode hit rate 67.8%, the same as both;
and 942.6 GiB read from disk against 931.2 GiB at issue 02, because the
borrowed ring finds fewer of a layer's experts already seated to copy from
(8928 copied against 12000). The boot line reads `4000 seats ... 10.30 GiB
... 1024 of which a prefill fire borrows`: the same knob costs 2.64 GiB
less than when the ring was reserved beside the pool, which is the change's
real win.

The first report of this run claimed the last-512 figure fell from 109 to
80 ms because the returned ring seats were free hits for the decode behind
them. That was wrong twice over: the hit rate did not move, and a later run
of functionally identical code (the cleanup gate) read 122.24 ms for the
same figure. See issue 13.

2026-09-17, the measurement that answers for this change. Three reps of the
same build at each of two knobs: 4000 seats (10.30 GiB) gives `T(1024)` of
148.0 / 132.4 / 132.6 s at a 67.8% hit rate; 5024 seats (12.94 GiB, the
memory the old reserved-ring build took at knob 4000) gives 122.2 / 122.4 /
128.0 s at 72.6%. Against the baseline's 140.5 s and 66% at that same
memory, borrowing the ring is worth about 13% end to end, because the 1024
seats it used to reserve are seats a decode can now hit. Table in issue 13.
