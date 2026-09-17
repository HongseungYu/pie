# 04 Planting misses in chosen layers

Status: resolved
Type: task

Configuration 4 asks for steps whose misses sit in known layers at known counts, so the
cost of a read call is measured on its own: the same m spread over 48 calls or over 6, and
L = 1, 4, 12, 24, 48 at one miss each. The pool's LRU cannot be told to do that; a knob can.

`PIE_EXPERT_CACHE_FORCE_MISS=<layers>:<k>` — `<layers>` is `all`, `every:<n>` (layers
0, n, 2n, ...) or a comma list with `lo-hi` ranges; `<k>` is how many of a decode segment's
distinct routed experts are read again whatever the pool holds. Parsed once in `Knobs::of`
(`experts/plan.rs`), resolved to a per-group count when the tier opens, applied in
`Tier::decide`: the first `k` distinct experts skip the hit branch, take a seat and go on
the read list like any miss; `assign` then strips the seat the expert held, so the pool's
occupancy does not change and the same seat serves the next plant. Prefill segments run on
the ring and are untouched. The plants are counted as misses, so the cut log prices them and
`stepmodel.py` reads L and m_l off it as for any run.

Gate: at the large pool, primed by `all_in_mem`, `all:1` reads as 48 misses in 48 layers a
step, `every:4:10` as 120 in 12.

## Gate (2026-09-17 17:50, `sm-g1-all1`, `sm-g2-every4-10`; 11204 seats, primed, 64 tokens)

    all:1        48.0 misses in 48.00 layers a step, hit rate 0.900, forced tokens verified
    every:4:10  120.0 misses in 12.00 layers a step, hit rate 0.750, forced tokens verified

The step adds up: 91.3 ms measured at `all:1` against the 43.2 ms floor + 48 x 0.99 ms
calls = 90.9; the device time is the floor's (33.3 against 32.8). A one-miss call reads
0.99 ms here and a ten-miss call 4.40, so at this pool a = 0.61 and b = 0.38 ms a miss —
`a` well above the 0.35 of the dense-read fit at 4000 seats, which issue 05 traces to the
host's memory pressure at 11204 seats rather than to the read itself.
