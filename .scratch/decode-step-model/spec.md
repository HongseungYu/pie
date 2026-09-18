# Decode-step latency model: batch 1, Qwen3.8-Flash-Next, expert LRU pool on Metal

Branch: `metal-expert-cache`. Opened 2026-09-17. Machine: M4 Pro 48 GB. Model: 48 layers
(36 GDN + 12 full attention), 512 experts, top-k 10, one expert B = 2.637 MiB. Tools live in
`~/hongseung/scratch/qwen38-profile/` (outside git; `stepmodel.py` is this effort's reader),
records here, engine knobs on this branch.

Follows `metal-expert-pool-deepening` (issues 13-16 measured what a step costs and found the
two effects below).

## The form to hold

    step_ms = C + F + sum over layers with m_l >= 1 of ( a + b * m_l * B )

C compute (device time a step), F host-side fixed cost, a one layer's read call, b ms a MiB.
Target: +-3% of the measured mean over 128-256 steady-state steps, in four configurations:

1. **max misses** - the pool at its floor (1024 seats with the ring; 512 with `PREFILL=0`)
2. **min misses** - the largest pool that boots (about 11204-12000 seats)
3. **zero misses** - `--mode all_in_mem`: the same forced request twice, the second finds its
   experts seated (the oracle: teacher forcing makes the future known, the prime pass preloads)
4. **misses concentrated in chosen layers** - a forced-miss knob, so the call cost `a` is
   measured at known L and m, and "same m, different L" is a direct experiment

The parameters are tuned; the form is kept unless a term is provably missing.

## What has to be conditioned first (the user's rule: compute must not depend on misses)

- **GPU clock.** Device time a step is 30.6 ms at no misses and 43 ms at 264, heater on
  (issue 15 of the previous effort). The heater arms only after a segment that read from disk,
  never over the ~9 ms inter-fire gap, and its two in-flight 16 MiB kernels overlap the real
  frame at `pause()` (about +0.23 ms a missing layer). Fix: arm policy + kernel shape knobs,
  swept for the setting where device time is the same at 0 and max misses.
- **Host read path.** `copy_ms = 0.35 + 0.36 m` when reads are dense (4000 seats), `0.83 +
  0.41 m` when rare (11204). Sixteen threads are spawned per call (`device/alloc.rs`), and an
  idle host/IO path pays to wake. Fix: probe which of thread creation / NVMe idle / CPU DVFS
  it is, then the smallest knob that keeps it warm. Decided with the user: fix it in the
  engine, do not model `a` as a function of read density.

## Decisions

| # | Decision | Why |
|---|---|---|
| 1 | C + F is measured (the zero-miss floor), not fitted. | Fits extrapolate from runs whose device time was already inflated (issue 16). |
| 2 | The heater's winner becomes what `PIE_METAL_HEATER=on` means; the sweep knobs stay. | One env word for the conditioned system. |
| 3 | Read-path warmth is an engine knob, default off, on for every measured run. | User's call, 2026-09-17. |
| 4 | Forced misses are counted as misses and re-read into a fresh seat; occupancy unchanged. | The cut log then prices them like any miss. |
| 5 | Windows: mean of the last 256 steps (median, p10, p90 beside it), two fresh-server runs. | Issue 13: the harness's differenced figures are not measurements of a step. |
| 6 | Records: `spec.md`, `issues/NN-*.md` one per experiment, `RESULTS.md` the final table; run tags `sm-*` in `out/`. | Traceable later. |
| 7 | The heater is `alu` 1024 x 8192, two in flight, armed at every gap (issue 02). | Cut frames 29.7 at no misses, 31.5 at 331; every other shape loses on one end. |
| 8 | "Largest pool" is 8192 seats, not the 11204 that boots: from 9216 up the OS evicts the n-gram table's pages between requests and a step pays ~6 ms of faults (issue 05). | An operating-system cost no fixed F can carry. |
| 9 | Zero-miss windows are 40 tokens, three reps; `steps` runs warm the whole sequence first. | What 8192 seats hold of the head beside two ring borrows; the page cache. |
| 10 | a, b are the minimax pick over the four configurations' windows; the planted grid checks the form's structure but not its coefficients. | The drive's cache serves a planted re-read below a cold miss's price (issue 07). |
| 11 | A step whose reads cost > 1.5x the drive's curve is a drive stall and set aside, counted and shown; runs spoiled whole are rerun and the originals kept. | The drive pauses for seconds every ten minutes or so; no fixed a, b can carry it (issue 07). |
| 12 | Reads within seconds of a 55 GiB prefill burst (the 128-token `all_in_mem` window) are a fifth regime, reported apart. | 0.60 ms a miss against 0.53; the drive's, not the engine's. |
| 13 | The n-gram rows are read by uncached pread at offsets known from the load, prefetched as the fire opens from host-computed ids; no separate file (issue 08). | Their cost was the page cache's; now 0.02 ms a step, constant. The floor is 37.05. |
| 14 | Above 8192 seats the read call's a, b ramped up between 34.2 and 36.9 GB pinned (a x 1.64, b x 1.04 at the top) — the engine serialising an expert's preads across store chunks (issue 09); fixed in issue 10, after which a, b are flat to 13200 seats. | The engine issues one threaded read pass per store chunk; past that size the pool's band regions stop sharing a Metal chunk and an expert's two preads serialise (issue 09). Not a host effect: measured flat against alignment, pressure, destination size and virgin pages. |

## How to repeat this

`METHOD.md` is the manual: pre-flight, the four conditionings with their gates, the measurement
protocol, the five configurations, the fit, and a porting section for another machine, model or
batch size. The issues below are its evidence, cited from it as "(why: issue NN)" — read one only
when the manual sends you.

## Plan

- 01 scaffolding: `stepmodel.py`, this spec, `macmon` for a direct GPU-frequency reading.
- 02 heater knobs + sweep (GPU clock).
- 03 read-path probe + knob (host warmth).
- 04 forced-miss knob.
- 05 the largest pool the host can spare (memory pressure, the n-gram page cache).
- 06 a pool at the floor refuses a second prefill (fixed).
- 07 the fit and the four configurations; `RESULTS.md` and `out/sim/configs/mac_m4pro_qwen38flash_np1_v2.json`.
- 08 the n-gram rows off the SSD every time, prefetched; the floor re-measured (`..._v3.json`).
- 09 the read call above 8192 seats: a ramp, traced to the engine serialising reads per store chunk (`..._v4.json`, before the fix).
- 10 one threaded pass over the chunks: the ramp removed; the flat model holds to 13200 seats (`..._v5.json`).
