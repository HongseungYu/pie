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
| 8 | "Largest pool" is 9216 seats, not the 11204 that boots: above it the OS evicts the n-gram table's pages and every step pays ~6 ms of faults (issue 05). | An operating-system cost no fixed F can carry. |
| 9 | Zero-miss windows are 48 tokens, three reps; `steps` runs warm the whole sequence first. | What 9216 seats hold of the head; the page cache. |

## Plan

- 01 scaffolding: `stepmodel.py`, this spec, `macmon` for a direct GPU-frequency reading.
- 02 heater knobs + sweep (GPU clock).
- 03 read-path probe + knob (host warmth).
- 04 forced-miss knob.
- 05 the four configurations + mid points under the conditioned settings.
- 06 fit, validate, iterate; `RESULTS.md` and a simulator config.
