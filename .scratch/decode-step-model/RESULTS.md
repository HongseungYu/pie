# Decode-step model: results

Branch `metal-expert-cache`, 2026-09-17. Batch 1, mlx-community/Qwen3.8-Flash-Next-4bit
(`qwen38-flash-next-u4g64-kv-bf16`), M4 Pro 48 GB, 1k prompt, the teacher-forced 1024-token
sequence of `out/forced_tokens.json`. Reader: `scratch/qwen38-profile/stepmodel.py`;
pipeline: `validate.py`; runs `out/sm-*` (`out/sm_runs.log` lists every run with its
thermal state and memory before, during and after).

## The form

    step_ms = C + F + sum over layers with m_l >= 1 of ( a + b * m_l * B ),   B = 2.637 MiB

## What had to be conditioned first

| effect | symptom | cure | knob |
|---|---|---|---|
| GPU heater's own kernels beside the frames | cut frames 30.6 ms at no misses, 43.4 at 331, clock at 1578 MHz throughout (issue 02) | a narrow ALU kernel at every gap | `PIE_METAL_HEATER=on PIE_METAL_HEATER_ARM=always PIE_METAL_HEATER_KERNEL=alu` |
| host asleep between reads | a call 0.35 ms when dense, 0.83 when rare (issue 03) | one thread spinning | `PIE_METAL_CPU_HEATER=1` |
| n-gram (PLE) rows faulting from disk | 6-7 ms a step at pools >= 9216 that page the host, or on a sequence's first pass (issue 05) | pool <= 9216 seats; warm the sequence first | (measurement protocol) |

## Parameters

    step_ms = 37.69 + sum over layers with m_l >= 1 of ( 0.368 + 0.1360 * m_l * 2.637 )

| term | value | how |
|---|---|---|
| C (compute) | 33.17 ms | device time a step at zero misses: cut frames 30.11 + final frame 3.06 (`sm-c3-n40-a/b/c`) |
| F (fixed) | 4.53 ms | the rest of the zero-miss step: turnaround at the 48 cuts 3.00, seating 0.20, n-gram rows 0.04, encode 0.97, after the walk 0.31 |
| C + F | **37.69 ms** | measured, not fitted; reps 38.37 / 37.07 / 37.62, p10-p90 36.8-38.5 |
| a (a layer's read call) | **0.368 ms** | minimax over the four configurations' windows |
| b (a MiB read) | **0.1360 ms/MiB** = 0.359 ms an expert | same |

The user's earlier figures for comparison: C 41.5 (the heater-sagged device time), F 11.7, a ~0.3,
b 0.124. The cut-level curve of the read call itself, from 9568 planted and ~20k natural reads,
is `0.387 + 0.357 m` (0.135 ms/MiB): the step pays its reads at exactly their own cost, and the
minimax pick lands on that curve — there is no hidden per-miss or per-call term beyond the read.

Against the old model on the same windows: the previous parameters (C 41.5 + F 11.7 = 53.2 floor,
0.3 + 0.124 MiB) read +39% at zero misses, +16% at the min-miss pool (8192 seats), -9% at 4000
seats and +0.4% at the floor pool — right where they had been tuned, wrong everywhere else. This
one is within 2% at every point.

## The four configurations

Windows: `steps` runs, the last 256 steps of a 512- or 1024-token request after a 1024-token
warm-up; `all_in_mem` runs, the last 32 steps of the resident pass. "clean" steps are those
whose n-gram rows were cached (`ple_ms` < 0.5) and whose reads cost less than 1.5x the drive's
own curve (the drive stalls for some seconds every ten minutes or so on this box: `sm-mid-s4000-c`
lost 23 of 256 steps to one such stall, and two earlier runs were spoiled outright and rerun —
issue 07). The error is over the clean steps; the last column over every step.

**Model** (minimax over ['c1', 'c2', 'c3', 'mid']): step = 37.69 + sum over layers with m >= 1 of (0.368 + 0.1360 * m * 2.637)   [b = 0.359 ms a miss]

| config | tag | seats | steps (clean) | misses/step | L/step | device | copy | ple | measured | predicted | error | error, all steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c3 | sm-c3-n40-a | 8192 | 32 (32) | 0.0 | 0.00 | 33.28 | 0.00 | 0.03 | 38.37 | 37.69 | -1.75% | -1.75% |
| c3 | sm-c3-n40-b | 8192 | 32 (32) | 0.0 | 0.00 | 32.84 | 0.00 | 0.01 | 37.07 | 37.69 | +1.68% | +1.68% |
| c3 | sm-c3-n40-c | 8192 | 32 (22) | 0.0 | 0.00 | 33.49 | 0.00 | 2.41 | 37.62 | 37.69 | +0.19% | -5.82% |
| c1 | sm-c1-s1024-a | 1024 | 256 (256) | 331.4 | 47.68 | 34.58 | 135.16 | 0.01 | 175.15 | 174.09 | -0.60% | -0.60% |
| c1 | sm-c1-s1024-b | 1024 | 256 (256) | 331.4 | 47.68 | 34.60 | 133.84 | 0.01 | 173.85 | 174.09 | +0.14% | +0.14% |
| c1 | sm-c1-s512-a | 512 | 256 (256) | 383.4 | 47.96 | 34.67 | 149.50 | 0.01 | 189.65 | 192.85 | +1.69% | +1.69% |
| c2 | sm-c2-s8192-a | 8192 | 256 (256) | 52.8 | 26.57 | 35.01 | 28.13 | 0.02 | 67.57 | 66.42 | -1.70% | -1.70% |
| c2 | sm-c2-s8192-b | 8192 | 256 (256) | 52.8 | 26.57 | 33.97 | 28.30 | 0.01 | 66.68 | 66.42 | -0.38% | -0.38% |
| mid | sm-mid-s1536 | 1536 | 256 (256) | 291.8 | 47.06 | 34.49 | 119.17 | 0.01 | 159.03 | 159.66 | +0.39% | +0.39% |
| mid | sm-mid-s3072-c | 3072 | 256 (256) | 196.6 | 44.57 | 34.28 | 84.70 | 0.01 | 124.07 | 124.60 | +0.43% | +0.43% |
| mid | sm-mid-s4000-c | 4000 | 256 (233) | 166.9 | 43.00 | 34.27 | 74.38 | 0.01 | 113.65 | 113.37 | -0.25% | -14.17% |
| mid | sm-mid-s6144 | 6144 | 256 (256) | 116.4 | 38.95 | 34.06 | 54.60 | 0.01 | 93.48 | 93.78 | +0.32% | +0.32% |

**Worst error over the four configurations: 1.75%** (the zero-miss floor's own rep spread and
the 512-seat extreme); everything else within 1%.

| configuration | runs | what it holds |
|---|---|---|
| 1 max misses | 1024 seats (the floor with the ring) x2, 512 seats with the ring off | 331 / 383 misses a step in 48 layers, hit rate 0.31 / 0.20, steps 175 / 190 ms: -0.6, +0.1, +1.7% |
| 2 min misses | 8192 seats x2 (the largest pool that keeps the n-gram rows cached, issue 05) | 53 misses a step in 27 layers, hit rate 0.89, 67 ms: -1.7, -0.4% |
| 3 zero misses | 8192 seats, prime + resident, 40 tokens x3 | 0 misses, 37.1-38.4 ms: +-1.75% |
| 4 misses in chosen layers | 15 planted-miss runs (below) | the call term is separable and additive (device time unchanged within 1 ms); the grid's reads are cheaper than a real miss's |
| mid points | 1536 / 3072 / 4000 / 6144 seats | +0.4, +0.4, -0.3, +0.3% |

A fifth regime, kept apart: the 128-token `all_in_mem` window (`sm-c3-n128-c`, 71 misses a step
in 33 layers) reads -8.3% under this model. Its reads cost 0.60 ms a miss where every
`steps`-mode run at the same misses a call pays 0.53: they run within seconds of a 55 GiB prefill
burst (both 128-token runs show it, 0.593 and 0.605), and the drive has not recovered. A model of
the engine cannot carry that; a simulator that fires a long prefill and decodes at once should
know the drive is slower for a while afterwards.

### The planted grid under the final model

| config | tag | seats | steps (clean) | misses/step | L/step | device | copy | ple | measured | predicted | error | error, all steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c4 | sm-c4-0-23_1 | 8192 | 32 (30) | 24.0 | 24.00 | 33.26 | 17.11 | 0.21 | 54.75 | 55.13 | +0.70% | +0.75% |
| c4 | sm-c4-all_1-b | 8192 | 32 (32) | 48.0 | 48.00 | 33.60 | 33.48 | 0.07 | 71.43 | 72.57 | +1.60% | +1.60% |
| c4 | sm-c4-all_1 | 8192 | 32 (29) | 48.0 | 48.00 | 33.64 | 34.72 | 0.22 | 72.54 | 72.57 | +0.04% | -0.02% |
| c4 | sm-c4-all_5 | 8192 | 32 (32) | 240.0 | 48.00 | 34.20 | 105.39 | 0.02 | 145.00 | 141.43 | -2.46% | -2.46% |
| c4 | sm-c4-every_12_1 | 8192 | 32 (32) | 4.0 | 4.00 | 33.10 | 2.94 | 0.08 | 40.88 | 40.60 | -0.69% | -0.69% |
| c4 | sm-c4-every_2_10 | 8192 | 32 (1) | 240.0 | 24.00 | 34.24 | 91.94 | 6.68 | 131.34 | 132.60 | +0.95% | -4.31% |
| c4 | sm-c4-every_2_5 | 8192 | 32 (32) | 120.0 | 24.00 | 34.17 | 55.20 | 0.10 | 94.44 | 89.56 | -5.17% | -5.17% |
| c4 | sm-c4-every_48_1 | 8192 | 32 (31) | 1.0 | 1.00 | 32.85 | 0.74 | 0.08 | 37.91 | 38.42 | +1.34% | +1.33% |
| c4 | sm-c4-every_4_1 | 8192 | 32 (32) | 12.0 | 12.00 | 32.92 | 8.57 | 0.08 | 45.49 | 46.41 | +2.04% | +2.04% |
| c4 | sm-c4-every_4_10-b | 8192 | 32 (24) | 120.0 | 12.00 | 34.08 | 46.63 | 1.74 | 85.69 | 85.15 | -0.63% | -2.73% |
| c4 | sm-c4-every_4_10 | 8192 | 32 (26) | 120.0 | 12.00 | 34.30 | 46.24 | 0.89 | 85.68 | 85.15 | -0.62% | -2.23% |
| c4 | sm-c4-every_4_2 | 8192 | 32 (32) | 24.0 | 12.00 | 33.80 | 12.98 | 0.08 | 51.41 | 50.72 | -1.35% | -1.35% |
| c4 | sm-c4-every_4_4 | 8192 | 32 (31) | 48.0 | 12.00 | 33.97 | 23.06 | 0.08 | 61.85 | 59.32 | -4.08% | -4.65% |
| c4 | sm-c4-every_4_5 | 8192 | 32 (32) | 60.0 | 12.00 | 33.99 | 27.89 | 0.02 | 66.66 | 63.63 | -4.54% | -4.54% |
| c4 | sm-c4-every_8_8 | 8192 | 32 (32) | 48.0 | 6.00 | 33.74 | 19.98 | 0.02 | 58.33 | 57.12 | -2.09% | -2.09% |

Over-predicted by up to 5%, as expected from issue 07: a planted read fetches an expert the prime
pass read seconds before, and the drive's cache serves it below a cold miss's price.

## C by layer type and module

`sm-kern-s8192`, `--diag kernel-profile=2` (every kernel in its own command buffer), the
last 32 decode fires, classified by entrypoint and shape (`kernelsplit.py`), scaled to the
frames' device time at zero misses:

```
sm-kern-s8192: 32 decode fires, kernels sum 36.74 ms a fire; frames 33.17 -> compute_scale 0.9029

| layer type | count | attention | hyper_connection | ffn | per layer (scaled) |
|---|---|---|---|---|---|
| linear_attention | 36 | 0.281 | 0.126 | 0.198 | 0.605 |
| full_attention | 12 | 0.413 | 0.126 | 0.198 | 0.737 |

output (readout) 2.553 ms, embed/PLE kernels 0.006 ms, unclassified 0.000 ms
sum check 33.17 ms against the frames' 33.17
```

(Norms and the low-rank pair are counted as hyper-connection; the router, the shared
expert and the experts as ffn; the GDN scan/conv/gates and the paged SDPA as attention.)

## The planted-miss grid: the form's structure

Fifteen `all_in_mem` runs at 8192 seats, primed, `PIE_EXPERT_CACHE_FORCE_MISS=<layers>:<k>`,
last 32 steps of the resident pass. The cut-level fit over their 9568 reads:

```
## Read call, cut level: 9568 cuts with a miss over 15 planted-miss runs

    copy_ms = 0.387 + 0.357 * m     (r2 0.843)   ->  a = 0.387 ms a call, b = 0.1352 ms a MiB
      m= 1:  0.711 measured (median 0.695) vs  0.744 modelled, 4384 cuts
      m= 2:  1.081 measured (median 1.070) vs  1.101 modelled, 384 cuts
      m= 4:  1.952 measured (median 1.761) vs  1.814 modelled, 384 cuts
      m= 5:  2.244 measured (median 2.099) vs  2.170 modelled, 2688 cuts
      m= 8:  3.330 measured (median 3.282) vs  3.240 modelled, 192 cuts
      m=10:  3.876 measured (median 3.798) vs  3.953 modelled, 1536 cuts
    fitting on planted: 15 runs, 480 steps
```

Equal misses, different call counts (48 misses: `all:1` 72.0 / 71.4 ms, `every:4:4` 62.2,
`every:8:8` 58.3; 240 misses: `all:5` 145.0, `every:2:10` 131-139): the call term is real and
separable, and a planted layer costs the step exactly its own read (device time unchanged
within 1 ms). The grid's coefficients are optimistic, though: a planted read fetches an
expert the prime pass read seconds before, and the drive's own cache serves it (a 1-miss
call 0.71 ms against 0.80 for a real miss, whose median is the same 0.71 but whose mean
carries cold reads). So a and b are set on the natural-miss configurations below, and the
grid is the structural check.

## How to reproduce

    HEAT="PIE_METAL_HEATER=on PIE_METAL_HEATER_ARM=always PIE_METAL_HEATER_KERNEL=alu PIE_METAL_HEATER_ALU_ITERS=8192"
    SEATS_MAX=9216 WIN=48 ./sm_sweep3.sh c3 c4 c1 c2 mid     # ~1 h
    .venv/bin/python validate.py --win 40 --last 256
