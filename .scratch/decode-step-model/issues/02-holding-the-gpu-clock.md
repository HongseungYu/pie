# 02 Holding the GPU clock so compute does not depend on the miss count

Status: claimed
Type: research

The rule the user set: before any fitting, the device time a step must be the same whatever
the pool misses. Issue 15 of the previous effort measured 30.6 ms at no misses and 43 ms at
264, heater on (`PIE_METAL_HEATER=on`: a 16 MiB scale kernel, two in flight, armed only after
a segment that read from disk). Two gaps in that rule: the ~9 ms between one fire's last
frame and the next fire's first is covered only if layer 47 read, and the two kernels in
flight at `pause()` run beside the real frame (RESULTS.md's sweep found +0.23 ms a missing
layer that no size or depth removed).

Knobs added (commit 75452854): `PIE_METAL_HEATER_ARM=always`, `PIE_METAL_HEATER_KERNEL=alu`
(a narrow chain of FMAs: `_ALU_THREADS` 1024, `_ALU_ITERS`), `PIE_METAL_HEATER_LOG`. Every
run below also carries `PIE_METAL_CPU_HEATER=1` (issue 03) and `macmon` sampling.

## The measurement

D0: `all_in_mem` at 11204 seats, 64 tokens, the resident pass (0 misses), last 48 steps.
Dmax: `steps` at 1024 seats (the floor with the ring), 512 tokens, last 256 steps.
"device" is the cut frames' own device time a step (`gpu_cut_ms`; the final frame is not
captured on this branch, see issue 01). Tags `sm-<id>-d0` / `sm-<id>-dmax` in `out/`.

| id | env | D0 device | D0 step | D0 rest | Dmax device | Dmax step | Dmax misses/L | Dmax copy | Dmax rest | MHz D0/Dmax |
|---|---|---|---|---|---|---|---|---|---|---|
| h0 | H=on | 30.10 | 44.49 | 10.46 | 43.35 | 189.72 | 331 / 47.7 | 133.41 | 6.30 | 1566 / 1578 |
| h1 | H=on H_ARM=always | 45.74 | 65.64 | 11.65 | 43.48 | 189.81 | 331 / 47.7 | 133.32 | 6.44 | 1578 / 1578 |
| h3a | H=on H_ALU_ITERS=2048 H_ARM=always H_KERNEL=alu | 29.41 | 42.93 | 10.79 | 31.58 | 200.73 | 331 / 47.7 | 159.18 | 5.76 | 1578 / 1578 |
| h3b | H=on H_ALU_ITERS=8192 H_ARM=always H_KERNEL=alu | 29.70 | 43.10 | 10.69 | 31.52 | 173.05 | 331 / 47.7 | 132.71 | 5.47 | 1578 / 1578 |
| h3c |  | 32.63 | 128.40 | 11.66 | - | - | - | - | - | 1578 / nan |
| hoff | H=off | 30.06 | 44.36 | 10.35 | - | - | - | - | - | 1523 / nan |

## What macmon says about H0 (2026-09-17 16:45)

At the floor the GPU reads **1578 MHz in every sample of the decode, min = max, active
1.00**: with 331 misses a step the heater is armed at every cut and the clock never falls.
Yet the cut frames take 43.35 ms against 30.10 at no misses (where the clock reads 1566
median, the heater never armed). So the 13 ms is not the clock. It is the heater's own
kernels: two 16 MiB scales in flight when a frame commits, 48 times a step, about 0.27 ms
each — the "+0.23 ms a missing layer" of RESULTS.md's sweep, now explained. A kernel that
holds the device active while taking a sliver of it is what the `alu` variants test.

Also seen: the cold prime pass costs 2.11 ms a call with or without the CPU heater
(`cold-s11204` steps 1-63 against `sm-h0-d0`'s prime): per miss 0.47 ms against 0.34-0.36
in steady state. The first write into a never-used seat faults its pages in; the pool must
be fully touched before a window is measured (warm-up of 512+ steps at large pools).

## The ALU kernel (2026-09-17 17:05)

`alu`, 1024 threads through 8192 (h3b) or 2048 (h3a) dependent FMAs, two in flight, armed at
every gap: the cut frames read 29.7 ms at no misses and 31.5 ms at 331 — the 13.3 ms gap of
H0 is 1.8 ms — and the turnaround a cut (`turn`, wait minus device) falls from 3.75 to 2.5 ms
at no misses and from 6.4 to 3.1 at the floor: a device that is never idle answers a commit
faster too. The floor's step goes 189.7 -> 173.1 ms (-8.8%), the zero-miss step 44.5 -> 43.1.
The wide kernel armed at every gap (H1) is the opposite lesson: 45.7 ms of device time at no
misses, its scales running beside every one of the 48 frames.
