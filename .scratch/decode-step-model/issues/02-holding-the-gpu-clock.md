# 02 Holding the GPU clock so compute does not depend on the miss count

Status: resolved
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
| h2a | H=4 H_ARM=always | 31.58 | 46.57 | 10.97 | 32.38 | 175.52 | 331 / 47.7 | 132.80 | 6.00 | 1578 / 1578 |
| h2b | H=8 H_ARM=always | 36.91 | 53.14 | 10.83 | 35.34 | 197.54 | 331 / 47.7 | 150.97 | 5.97 | 1578 / 1578 |
| h2c | H=4 H_ARM=always H_INFLIGHT=1 | 30.31 | 51.38 | 17.26 | 32.12 | 175.44 | 331 / 47.7 | 133.06 | 5.70 | 1578 / 1578 |
| h3a | H=on H_ALU_ITERS=2048 H_ARM=always H_KERNEL=alu | 29.41 | 42.93 | 10.79 | 31.58 | 200.73 | 331 / 47.7 | 159.18 | 5.76 | 1578 / 1578 |
| h3b | H=on H_ALU_ITERS=8192 H_ARM=always H_KERNEL=alu | 29.70 | 43.10 | 10.69 | 31.52 | 173.05 | 331 / 47.7 | 132.71 | 5.47 | 1578 / 1578 |
| h3c | H=on H_ALU_ITERS=32768 H_ARM=always H_KERNEL=alu | 30.11 | 44.50 | 10.69 | 33.09 | 179.54 | 331 / 47.7 | 137.12 | 5.30 | 1578 / 1578 |
| h3d | H=on H_ALU_ITERS=8192 H_ARM=always H_INFLIGHT=1 H_KERNEL=alu | 30.21 | 44.51 | 10.70 | 31.49 | 173.33 | 331 / 47.7 | 132.48 | 5.36 | 1578 / 1578 |
| h4a | H=on H_ALU_ITERS=64 H_ARM=always H_INFLIGHT=1 H_KERNEL=spin | 33.35 | 51.71 | 14.73 | - | - | - | - | - | 1578 / 778 |
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

Rows h2a-c (the scale kernel at 4 and 8 MiB, every gap): 4 MiB comes within 0.9 ms of the
ALU kernel at the floor but costs 1.5-2 ms at no misses, 8 MiB slows the reads (151 ms of
copy against 133). h3c (32768 iterations, ~0.45 ms a kernel) blocks frames: 33.1 at the floor.
h3a (2048, ~0.03 ms a kernel) has the heater thread committing ~30k buffers a second and
the reads pay for it (159 ms of copy). h3b and h3d (8192 iterations, 0.106-0.115 ms a kernel
at full clock by the heater's own log) are the shape that works; two in flight reads 0.5 ms
better at no misses than one.

Repeated zero-miss runs of h3b's setting (`sm-h3b-d0`, `sm-e1-nocut`, `sm-e3-s11204`) put the
cut frames at 29.70 / 30.63 / 29.76 ms and the step at 43.1 / 44.1 / 43.2: about +-1.5% on a
48-step window, which is the noise the 256-step windows of issue 05 are for.

## The spin kernel, and the decision (2026-09-17 18:05)

A kernel that spins until the host clears a flag (`PIE_METAL_HEATER_KERNEL=spin`) would have
made the heater one dispatch an armed window. It does not work on this device: in a cut's
0.08 ms gap the flag is already cleared when the kernel starts (it exits after 2.8 us, useless),
and in the long gaps the kernel never sees the host's write and runs to its bound — at 1024
chunks (4.1 ms) it sits beside the frames and the cut frames read 36.8 ms at no misses
(`sm-h4b-d0`); at 64 chunks the host pays instead (`enc` 14.4 ms, `sm-h4a-d0`). The GPU's view
of a shared word written by the CPU mid-kernel is not coherent enough for this.

**Decision: `alu`, 1024 threads x 8192 FMAs, two in flight, armed at every gap** —
`PIE_METAL_HEATER=on PIE_METAL_HEATER_ARM=always PIE_METAL_HEATER_KERNEL=alu` with the default
iterations — is the conditioned setting for every measurement from here (`HEAT` in
`sm_sweep3.sh`). Cut frames 29.7-30.6 ms at no misses, 31.5 at 331: the residual 1.8 ms is a
device cost of the reads' gaps that the form's per-call term will carry. It becomes what
`PIE_METAL_HEATER=on` means once the measurements are in (one rebuild, at the end).
