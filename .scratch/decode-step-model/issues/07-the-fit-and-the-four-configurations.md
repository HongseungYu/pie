# 07 The fit, and the four configurations against it

Status: resolved
Type: research

Everything below runs under the conditioned settings (issues 02, 03, 05, 06): the ALU
heater at every gap, the CPU heater, 8192 seats for the large pool, the whole sequence
warmed before every `steps` run, the floor from 40-token `all_in_mem` windows.

## The floor is measured

Three reps, the resident pass's last 32 steps, only steps whose n-gram rows were cached
(`ple_ms` < 0.5; issue 05): floor = C + F = **37.69 ms** (reps 38.37 / 37.07 / 37.62; p10-p90
36.8-38.5). C = 33.17 (cut frames 30.11 + final frame 3.06), F = 4.53 (turnaround 3.00,
seating 0.20, n-gram rows 0.04, encode 0.97, after the walk 0.31).

## The read call is not quite a line

The planted grid (issue 04, 15 runs) and every natural run agree on the shape of a call's
cost against its misses, and nine of eleven natural runs from 512 to 8192 seats agree on
its values:

    m       1      2      4      8       (ms a call, natural misses)
    cost  0.70   1.05   1.75   3.22

i.e. 0.35 a miss from 1 to 2, 0.35 from 2 to 4, 0.37 from 4 to 8, over an intercept of
0.35 — nearly linear, with the first miss dearer than the next ones (sixteen threads read
2m requests in parallel, and one request alone pays the drive's first-byte latency in
full). A straight line `a + b * m * B` through that curve is a compromise, and which
compromise depends on where the misses sit: a pool at the floor takes 7 misses a call, the
large pool 2.

## The pick

`validate.py --fit-on minimax`: the (a, b) whose worst window error over the natural-miss
configurations (c1, c2, c3, mid) is smallest. The planted grid is shown against it but
kept out of the objective: a planted read fetches an expert the prime pass read seconds
before and the drive's cache serves it (issue 04: 0.71 ms a one-miss call against 0.80
for a cold one).

## Two runs the drive spoiled

`sm-mid-s3072` and `sm-mid-s4000`, run back to back after forty minutes of multi-GB/s
reads, read every call ~20% slower than the other nine runs at every m (one miss: 0.85 /
0.92 ms against 0.70; eight: 3.85 / 3.88 against 3.22) with r2 0.24-0.26 against 0.96-0.98
elsewhere. Nothing on the host moved (no swap growth, the same CPU load, power and GPU
clock, thermal `nominal`); the drive's own state is the remaining suspect. They were rerun
(`-b`); the originals stay in `out/` and out of the fit.

## Result (2026-09-17 22:10, `out/validate-final.txt`)

    step = 37.69 + sum over layers with m >= 1 of (0.368 + 0.1360 * m * 2.637)

Over the four configurations' clean windows: max misses -0.6 / +0.1 / +1.7%, min misses -1.7 /
-0.4%, zero misses +-1.75% (the floor reps' own spread), the mid points +0.4 / +0.4 / -0.3 /
+0.3%. Worst 1.75%. The planted grid, whose reads the drive's cache cheapens, sits up to +5%
above it and is the structural check only. The 128-token window (reads within seconds of a
55 GiB prefill, 0.60 ms a miss against 0.53 elsewhere) is -8% and is reported apart as a fifth
regime the drive owns. Full tables in `RESULTS.md`.

The pick lands on the cut-level read curve itself (0.387 + 0.357 m): the step pays its reads at
exactly their own cost and nothing else moves with the misses once the device is held at its
clock and the host awake.

## A launch pitfall, for the record

Runs started from an interactive zsh with the heater knobs in an unquoted variable (`$H`)
handed `profile_run.py` one word, `PIE_METAL_HEATER=on PIE_METAL_HEATER_ARM=always ...`,
and the heater read it as neither `on` nor a MiB count: **off**. That is why the first
reruns of `sm-mid-s3072-b` / `sm-mid-s4000-b` show the GPU at ~900 MHz and 42 ms of device
time, and why the d1-d4 diagnostics of issue 05 ran without a heater (their miss counts
and `ple_ms` are unaffected; their device times are not the conditioned ones). Every run
in the tables below came through the bash scripts, where the words split; `sm_run.sh` now
refuses a `PIE_METAL_HEATER` word it cannot parse.
