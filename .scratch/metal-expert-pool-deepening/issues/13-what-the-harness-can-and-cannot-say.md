# 13 What the harness can and cannot say about a step

Status: resolved
Type: research

Every gate on this branch quoted `step_ms_mean` and `step_ms_last_half` out
of `profile_run.py --mode clean`. Those are not measurements of a step. The
mode fires three requests over the same 1k prompt — 1 token, 512, 1024 —
and derives

    mean = (T(1024) - T(1))   / 1023
    tail = (T(1024) - T(512)) / 512

so each figure is the difference of two *separate* requests, each of which
re-prefills the prompt. A prefill here reads about 55 GiB (the ring pulls
whole layers), so it is bound by the drive, whose effective rate swings run
to run: the same harness has logged 6.96 GB/s and 11.17 GB/s for the load's
own planes. When a run's prefill is slow, `T(1)` and `T(512)` grow while
`T(1024)` barely moves, and both derived figures improve. They read as a
speed-up and are the opposite.

The evidence, all at 4000 seats with the heater on:

| run | T(1) | T(512) | T(1024) | mean | last half |
|---|---|---|---|---|---|
| baseline rep0 | 7.08 | 82.04 | 140.49 | 130.40 | 114.15 |
| baseline rep1 | 7.49 | 77.85 | 140.66 | 130.17 | 122.66 |
| issue 02 | 8.13 | 77.28 | 133.30 | 122.36 | 109.40 |
| issue 04 | 10.82 | 83.84 | 133.35 | 119.78 | 96.70 |
| issue 05 | 11.60 | 91.31 | 132.42 | 118.10 | 80.28 |
| cleanup | 7.28 | 71.77 | 134.36 | 124.22 | 122.24 |

The cleanup run is issue 05's code plus three commits that delete dead code
and move counters: it cannot have changed a step. Its last-half figure is
122.24 ms against issue 05's 80.28 ms, a 52% swing on identical behaviour.
The baseline's own two reps differ by 7% on the same figure.

`T(1024)` is the number that holds still: 140.5 before this branch's
seat-space fix, then 133.3 / 133.4 / 132.4 / 134.4 across every commit
since, inside 1.5%.

## What to quote from now on

- **`T(1024)`**, with at least two reps, as the end-to-end number.
- **The pool's own counters** from the `expert-cache:` line — hit rate,
  bytes read, copies from seats — which are counted, not differenced.
- Never `step_ms_last_half` on its own, and never `step_ms_mean` across
  builds without the raw `T` values beside it.

## What the branch actually did to the numbers

`T(1024)` 140.5 -> 133.3 at issue 02, flat since. Hit rate 67.8-67.9%
throughout. Disk 931 -> 943 GiB at issue 05, because a borrowed ring finds
fewer experts already seated to copy from. The memory at one knob fell
12.94 -> 10.30 GiB. Correctness held everywhere: every run decoded the
recorded teacher-forced tokens exactly.
