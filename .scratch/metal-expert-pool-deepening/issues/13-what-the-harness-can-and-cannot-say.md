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

## The comparison that was missing

Every gate above ran at `PIE_EXPERT_CACHE=4000`, but that knob does not
mean the same thing before and after issue 05. It used to buy 4000 pool
seats *plus* a reserved ring of 1024 (12.94 GiB); it now buys 4000 seats
total, of which a prefill borrows 1024 and gives them back (10.30 GiB). The
runs were comparing different amounts of device memory.

Three reps of each, same build (the cleanup commit), heater on:

| knob | memory | T(1024) per rep | hit rate | disk |
|---|---|---|---|---|
| 4000 | 10.30 GiB | 148.0 / 132.4 / 132.6 | 67.8% | 2497 GiB |
| 5024 | 12.94 GiB | 122.2 / 122.4 / 128.0 | 72.6% | 2184 GiB |

At the memory the old build used at knob 4000, this build wants knob 5024,
and there it holds 72.6% of its routes against the baseline's 66% and runs
`T(1024)` at about 122.4 s against the baseline's 140.5 s. The reserved
ring's 1024 seats were memory no decode could ever hit; borrowed, they are
seats the decode uses between prefills, and a prefill gives them back
holding the last two layers it read. That is the change's win, and it is
worth about 13% end to end at equal memory — not the step figures first
quoted.

At the same knob instead of the same memory, the trade reads the other way
and honestly: 2.64 GiB less memory for 1024 fewer usable seats, 132.6 s
against 140.5 s.

The first rep of a run can be an outlier (148.0 above): quote the median of
three, or at least two agreeing reps.

## What the branch actually did to the numbers

At one knob: `T(1024)` 140.5 -> 133.3 at issue 02, flat since, with the
memory falling 12.94 -> 10.30 GiB at issue 05. At one memory: 140.5 ->
122.4 s and 66% -> 72.6% of routes held, which is the borrowed ring paying
for itself. Correctness held everywhere: every run decoded the recorded
teacher-forced tokens exactly.

## The measurement that answers the question asked

The question is: from a cold pool, one prefill and 1024 decoded tokens,
what does a step cost over the last 512? That is `--mode steps --warmup 0
--reps 1`, which fires one teacher-forced request against a freshly started
server and logs every fire. A step is the gap between two fires' `t_ms`, so
one request answers for its own steps. `coldsteps.py` reads the last N off
that log.

`--reps` above 1 does not repeat this: only the first request meets a cold
pool. Repeat by restarting the server.

Cold, heater on, 1k prompt, 1024 teacher-forced tokens, this build:

| knob | memory | last 512: mean | median | p10 | p90 | hit rate | misses/step |
|---|---|---|---|---|---|---|---|
| 4000 | 10.30 GiB | 117.74 ms | 116.06 | 89.83 | 146.51 | 0.705 | 141.8 |
| 5024 | 12.94 GiB | 107.76 ms | 105.96 | 81.62 | 138.08 | 0.754 | 118.2 |

The log accounts for the request: 1022 steps summing to 125.0 s (4000) and
115.2 s (5024) against wall times of 133.6 s and 126.3 s, the rest being the
prefill that the first decode fire's `t_ms` marks at 8.5 s and 11.0 s. The
first 512 steps cost more than the last 512 (126.81 against 117.74; 117.68
against 107.76), which is the pool filling after a prefill took the ring —
the effect the `clean` mode's tail figure was trying to see and could not.

A step's spread is wide (p10 to p90 is 57 ms at 4000 seats) because a step's
cost is its miss count, so quote the median with the quantiles, not a mean
alone.

## Repeating a cold request without restarting the server

`PIE_EXPERT_CACHE_COLD=1` (engine commit 699e2f87) drops every expert the
pool holds as a prefill opens, so `--reps` repeats the same cold request.
Verified at 4000 seats, a 256-token prompt, 512 teacher-forced tokens,
three reps in one server:

| rep | wall | decode hit | last 256: mean | median | misses/step | read |
|---|---|---|---|---|---|---|
| 0 | 69.2 s | 0.7014 | 122.81 ms | 123.03 | 153.5 | 101.2 GiB |
| 1 | 69.3 s | 0.7014 | 122.96 ms | 123.08 | 153.5 | 101.2 GiB |
| 2 | 71.3 s | 0.7014 | 123.02 ms | 122.59 | 153.5 | 101.2 GiB |

Hit rate, misses a step and bytes read are identical to the digit across
all three: the requests walked the same cache trajectory, which is what
the clean is for. Step time agrees inside 0.2%.

With the knob off the same three reps ran 71.7 / 79.1 / 84.0 s and their
step distributions came apart (the fit's r2 fell 0.960 -> 0.221 -> 0.228).
That is not a clean control, though: the off run followed the on run, so
the drive and the SoC had already been working for four minutes. What the
pair does show is that the knob delivers repeatability, not that its
absence costs 15%.

Note the steady state is the same either way: the last 256 steps hold
0.680 of their routes whichever way the knob is set. At 4000 seats over
24576 (layer, expert) pairs the starting cache washes out; the clean's
worth is that the whole trajectory, early steps included, repeats.

## Seats against the step

One cold request each (fresh server, 1k prompt, 1024 teacher-forced
tokens, heater on), last 512 steps:

| seats | memory | step ms (median) | tok/s | hit rate | misses/step | read ms/step | ms a miss (measured) | ms a miss (fitted) | step at no misses |
|---|---|---|---|---|---|---|---|---|---|
| 1536 | 3.96 GiB | 167.82 | 5.96 | 0.451 | 263.7 | 112.51 (66.9%) | 0.427 | 0.365 | 72.13 |
| 3072 | 7.91 GiB | 129.42 | 7.73 | 0.639 | 173.2 | 76.78 (58.6%) | 0.443 | 0.403 | 61.31 |
| 4000 | 10.30 GiB | 116.06 | 8.62 | 0.705 | 141.7 | 64.29 (54.6%) | 0.454 | 0.417 | 58.64 |
| 5024 | 12.94 GiB | 105.96 | 9.44 | 0.754 | 118.2 | 55.24 (51.3%) | 0.468 | 0.432 | 56.70 |
| 6144 | 15.82 GiB | 97.46 | 10.26 | 0.799 | 96.3 | 46.96 (47.6%) | 0.488 | 0.455 | 54.84 |

A miss costs 0.43-0.49 ms whatever the pool size (2.637 MiB an expert, so
about 0.17 ms a MiB), and the step with no misses falls slowly from 72 to
55 ms as the pool grows, which is the cut's own bookkeeping thinning out.
Everything else is the miss count: seats buy hit rate, hit rate buys time,
and the return per seat is falling — the first 1536 seats past the floor
take 38 ms off a step, the last 1120 take 8.
