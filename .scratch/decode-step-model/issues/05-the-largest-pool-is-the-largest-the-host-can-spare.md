# 05 The largest pool is the largest the host can spare

Status: resolved
Type: research

Configuration 2 asks for the largest pool that boots, about 11204-12000 seats on this 48 GB
box. Measured at 11204 seats (28.85 GiB of seats, ~35 GiB wired with the planes), the host
between cuts — the walk's own time outside every wait, read and seating, `enc` in
`stepmodel.py` — costs 7.2 ms a step at no misses, against 1.2-1.7 ms at 1536-6144 seats,
and it decays over a long run (7.8 -> 4.4 ms over 1000 steps in `cold-s11204`). It does not
follow the miss count (r2 0.00 against L within a run) and it is not the cut log (7.19 ms
with the log off, `sm-e1-nocut`).

`vm_stat` during an 11204-seat run: **5231 pages free (80 MB)**, 714k pages in the
compressor, 1.25 GB swapped. The host's other memory (8-9 GB compressed before any run) has
nowhere to go once 35 GiB are wired, so the encode path — the Metal driver, the walk — runs
against a compressor, and so do the reader threads: a one-miss call costs 0.99 ms here
(`sm-g1-all1`) against 0.71 with dense reads at 4000 seats.

That cost is real, but it is the operating system's, not the engine's or the model's: it
depends on what else the machine holds, and it drifts. A model with one F cannot carry it,
and a simulator should not have to. So configuration 2 is the largest pool that leaves the
host its memory: `sm_probe_seats.sh` tries 8192 / 9216 / 10240 / 11204 with `all_in_mem`
128 tokens and takes the largest whose run ends with 2 GB or more free and whose resident
pass misses nothing.

Every run's memory before and after is now on `out/sm_runs.log`.

## The probe (2026-09-17 18:20, `sm-probe-s*`, `all_in_mem` 128 tokens, heater h3b)

| seats | resident pass misses/step | prime pass `enc` | resident `enc` | 1-miss call (median) |
|---|---|---|---|---|
| 8192 | 70.4 | 7.65 | 3.05 | 0.74-0.75 |
| 9216 | 58.0 | 7.73 | 7.60 | 0.76 |
| 10240 | 46.4 | 7.64 | 7.47 | 0.78 |
| 11204 | 23.1 | 7.85 | 7.20 | 0.96-1.07 |

Two findings. **No pool that boots holds a 128-step window of this sequence's head**: the
prime pass touches about 100 new (layer, expert) pairs a step (128 x ~98 = 12.5k, plus the
ring's 1024), more than 11204 seats. Sixty-four steps (7.1k + 1k) fit at 11204 and just about
at 9216. The sequence's tail is a loop (85 distinct tokens over positions 768-1024), so
replaying it (`profile_run.py --forced-offset 768`) is the way to a long zero-miss window.

**`enc` is 7.6 ms early in the life of every pool from 8192 up**, not only at 11204, and
it falls with the run's age (8192's resident pass: 3.05). It does not follow the misses.
The one host cost inside the walk that depends on the page cache is the n-gram (PLE) row
gather, mmap'd from a 25.6 GB table: each step needs rows at random pages of it, and once
the pool has wired most of the host's memory those pages are evicted between uses and
re-faulted from disk — the previous effort saw `ple_host` swing 0.1-1.2 ms with the page
cache at 4000 seats. From commit (this one) the fire record carries `ple_ms`, the gather's
own host time, so the reader shows it apart from the encode.

## `ple_ms` measured (2026-09-17 18:40, `sm-d1-tail-s8192`, `sm-d2-tail-s11204`; the sequence's tail, 128 tokens)

| seats | pass | misses/step | `ple_ms` | `enc` (encode proper) | swap during the run |
|---|---|---|---|---|---|
| 8192 | prime | 94.7 | **6.10** | 1.11 | 1.97 -> 2.52 GB |
| 8192 | resident | 76.5 | **0.06** | 1.04 | |
| 11204 | prime | 78.5 | 5.77 | 1.00 | 2.52 -> 3.07 GB |
| 11204 | resident | 37.2 | **5.64** | 0.91 | |

That is the whole of it: the walk's encode proper is 1.0-1.1 ms a step at any pool, and the
n-gram row gather is 0.06 ms when its pages are cached and ~6 ms when each step faults them
in. At 8192 seats the prime pass warms them and the resident pass finds them; at 11204 the
host is paging (swap grows half a gigabyte during a one-minute run) and the resident pass
faults them again 20 seconds later. A model with one F can hold the first case and not the
second, so the largest pool this effort measures is the largest at which the n-gram pages
survive between requests, and every `steps` run warms the whole sequence first.

Also from these two runs: replaying the sequence's looping tail does not shrink the routed
union (76 misses a step in the resident pass at 8192, 37 at 11204): a repeated token in a
different context routes to different experts. The zero-miss window stays what a pool holds
of the head, about 64 steps.

## Answer (2026-09-17 19:05)

`sm-d3-head-s8192` and `sm-d4-head-s9216` (the sequence's head, 64 tokens):

| seats | resident misses/step | `ple_ms` | step (last 40) | note |
|---|---|---|---|---|
| 8192 | 12.4 (153, 112, 38 ... then 1-7) | 0.03 | 45.9 | the resident's prefill borrows the 1024 coldest seats, which at a full pool are the earliest decode steps' experts |
| 9216 | **0.0** | 0.09 | **38.56** (p10 38.21, p90 38.94) | clean |
| 11204 | 0.0 (issue 01/02 runs) | ~7 | 43.2 | the n-gram pages fault every step |

**Configuration 2-4 run at 9216 seats** (24.3 GiB of seats; the largest at which the n-gram
rows stay cached between requests), the zero-miss window is **48 tokens** (a 64-step head
fits at 9216 with the ring's 1024 to spare; 48 leaves margin), and every `steps` run warms
the whole 1024-token sequence first. The floor is 38.6 ms, not the 43-45 of the earlier
11204-seat runs, whose extra was page faults: C = 33.4 (cut frames 30.3 + final frame
3.0), F = 5.2 (turnaround 3.9, seating 0.2, n-gram rows 0.1, encode 0.8, after the walk 0.3).

## Correction (2026-09-17 19:20): 9216 is not stable; 8192 is

The three zero-miss reps at 9216 (`sm-c3-n48-a/b/c`) found their n-gram rows evicted again
between the prime and the resident pass: `ple_ms` 5.96 / 5.13 / 10.75 in the resident passes
(falling to 0.1 only in a pass's last steps), steps 42.8 / 41.9 / 47.6. `sm-d4-head-s9216`'s
0.09 was luck: `sm-d3-head-s8192` had warmed the same rows a minute before and they survived
one more prefill. At 8192 seats both `sm-d1-tail-s8192` (128 tokens) and `sm-d3-head-s8192`
(64) kept them (0.06, 0.03). So the pool is **8192 seats** (21.6 GiB of seats), and the
zero-miss window **40 tokens** (48 steps' union with the two ring borrows is tight at 8192:
`sm-d3` at 64 lost the earliest steps' experts to the borrow). The Phase 3 launcher checks
the c3 reps' resident `ple_ms` before spending the hour.
