# 09 The read call above 8192 seats: a step at the memory's edge, modelled as a ramp

Status: resolved
Type: research

The user pins up to ~42 GB of the 48 on this box (about 13200 seats) and asked whether the
read cost's rise above 8192 seats is the compressor/swap's I/O contention, and for a model of
it, continuous with the ≤ 8192 model at the boundary, within ±3%.

## Environment

Other Claude sessions closed by the user; Safari, Rhino, mediaanalysisd/photoanalysisd killed
here (launchd respawns the analysis daemons; they stayed small). Spotlight (`mds_stores`,
0.6 GB) and Time Machine need `sudo mdutil -a -i off` / `sudo tmutil disable`, which this session
cannot run; their activity was sampled instead. Other resident memory during the sweep: ~1.5 GB.
13200 seats boots with `pie-qwen38-big.toml` (`device_weight_budget = "44GiB"`,
`gpu_mem_utilization = 0.96`); pinned GB below = seats x 2.637 MiB + 6.5 GB (planes, arena, KV,
tables).

## What the sweep shows (`sm5-*`, 2026-09-18)

Planted misses (`all:1`, `every:4:10`, `all_in_mem` 40 tokens, primed) give the read call at
known m at each pool; the `steps` windows (warm 1024 + 1024, last 256) give what the model
must predict; beside every run, `vm_stat` deltas and `iostat`.

 seats | pinned GB | 1-miss call (mean/median) | 10-miss call | floor (0-miss steps) | ple join | natural: misses/L | step | copy | least free | swapout/s | compress/s 
---|---|---|---|---|---|---|---|---|---|---|---
 8192 | 28.6 | 0.737 / 0.676 (1536) | 3.756 / 3.738 (384) | 37.76 | 0.13 | - | - | - | 0.1G | 0 | 7052 
 9216 | 31.4 | 0.715 / 0.696 (1536) | 4.189 / 3.857 (384) | 37.80 | 0.01 | 40.5 / 22.9 | 60.25 | 22.21 | 0.1G | 0 | 2009 
 10240 | 34.2 | 0.788 / 0.726 (1536) | 4.334 / 4.009 (384) | 38.07 | 0.14 | 21.7 / 14.2 | 50.55 | 12.86 | 0.1G | 0 | 9677 
 11264 | 36.9 | 0.971 / 0.962 (1536) | 4.736 / 4.325 (384) | 37.45 | 0.01 | 9.5 / 7.4 | 45.38 | 8.15 | 0.0G | 0 | 11040 
 12288 | 39.7 | 1.051 / 0.963 (1536) | 4.358 / 4.321 (384) | 37.32 | 0.01 | 6.3 / 5.2 | 43.22 | 5.86 | 0.1G | 0 | 12522 
 13200 | 42.1 | 0.983 / 0.962 (1536) | 4.390 / 4.309 (384) | 37.25 | 0.01 | 4.9 / 4.1 | 43.74 | 5.89 | 0.1G | 0 | 13952 

Natural runs, cut level, a one-miss call (mean / median): 9216 0.70, 10240 0.72, 11264 0.99,
12288 1.02, 13200 1.36 / 1.02 (the mean carries one drive stall in the window's first 64
steps: 2.64 ms a miss there, 0.99-1.03 after).

**It is a step, not a slope.** A one-miss call costs 0.68-0.73 ms up to 10240 seats (34.2 GB
pinned) and 0.96-1.02 ms from 11264 (36.9 GB) on, the same at 12288 and 13200; a ten-miss
call 3.74-4.01 up to 10240 and 4.31-4.33 from 11264. Solving the two: a ≈ 0.59, b ≈ 0.37 ms a
miss above the edge against 0.35-0.39 / 0.34-0.36 below — the call's fixed part rises by
~0.22 ms, the per-miss part by ~6%. The floor does not move (37.25-38.07 across the sweep), nor
device time (33.2-33.9), nor the prefetched n-gram join (0.01-0.14).

**It is not swap I/O.** No run swapped out at all (`swapouts/s` 0 throughout); the compressor
worked at boot (7k-50k compressions/s while the pool wired 21-36 GB) and free memory sat at
0.0-0.1 GB during every run from 8192 up — the OS keeps the file cache at the edge whatever the
pool. What differs from 11264 on is that the pinned memory has eaten the last ~2 GB of
reclaimable pages: an uncached pread must then find its kernel-side pages on the reclaim path
at each call, a fixed latency per call — the same ~0.2-0.3 ms the earlier 11204-seat runs paid
(issue 04's 0.99 ms one-miss call). So the answer to the user's question: contention, yes, but
for pages, not for the disk.

## The model (`out/validate-sm5.txt`, config `mac_m4pro_qwen38flash_np1_v4.json`)

A ramp, so the function is continuous with the <= 8192 model at its foot and flat past its top —
the step the data show, spread over the 2.7 GB between the last untouched size and the first
fully affected one, where 10752 seats (35.5 GB) measured halfway (a x 1.31):

    P  = pinned GB = N x 2.637 MiB + 6.5 GB (planes, arena, KV, tables; ~1.5 GB of other resident memory beside)
    r  = clamp((P - 34.2) / 2.7, 0, 1)          (N = 10240 -> 0, N = 11264 -> 1 on this box)
    a(P) = 0.368 x (1 + 0.64 r)     b(P) = 0.1360 x (1 + 0.04 r) ms/MiB     floor = 37.05 (unchanged)
    step = floor + sum over layers with m_l >= 1 of ( a(P) + b(P) x m_l x 2.637 )

The inflation ratios come from the planted cut curves pooled below (8192-10240: 0.375 + 0.372 m)
and above (11264-13200: 0.614 + 0.388 m) the edge, which are unbiased by the planted reads'
cheapness; the floor's own terms did not move, so the floor has no slope.

Against the natural windows (`steps`, last 256, clean steps = drive not stalled):

| config | tag | seats | steps (clean) | misses/step | L/step | device | copy | ple | measured | predicted | error | error, all steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c2 | sm5-c2-s10240-a | 10240 | 256 (256) | 21.7 | 14.22 | 33.57 | 12.86 | 0.01 | 50.55 | 50.06 | -0.97% | -0.97% |
| c2 | sm5-c2-s11264-a | 11264 | 256 (253) | 9.5 | 7.47 | 33.28 | 8.18 | 0.02 | 45.39 | 45.12 | -0.60% | -0.67% |
| c2 | sm5-c2-s12288-a | 12288 | 256 (243) | 6.2 | 5.20 | 33.29 | 5.73 | 0.11 | 42.98 | 42.51 | -1.10% | -1.58% |
| c2 | sm5-c2-s12288-b | 12288 | 256 (97) | 7.2 | 5.93 | 33.49 | 6.60 | 1.27 | 44.06 | 43.31 | -1.69% | -11.77% |
| c2 | sm5-c2-s13200-a | 13200 | 256 (209) | 5.0 | 4.23 | 33.31 | 4.73 | 0.50 | 42.05 | 41.48 | -1.36% | -5.45% |
| c2 | sm5-c2-s13200-b | 13200 | 256 (248) | 4.9 | 4.15 | 33.25 | 4.67 | 0.01 | 41.95 | 41.37 | -1.37% | -1.50% |
| c2 | sm5-c2-s9216-a | 9216 | 256 (256) | 40.5 | 22.86 | 33.80 | 22.21 | 0.01 | 60.25 | 60.00 | -0.42% | -0.42% |

Every natural window within -0.4 .. -1.7%. Two of them met drive stalls ("all steps" column:
12288-b -11.8% with 159 of 256 steps stalled, 13200-a -5.5%): those seconds are the drive's,
as in issue 07, and are shown rather than modelled. The planted grid under the same ramp
(structural reference; a planted read is cheaper than a cold one at m = 1 and the ten-miss
calls above the edge came out ~7% dearer than the ramp's b):

| config | tag | seats | steps (clean) | misses/step | L/step | device | copy | ple | measured | predicted | error | error, all steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c4 | sm5-p1-s10240 | 10240 | 37 (36) | 48.0 | 48.00 | 33.65 | 37.84 | 0.12 | 75.75 | 71.93 | -5.05% | -5.24% |
| c4 | sm5-p1-s10752 | 10752 | 37 (37) | 48.0 | 48.00 | 33.36 | 43.61 | 0.01 | 81.11 | 77.85 | -4.02% | -4.02% |
| c4 | sm5-p1-s11264 | 11264 | 37 (37) | 48.0 | 48.00 | 33.28 | 46.74 | 0.01 | 84.20 | 83.92 | -0.33% | -0.33% |
| c4 | sm5-p1-s12288 | 12288 | 37 (36) | 48.0 | 48.00 | 33.28 | 50.31 | 0.08 | 87.64 | 83.92 | -4.24% | -4.43% |
| c4 | sm5-p1-s13200 | 13200 | 37 (37) | 48.0 | 48.00 | 33.26 | 47.33 | 0.01 | 84.64 | 83.92 | -0.85% | -0.85% |
| c4 | sm5-p1-s8192 | 8192 | 37 (34) | 48.0 | 48.00 | 33.55 | 35.13 | 0.25 | 72.78 | 71.93 | -1.17% | -1.49% |
| c4 | sm5-p1-s9216 | 9216 | 37 (37) | 48.0 | 48.00 | 33.61 | 34.31 | 0.01 | 72.13 | 71.93 | -0.29% | -0.29% |
| c4 | sm5-p10-s10240 | 10240 | 37 (37) | 120.0 | 12.00 | 33.89 | 52.15 | 0.02 | 91.00 | 84.50 | -7.14% | -7.14% |
| c4 | sm5-p10-s10752 | 10752 | 37 (37) | 120.0 | 12.00 | 33.81 | 55.89 | 0.02 | 94.52 | 86.75 | -8.22% | -8.22% |
| c4 | sm5-p10-s11264 | 11264 | 37 (37) | 120.0 | 12.00 | 33.75 | 56.72 | 0.01 | 95.41 | 89.05 | -6.67% | -6.67% |
| c4 | sm5-p10-s12288 | 12288 | 37 (37) | 120.0 | 12.00 | 33.85 | 52.44 | 0.01 | 91.22 | 89.05 | -2.38% | -2.38% |
| c4 | sm5-p10-s13200 | 13200 | 37 (37) | 120.0 | 12.00 | 33.81 | 52.98 | 0.01 | 91.72 | 89.05 | -2.91% | -2.91% |
| c4 | sm5-p10-s8192 | 8192 | 37 (37) | 120.0 | 12.00 | 33.76 | 45.12 | 0.02 | 83.72 | 84.50 | +0.94% | +0.94% |
| c4 | sm5-p10-s9216 | 9216 | 37 (37) | 120.0 | 12.00 | 33.93 | 50.56 | 0.02 | 89.40 | 84.50 | -5.48% | -5.48% |

A free minimax over every window (`--fit-on fitknee`) would put the ramp at 30.75-36.75 GB with
a x 1.51 and b x 1.19, worst 4.2% with the planted rows in the objective — a compromise toward
the planted ten-miss calls; the cut-curve ramp above is the one whose knee the data locate and
whose natural-window errors are smallest, so it is the pick.

Beyond ~42 GB pinned there is no data (13200 seats is the box's ceiling with the big config);
the ramp is flat there by construction. On another box, or with more resident memory beside the
engine, the knee moves with the free memory left: it is at "the last ~2 GB of reclaimable
pages", not at a seat count.
