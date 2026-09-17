# 03 What a read call costs after the host has been idle, and what keeps it warm

Status: resolved
Type: research

The cut log said a layer's read call costs 0.35 ms when reads are dense and 0.83 ms when
they are rare (issue 16 of the previous effort), and the user's form has one constant for
it. Before touching the engine: is it the SSD, the CPU, or the threads?

`scratch/qwen38-profile/tools/ssd_gap.c`: one call = m experts, per expert one 1.8 MiB and
one 0.9 MiB `F_NOCACHE` pread at random offsets of the 98 GiB artifact, chunked over up to
16 threads spawned per call exactly as `FileWriter::pread` does; between calls the caller
sleeps G ms. Machine idle, thermal `nominal`, 2026-09-17 16:35. Mean ms a call
(`out/ssd_gap.tsv`):

| variant | m | G=0 | 0.5 | 1 | 2 | 5 | 10 | 30 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| spawn (as pie) | 1 | 0.71 | 0.72 | 0.79 | 1.01 | 1.15 | 1.35 | 1.44 | 2.34 |
| spawn | 4 | 1.84 | 1.78 | 1.78 | 1.82 | 2.44 | 2.58 | 2.64 | 3.36 |
| pool (16 threads spin-waiting) | 1 | 13.4 | 12.6 | 12.9 | 10.8 | 11.4 | 10.3 | 10.9 | 10.7 |
| keeper (4 KiB pread every 1 ms) | 1 | 0.52 | 0.63 | 0.64 | 0.75 | 1.04 | 1.27 | 1.26 | 1.31 |
| keeper | 4 | 1.82 | 1.73 | 1.74 | 1.79 | 2.31 | 2.51 | 2.50 | 2.55 |
| **cpu (one thread spinning)** | 1 | 0.58 | 0.53 | 0.54 | 0.53 | 0.55 | 0.57 | 0.58 | 1.29 |
| cpu | 4 | 1.24 | 1.29 | 1.30 | 1.29 | 1.28 | 1.33 | 1.26 | 1.97 |

- The engine's pattern (spawn) pays for idleness from about 2 ms on: +0.3 ms a call at a
  2 ms gap, +0.65 at 10 ms, +1.6 at 100 ms. A pool that misses in 14 of 48 cuts leaves the
  read path idle for 2-10 ms between calls, which is exactly the 0.83 ms it was charged.
- Sixteen spinning reader threads oversubscribe the 14 cores and stall the reads: 10-13 ms.
  Not a cure.
- A tiny read every millisecond does not hold the cost flat past a 2 ms gap: the drive is
  not what sleeps (or not what matters).
- One spinning thread holds it flat, 0.53-0.58 ms at m=1 and 1.24-1.33 at m=4, from no gap
  to 30 ms; it also takes 0.15 ms off the back-to-back call. Only a 100 ms gap gets past
  it, and no decode leaves the read path idle that long. So the wake-up is the CPU's: thread
  creation and scheduling on cores that have gone to sleep, and the P-cluster's clock with
  them.

Taken: `PIE_METAL_CPU_HEATER=1` (`device/spinner.rs`), a thread spinning for the load's
life, off by default, on for every measured run from here. Its worth in the engine is
issue 04's gate: `fitcuts` must give one `a` and one `b` at 1024, 4000 and 11204 seats.
