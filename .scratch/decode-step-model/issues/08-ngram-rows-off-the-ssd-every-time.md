# 08 The n-gram rows off the SSD every time, and ahead of the fire

Status: resolved
Type: task

Issue 05 found the PLE (n-gram) row gather to be the page cache's time: 0.03-0.1 ms a step
with the rows' pages cached, 6-7 ms with them evicted, and it was the eviction — any pool
from 9216 seats up, or a sequence's first pass — that set the "largest pool" at 8192 and
made F a property of the machine's state. The user's ask: read the rows from the SSD every
time, like the experts, so the latency is one number, and read them ahead of the fire so
the step does not pay it at all.

## No separate file

`Slab::open` already knows each band's row-0 offset in the artifact and a row's stride
(`gather.rs`), the artifact's file already carries `F_NOCACHE` (the tier set it), and the
tier's batched pread (`Store::write_from_file`, sixteen threads) takes exactly the jobs the
slab needs. A row is ~90 bytes, one uncached block; sixteen heads are sixteen preads a step.
Copying the table into its own file would add 25.6 GB and a second source of truth for
nothing.

## Ahead of the fire

The hasher's ids are a function of the fire's token ids and each slot's last `ngram-1`
tokens, which the kernel keeps in a recurrent state slab in shared memory. The kernel crate
carries a bit-exact host reference (`kernels_metal::attn::ple::reference::walk`), and the
engine has the tokens per lane and slot before the walk (`serve.rs::enqueue`), so the slab
reads a copy of each slot's window off the state slab (`Pools::read_state`), names the rows,
seats them and lands them on a thread as the fire opens; the hasher's cut joins the thread
and reads the device's own ids as before. An id the host did not name is read at the cut
and counted (`ple_prefetch_misses`): a wrong guess costs a read, never a wrong row.

## A/B (2026-09-18, 8192 seats, `all_in_mem` 40 tokens, heater on; `sm-ple-*`)

| path | prime pass `ple_ms` (first touch) | resident pass `ple_ms` | prefetch misses | resident step |
|---|---|---|---|---|
| mapping (old) | 7.23 | 0.09 | - | 37.35 |
| uncached pread at the cut | 0.63 | 0.68 | - | 37.63 |
| pread, prefetched at fire open | **0.01** | **0.19** | **0** (both passes) | **37.02** |

The pread path costs the same whether or not the pages were ever seen — the page cache is
out of the step. The prefetch hides it: the host's ids matched the device's in every step of
the prefill (chunked walk) and the decode, 16 rows a step, none read at the cut. Defaults:
`PIE_PLE_SOURCE=pread`, `PIE_PLE_PREFETCH=1`.

## Re-measurement (`sm4-*`, 2026-09-18; `out/validate-sm4.txt`)

The floor moved by the n-gram term and a little turnaround: **37.05 ms** (reps 37.34 / 36.89 /
36.95), C = 32.86 (cut frames 29.83 + final frame 3.04), F = 4.19 (turnaround 2.69, seating
0.21, n-gram rows 0.02, encode 0.96, after the walk 0.31). Nothing else changed, so the read
call's a, b were kept at 0.368 / 0.1360 and checked, not refit:

| config | tag | seats | steps (clean) | misses/step | L/step | device | copy | ple | measured | predicted | error | error, all steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c3 | sm4-c3-n40-a | 8192 | 32 (30) | 0.0 | 0.00 | 33.01 | 0.00 | 0.48 | 37.34 | 37.05 | -0.77% | -2.17% |
| c3 | sm4-c3-n40-b | 8192 | 32 (32) | 0.0 | 0.00 | 32.80 | 0.00 | 0.01 | 36.89 | 37.05 | +0.44% | +0.44% |
| c3 | sm4-c3-n40-c | 8192 | 32 (32) | 0.0 | 0.00 | 32.79 | 0.00 | 0.04 | 36.95 | 37.05 | +0.29% | +0.29% |
| c1 | sm4-c1-s1024-b | 1024 | 256 (256) | 331.4 | 47.68 | 34.63 | 132.57 | 0.01 | 172.57 | 173.45 | +0.51% | +0.51% |
| c1 | sm4-c1-s1024 | 1024 | 256 (255) | 331.5 | 47.68 | 34.54 | 140.66 | 0.04 | 180.65 | 173.49 | -3.96% | -3.97% |
| c2 | sm4-c2-s8192-a | 8192 | 256 (256) | 52.8 | 26.57 | 34.06 | 28.26 | 0.01 | 66.69 | 65.78 | -1.36% | -1.36% |
| c2 | sm4-c2-s8192-b | 8192 | 256 (256) | 52.8 | 26.57 | 33.98 | 28.17 | 0.01 | 66.52 | 65.78 | -1.11% | -1.11% |
| mid | sm4-mid-s3072 | 3072 | 256 (256) | 196.6 | 44.57 | 34.29 | 84.60 | 0.01 | 123.99 | 123.96 | -0.02% | -0.02% |
| mid | sm4-mid-s6144 | 6144 | 256 (256) | 116.4 | 38.95 | 33.95 | 54.90 | 0.01 | 93.65 | 93.14 | -0.54% | -0.54% |

Every row within 1.4% but the first floor-pool rep, whose reads ran ~5% slower at every m
(m=4: 1.87 ms against 1.77; m=8: 3.39 against 3.25 — the drive's state for that run, no stall
bursts); its second rep is +0.5%. A minimax over the new runs alone would move a, b only to
0.386 / 0.1385 (worst 2.3% with the slow rep in), so the parameters stand.

**Does the pool limit lift?** With the rows no longer the page cache's, 9216 / 10240 / 11204
seats were probed again (`sm4-probe-s*`, `all_in_mem` 64). They no longer *fault* pages, but
at 9216 and 11204 the host sat at 0 GB free and every read slowed: the n-gram preads took
5.5-5.7 ms to land (the join at the cut) and a one-miss expert call cost 1.13 ms against 0.78
(10240 happened to survive: 0.04 ms, floor 37.21). Memory pressure now reaches the I/O path
itself rather than the page cache, so **8192 seats stays the largest pool this box measures
cleanly**; the change removed the cache dependence, not the pressure.

