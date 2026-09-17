# 15 Does the pool's size touch a kernel, or only the miss count?

Status: resolved
Type: research

The claim to test: a seat is where an expert's bytes sit, not what the
matmul does with them, so the pool's size should change how many misses a
step takes and nothing else. `--diag kernel-profile` device-times every
kernel of a fire, so it answers directly.

Four runs, 48 decoded tokens each, heater on, last 24 decode fires
averaged (`kernels.py` reads the blocks the engine prints):

| run | misses a fire | host copy | device, all kernels | the routed matmul |
|---|---|---|---|---|
| batch 1, 1536 seats | 216.4 | 93.76 ms | 39.23 ms | 9.683 ms |
| batch 1, 6144 seats | 73.4 | 39.90 ms | 37.72 ms | 8.283 ms |
| batch 4, 1536 seats | 1675.5 | 583.75 ms | 84.10 ms | 26.646 ms |
| batch 4, 6144 seats | 454.5 | 177.10 ms | 84.08 ms | 26.446 ms |

**Launch counts are identical across pool sizes**, kernel for kernel: 413
dense matvecs, 96 routed matmuls, 228 dense GEMVs, 48 routers, 36 scans at
batch 1; 96 / 228 / 292 / 48 / 36 at batch 4. The pool does not change what
runs or how often.

**Device time is too, at batch 4**: 3.7x the misses and 3.3x the host copy
between the two pools move the fire's device time by 0.02 ms out of 84, and
the routed matmul by 0.8%.

At batch 1 the routed matmul reads 1.4 ms apart (9.683 against 8.283),
which is nearly all of that pair's 1.5 ms difference. Read against batch 4
it is not a per-miss device cost: 1221 more misses there move the same
kernel by 0.2 ms, where a per-miss cost of the batch-1 size would have cost
12 ms. It is more likely clock or thermal drift between two runs.

So: **a miss is host time.** The step is the kernels plus what the host
spends reading, and the pool's size moves only the second term.

## What the batch changes, which is a different question

The batch changes the kernels themselves, because the quantized points pick
a row-batched variant:

| | batch 1 | batch 4 |
|---|---|---|
| dense projections | `affine_qmv_fast` 413 x, 21.2 ms | `affine_qmv_rows_r_4` 292 x, 20.6 ms + `dense_gemm_t_bm_8` 228 x, 22.2 ms |
| routed experts | `affine_qmv_routed` 96 x, 8-9.7 ms | `affine_qmv_routed` 96 x, 26.5 ms |
| attention | `sdpa_paged_decode` 12 x, 1.9 ms | 12 x, 2.9-3.5 ms |
| device a fire | 38 ms | 84 ms |

Four rows cost 2.2x the device time, not 4x — the compute batches well. It
is the miss count that does not (issue 14: 5.3x the misses at 4x the rows).

Note `kernel-profile` puts each kernel in its own command buffer, so these
absolute times are inflated; read them against each other, not against a
step.
