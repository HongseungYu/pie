# 12 C by layer and module at batch 2, 4 and 8

Status: resolved
Type: research

Issue 11 measured `floor(batch)` and left the split of `C` alone: the sim config
`mac_m4pro_qwen38flash_batch_v1.json` carried **batch 1's** `layer_types` beside a
per-batch `floor_by_batch`. A simulator that spends the floor as one number was right;
one that walks layers was wrong from batch 2 up, and badly wrong at batch 4. This
measures the split at every batch and says what moved.

## What had to change in the tools

`--diag kernel-profile=2` prints one block a fire: device ms and launch count per
(entrypoint, shape). `kernelsplit.py` read only 1-row fires and classified by entrypoint
prefix with a handful of shape strings as fallback. Two things broke at batch > 1:

- **the fire filter.** `blocks()` kept `fire of 1 row(s)`; a decode fire at batch N has N
  rows. Now `--rows N` (`kernelsplit.py:70`).
- **the shapes.** A shape carries the batch either as a trailing dimension
  (`[2560,16384,N]`) or as the leading one (`[N,4,10240]`); everything else is the model's.
  Rather than a rule per batch — the pitfall METHOD.md's line-number table already warns
  about — `norm_shape(shape, batch)` rewrites a shape into its batch-1 form and the
  existing rules run unchanged. `norm_shape(s, 1)` is the identity, so batch 1 reproduces
  `..._np1_v5.json` to the digit.

`sm_kern_batch.sh N` runs the profile at batch N (N copies of **one** prompt, the c3
recipe: identical routing keeps the zero-miss window as long as batch 1's), and
`layer_table.py` scales each batch's blocks to that batch's measured `C` and writes one
config. Every batch gates on `unclassified 0.000 ms` and on `sum check == C`.

The per-node CSV log the 2026-09-16 runs used (`PIE_KERNEL_LOG`, `PIE_KERNEL_LOG_NODES`
→ `<tag>.kernels.csv`, `<tag>.nodes.tsv`, node → `op` and → `layer.<i>.<part>`) is **gone
from the engine**; `profile_run.py:95,98` still sets the variables and nothing reads them.
That log was the only thing that attributed a kernel to a layer *index*; without it the
split is per layer **type**, as below. `out/kernel-s4000.nodes.tsv` is the last copy, and
it is what identifies the shapes named here.

## The split

Four runs, `all_in_mem` at 8192 seats, 40 tokens, resident pass `decode hit 1.0`, the last
32 decode fires, each scaled to its own measured `C` (issue 11's floor table):

| batch | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| linear_attention attention | 0.278 | 0.322 | 0.431 | 0.716 |
| linear_attention hyper_connection | 0.125 | 0.132 | 0.485 | 0.534 |
| linear_attention ffn | 0.196 | 0.297 | 0.517 | 0.874 |
| **linear_attention per layer** | **0.599** | **0.750** | **1.432** | **2.125** |
| full_attention attention | 0.409 | 0.455 | 0.622 | 1.044 |
| full_attention hyper_connection | 0.125 | 0.132 | 0.485 | 0.534 |
| full_attention ffn | 0.196 | 0.297 | 0.517 | 0.874 |
| **full_attention per layer** | **0.730** | **0.883** | **1.623** | **2.453** |
| output (readout) | 2.530 | 2.636 | 2.845 | 5.267 |
| embed / PLE kernels | 0.006 | 0.006 | 0.000 | 0.000 |
| C (device, measured) | 32.86 | 40.24 | 73.89 | 111.19 |
| kernels sum (profiled) | 36.74 | 42.55 | 72.96 | 109.78 |
| profile scale | 0.8945 | 0.9456 | 1.0128 | 1.0128 |

36 linear_attention (gated delta net) layers and 12 full attention layers: `attn_every = 4`
(`crates/models/src/qwen_4/model.rs:234`), `attn_at = |l| l % 4 == 3` (`model.rs:352`). The
two types share `hyper_connection` and `ffn` exactly because every layer of this model
routes (`qwen_4/forward.rs:509`) and carries the same low-rank residual pair; only the
mixer differs.

**The profile scale is worth reading.** At batch 1 the per-kernel command buffers inflate
the sum 11% over the real frame; by batch 4 the sum is 1.3% *under* it. The inflation is a
fixed per-dispatch cost, and the frames it sits in have grown — so the split is a better
approximation of the truth the larger the batch, not worse.

## Where the batch-4 step in C comes from

Issue 11 recorded `C` as a step function (x1.22, x2.25, x3.38 of batch 1) and named the
kernel switch. Per node, with every shape in its batch-1 form:

| node (batch-1 shape) | what it is | launches | b1 | b2 | b4 | b8 | kernel |
|---|---|---|---|---|---|---|---|
| `[1,4,10240]` | hyper-connection **inject** (`[streams 4, sh 10240]`) | 96 | 0.67 | 0.62 | **15.62** | **15.22** | `dense_gemv_t_ksplit` → `dense_gemm_t_bm_8_bk_64_bn_32` |
| `[1,512,2560]` | MoE **router** (512 experts) | 48 | 0.68 | 0.75 | 2.10 | 1.70 | same switch |
| `[1,1,2560]` | shared-expert **gate** | 48 | 0.22 | 0.21 | 2.00 | 2.00 | same switch |
| `[1,96,2560]` | GDN **in_ba** | 36 | 0.22 | 0.25 | 1.50 | 1.20 | same switch |
| `[2560,1280,...]` + `[640,2560,...]` | routed experts, both banks | 48+48 | 6.46 | 11.15 | 17.39 | 33.44 | `affine_qmv_routed` throughout |
| `[2560,16384]` | GDN **in_qkvz** | 36 | 6.70 | 6.59 | 6.70 | 12.01 | `affine_qmv_fast` → `affine_qmv_rows_r_N` |
| `[6144,2560]` | **out_proj** | 48 | 3.74 | 4.42 | 4.54 | 6.90 | same |
| `[12,32,2,0,0]` | `sdpa_paged_decode` | 12 | 2.38 | 2.44 | 4.03 | 6.93 | unchanged |
| `[2560,248320]` | readout | 1 | 2.73 | 2.69 | 2.71 | 4.90 | unchanged |

Three behaviours, and only one of them is work:

1. **The routed MoE scales with the batch** — 5.79 → 10.54 → 17.61 → 33.87 ms (scaled to
   `C`). That is real: each row picks its own 10 of 512 experts, so N rows read N times the
   expert weights. Nothing to fix.
2. **The dense projections are nearly free to batch** up to 4 — `in_qkvz` 6.70 / 6.59 /
   6.70, readout 2.73 / 2.69 / 2.71, `sdpa` 2.38 / 2.44. They are memory-bound: a second
   row rides the same weight read. Only at batch 8 do they start paying (x1.8).
3. **Four nodes fall off a cliff at batch 4** and then stop caring about the batch:
   1.61 → 1.73 → **21.49 → 20.38** ms. `dense_gemv_t_ksplit` hands them to
   `dense_gemm_t_bfloat16_bm_8_bk_64_bn_32` at batch ≥ 4, and the gemm's fixed cost swamps
   the work. The inject matmul alone is 15.6 ms — **21% of all decode compute at batch 4**,
   against 1.5% at batch 2 — and it is *the same 15 ms at batch 8*, which is the signature
   of an overhead, not of work.

**19.76 ms of the 33.65 ms that `C` gains from batch 2 to batch 4 — 59% — is this switch.**
Batch 4's 50.3 tok/s against batch 2's 43.8 is what is left after paying it. Whether the
`bm_8` tile is the wrong pick for a `[4, 4] x [4, 10240]` shaped matmul, or the ksplit gemv
should simply keep the job to batch 8, is an engine question this effort does not answer —
it is recorded here because the step-latency model inherits the cost either way.

## What the simulator takes from this

- `out/sim/configs/mac_m4pro_qwen38flash_batch_v2.json` — `layer_types_by_batch` with
  1/2/4/8, each summing to its own `C`; `layer_types` at the top level stays batch 1 for
  readers of the old key. `floor_by_batch` and `read_model` are unchanged from v1.
- Spending `floor(batch)` as one number is still exact; the per-layer walk is now
  consistent with it at every measured batch.
- **Do not interpolate between batch 2 and 4 on the layer numbers.** `hyper_connection`
  goes 0.132 → 0.485 on a kernel boundary, not a slope. Between measured batches, use the
  next measured batch down for shape and the measured `C` for scale, or measure that batch.
- The 12 full-attention layers are the only ones whose cost grows with context length; the
  36 linear-attention layers carry fixed-size state. pie's `qwen_4` does **not** implement
  the indexer the checkpoint carries (`indexer_budget`, `indexer_compress_ratio` in the HF
  config; no `indexer` anywhere in `crates/models/src/qwen_4/`, and `forward.rs:444` calls
  plain `ops::attn::decode`), so `sdpa_paged_decode` here is dense attention over the whole
  KV — the measured 0.409 → 1.044 ms carries no block selection.

## Reproduce

```bash
cd ~/hongseung/scratch/qwen38-profile
for N in 2 4 8; do ./sm_kern_batch.sh $N; done          # ~4 min a batch
.venv/bin/python layer_table.py \
  --batch 1:sm-kern-s8192:32.86 --batch 2:kern-b2-s8192:40.24 \
  --batch 4:kern-b4-s8192:73.89 --batch 8:kern-b8-s8192:111.19 \
  --base out/sim/configs/mac_m4pro_qwen38flash_batch_v1.json \
  --name mac_m4pro_qwen38flash_batch_v2 | tee out/layer_table.md
```

The `--frame-ms` of each batch is that batch's `C` from issue 11's floor table, never the
profiled sum. Gates: `resident ... decode hit 1.0` in the run log, `unclassified 0.000 ms`
and `sum check == C` per batch.
