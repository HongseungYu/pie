# 11 The model at batch 2, 4 and 8

Status: resolved
Type: research

METHOD.md §5.3 said batch > 1 was not measurable with the harness as it stood, and named what
was missing. This closes that: `profile_run.py --seqs N [--prompts a,b,...]` fires N sequences
at once against one server in every mode, `stepmodel.py`/`validate.py` take `--batch`, and
`pie-qwen38-b{2,4,8}.toml` seat them (`max_state_slots = 2 x batch`, KV for N x 2k tokens).
Each sequence replays **its own** greedy trace (`out/forced_prompt-8aN.json`), so routing is
deterministic and the sequences differ the way a real batch does.

Prompt sets, as METHOD.md §5.3 requires them to be recorded:
- **c1 / c1b / c2 / mid**: eight different documents (`prompt-8a0..7.txt`), one a sequence — the
  workload. Its miss counts are what the model must predict.
- **c3 / c4**: the same document N times. Identical routing keeps the zero-miss window as long
  as batch 1's, and neither the floor nor a planted miss depends on **which** experts are read.

## The answer

**The batch moves the floor and nothing else.** The read call is the same at every batch:

| batch | a (ms a call) | b (ms a MiB) | 1-miss call | 10-miss call |
|---|---|---|---|---|
| 1 | 0.402 | 0.1272 | 0.737 | 3.756 |
| 2 | 0.372 | 0.1296 | 0.714 | 3.789 |
| 4 | 0.361 | 0.1297 | 0.703 | 3.780 |
| 8 | 0.363 | 0.1297 | 0.701 | 3.780 |

so the model is the batch-1 one with a floor per batch:

    step_ms = floor(batch) + sum over layers with m_l >= 1 of ( 0.368 + 0.1360 * m_l * 2.637 )

| batch | floor | C (device) | F (host) | C vs batch 1 | tok/s at 0 misses | a sequence |
|---|---|---|---|---|---|---|
| 1 | 37.05 | 32.86 | 4.19 | x1.00 | 27.0 | 27.0 |
| 2 | 45.68 | 40.24 | 5.44 | x1.22 | 43.8 | 21.9 |
| 4 | 79.50 | 73.89 | 5.61 | x2.25 | 50.3 | 12.6 |
| 8 | 116.65 | 111.19 | 5.46 | x3.38 | 68.6 | 8.6 |

**F does not move** (5.4-5.6 ms from batch 2 up; batch 1's 4.19 is lower because its cut
turnaround is shorter). Only `C` grows, and not linearly: x1.22 to batch 2, then x1.84 to
batch 4 where the dense projections switch to the row-batched kernels, then x1.50 to batch 8.
Throughput rises 2.5x from batch 1 to 8 while a sequence's own rate falls 3.1x.

## Validation

Full table: `out/batch_table.md` (30 windows). Every configuration at every batch, against the
model above, over clean steps (n-gram rows cached, drive not stalled):

- batch 2: +1.88 .. -3.43%  (nine windows)
- batch 4: +2.09 .. -2.58%  (nine windows, the 1024-seat pool excepted)
- batch 8: +3.73 .. -4.15%  (nine windows, the 1024-seat pool excepted)

Worst outside the degenerate pools: **4.15%** (`b8-c4-every_2_5`); all but three windows are
inside 2.2%.

## Where the model stops: the pool must hold one step

At batch N a single fire routes to up to `N x layers x top_k` distinct experts. When the pool
is smaller than that it misses **everything** and the drive saturates:

| run | seats | misses/step | hit rate | copy a step | effective b |
|---|---|---|---|---|---|
| batch 4, c1 | 1024 | 1710 | 0.000 | 610-657 ms | 0.146 (vs 0.1297) |
| batch 8, c1 | 1024 | 3213 | 0.000 | 1099-1181 ms | 0.143 |
| batch 8, mid | 3072 | 3100 | 0.035 | 1056 ms | 0.141 |

The per-MiB cost rises ~10% because the drive is at its ceiling (7.2 GB/s), and the step-level
correlation collapses (r 0.14-0.44). These are reported and kept out of the fit; `c1b`, whose
pool scales as `1024 x batch`, is the max-miss point inside the model's domain (-1.83, +1.15,
-1.61% at batch 2, 4, 8).

## Two things the harness now guards

- `sm_run.sh`'s "a pie serve is alive" check matched its own caller whenever the command line
  mentioned a `pie-*.toml` and a `*.server.log`; it matches the process name now.
- Editing a sweep script while bash is running it shifts the byte offsets under the shell: one
  run's wrapper exited 1 after its runs had all completed. Copy the script (`sm_sweep_batch_run.sh`)
  before editing the original mid-sweep.

Simulator config: `out/sim/configs/mac_m4pro_qwen38flash_batch_v1.json` (floors by batch, the
shared read model, and the domain limit).
