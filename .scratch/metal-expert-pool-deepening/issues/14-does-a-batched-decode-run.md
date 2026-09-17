# 14 Does a batched decode run, and does it decode the same thing?

Status: resolved
Type: research

Asked after the branch's work: batched prefill runs again (issue 04), does
batched decode?

## It is the same path, gated on rows

There is no decode-specific MoE dispatch. `dispatch/linear.rs` tries
`matmul_select_batched` for every routed matmul and takes the per-row point
when it answers `Ok(false)`. The gate reads the row count and nothing else:

    tile_rows(pairs, experts, tuning) > 1
    should_batch = pairs >= experts * moe_batch_min_per_expert   (default 2)
    pairs = rows * top_k

For this model (512 experts, top_k 10) that is 1024 pairs, so the batched
point wants **103 sequences in one fire** at the default tuning.
`[engine.tuning] moe_batch_min_pairs` states a lower bound instead (the
worker copies that table into the engine's `[metal.tuning]`).

The rest of the branch's work already carries a batch: a segment names at
most one layer's experts however many rows it has, and the floor is 2E
(issue 02); the seat table is per group, not per row (issue 04); and the
ring keys off *per-lane* rows, so a decode of N one-row lanes stays on the
pool rather than taking the ring.

## It runs

`scratch/qwen38-profile/batched.py`, 8 concurrent requests, a config with
`max_state_slots = 32`, `total_pages = 512` and
`[engine.tuning] moe_batch_min_pairs = 40`:

- every fire carried more than one row (no single-row fire at all), the
  widest 36 — the scheduler batches concurrent requests into one fire;
- decode fires carried 8 rows, i.e. 80 pairs, over the stated bound, so by
  the gate's arithmetic the batched point ran. (Not observed by name: this
  build has no per-kernel log. The A/B below is what rules it in or out.)

## It decodes what a lone sequence decodes, bar one tie

Eight sequences forced to a lone run's own tokens read identical KV, so
each step's `picked` — the argmax before the substitution — is the same
arithmetic done eight times. Three runs, all identical:

    31 of 32 steps: all eight agree with the lone run
    step 6:         seven pick 14898, one picks 9564, as the lone run did

The prompt is "The capital of France is"; at step 6 the continuation is
"...Paris, and the capital of" and the two candidates are Italy and
Germany. The logits there are near enough to tied that a last-bit
difference decides it, and a row's tile position inside the fire is enough
to produce one: the split is the same in every run, so it is the shape of
the batch, not a race.

Without forcing, that one flip carries the rest of the continuation with
it, which is why a first look showed "seven sequences agree, one does
not". The same flip happens with the batched point turned off
(`moe_batch_min_pairs = 100000`), which is what rules out the batched MoE
kernel as its cause.

## What is not covered

The batched point was exercised at 80-360 pairs, not at the 1024 the
default tuning asks for: this box cannot hold 103 sequences of KV. Prefill
runs the same point at thousands of pairs on every request, so the path
itself is well covered; what is untested is the scheduler at that width.

## Four sequences at once, 1024 tokens each

Four concurrent requests, four different ~1k prompts (llama.cpp's build,
docker, speculative and function-calling docs), 1024 tokens each, 4000
seats, heater on, default tuning — so the prefills take the batched point
(1024 rows apiece) and the decodes take the per-row one (4 rows, 40 pairs,
under the 1024 the default asks for). Last 512 steps, one step being one
fire that advances all four:

| | batch 1 | batch 4 |
|---|---|---|
| step (median) | 116.06 ms | 385.75 ms |
| tok/s | 8.62 | 10.37 (2.59 a sequence) |
| misses a step | 141.7 | 757.9 |
| hit rate | 0.705 | 0.569 |
| read a step | 64.29 ms (54.6%) | 279.46 ms (72.3%) |
| ms a miss (measured) | 0.454 | 0.369 |
| fitted | 0.417/miss + 58.64 | 0.335/miss + 132.23 |
| read over the 512 steps | 186.8 GiB | 999.2 GiB |

Four sequences cost 5.3x the misses, not 4x: they route to different
experts and share a pool sized for one, so the hit rate falls from 0.705 to
0.569 as well. The step goes 3.3x while the tokens a step goes 4x, so
throughput rises 20% and each sequence sees 3.3x slower generation. Nearly
three quarters of a step is the host blocked reading, and the drive is
moving 2 GiB a step at about 5.2 GB/s.

A miss gets cheaper with the batch (0.454 -> 0.369 ms measured): more of
them land in one pread. The step with no misses doubles (59 -> 132 ms),
which is four sequences' attention over their own 1-2k contexts rather than
one.

The prompts are four unrelated documents, which is the hard case for a
shared pool. Four sequences over one document would overlap far more.
