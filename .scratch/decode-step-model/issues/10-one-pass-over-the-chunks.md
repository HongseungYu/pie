# 10 One threaded pass over the store's chunks: the ramp of issue 09 removed

Status: resolved
Type: task

Issue 09 traced the read call's step above 10240 seats to `Store::write_from_file` taking one
threaded pass a chunk, so that an expert whose band regions sit in different chunks reads them
one after the other. Before fixing it, the engine was made to say where its regions sit
(`weight store in N chunk(s); <band> seats a..b in chunk i..j` at open), and the old order was
kept behind `PIE_STORE_SERIAL_CHUNKS=1` so both orders run from one binary.

## Confirmed in the engine (`sm7-*`, `all_in_mem` 40 tokens, `all:1`, one-miss call)

| seats | store layout (engine's own log) | old: 1-miss call mean / median | new: mean / median | step old -> new |
|---|---|---|---|---|
| 8192 | weight store in 1 chunk(s); layer.0.experts_gate_up seats 0..13 in chunk 0..0; layer.0.experts_gate_up.scales seats 13..13 in chunk 0..0; layer.0.experts_gate_up.biases seats 13..14 in chunk 0..0; layer.0.experts_down seats 14..21 in chunk 0..0; layer.0.experts_down.scales seats 21..21 in chunk 0..0; layer.0.experts_down.biases seats 21..21 in chunk 0..0 | 0.705 / 0.695 | 0.706 / 0.694 | 71.8 -> 71.8 |
| 10240 | weight store in 2 chunk(s); layer.0.experts_gate_up seats 0..16 in chunk 0..0; layer.0.experts_gate_up.scales seats 16..17 in chunk 0..0; layer.0.experts_gate_up.biases seats 17..18 in chunk 0..0; layer.0.experts_down seats 18..26 in chunk 0..0; layer.0.experts_down.scales seats 26..26 in chunk 0..0; layer.0.experts_down.biases seats 26..27 in chunk 0..0 | - | 0.748 / 0.736 | - -> 73.9 |
| 11264 | weight store in 2 chunk(s); layer.0.experts_gate_up seats 0..17 in chunk 0..0; layer.0.experts_gate_up.scales seats 17..18 in chunk 0..0; layer.0.experts_gate_up.biases seats 18..20 in chunk 0..0; layer.0.experts_down seats 20..28 in chunk 1..1; layer.0.experts_down.scales seats 28..29 in chunk 1..1; layer.0.experts_down.biases seats 29..29 in chunk 1..1 | 0.987 / 0.975 | 0.738 / 0.721 | 84.9 -> 73.3 |
| 13200 | weight store in 2 chunk(s); layer.0.experts_gate_up seats 0..20 in chunk 0..0; layer.0.experts_gate_up.scales seats 20..22 in chunk 0..0; layer.0.experts_gate_up.biases seats 22..23 in chunk 0..0; layer.0.experts_down seats 23..33 in chunk 1..1; layer.0.experts_down.scales seats 33..34 in chunk 1..1; layer.0.experts_down.biases seats 34..34 in chunk 1..1 | 1.013 / 0.987 | 0.747 / 0.722 | 86.3 -> 73.6 |

At 8192 seats the store is one chunk and the two orders read alike. At 11264 and 13200 it is
two: `gate_up` with its scales and biases in chunk 0, `down` with its in chunk 1 — an expert is
six preads, three a chunk, and the old order ran the two triples serially. One pass brings the
call back to 0.74 ms (0.70 at 8192; the remaining 0.04 is within the run-to-run spread seen
throughout) and the step at the planted grid from 85-86 ms to 73-74.

`FileWriter::pread_many` (`device/alloc.rs`) flattens every chunk's jobs and spreads them over
one set of threads; the tier's cut reads, its prefetch, the ring fill and the n-gram prefetch
all go through it (commit "one threaded pass over every store chunk's reads").

## The model above 8192, after the fix (`out/validate-sm7.txt`)

The natural windows at 11264 and 13200 seats (`steps`, warm 1024 + 1024, last 256) against the
**flat** model of issue 08 — floor 37.05, a 0.368, b 0.1360, no ramp:

| config | tag | seats | steps (clean) | misses/step | L/step | device | copy | ple | measured | predicted | error | error, all steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c2 | sm7-c2-s11264-a | 11264 | 256 (255) | 9.5 | 7.45 | 33.48 | 6.29 | 0.02 | 43.97 | 43.20 | -1.75% | -1.77% |
| c2 | sm7-c2-s13200-a | 13200 | 256 (242) | 4.9 | 4.14 | 33.32 | 3.61 | 0.02 | 41.04 | 40.31 | -1.77% | -2.26% |

Before the fix the same windows read 45.39 and 42.05 ms (issue 09); now 43.97 and 41.04. The
ramp of issue 09 described the serialisation and nothing else: with one pass over the chunks,
a and b hold flat from the floor pool to 13200 seats (~42 GB pinned), within the same
+-2% as everywhere below. Config: `out/sim/configs/mac_m4pro_qwen38flash_np1_v5.json`
(the v4 ramp is kept there under `read_model_ramp_before_fix` for an engine without this
commit).

