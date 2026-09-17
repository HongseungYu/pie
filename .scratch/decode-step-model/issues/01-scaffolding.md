# 01 Scaffolding: the reader, the records, the clock probe

Status: resolved
Type: task

`scratch/qwen38-profile/stepmodel.py` reads a tag's `fires.csv` + `cuts.csv` (the format
`tally.rs` writes on this branch) and gives: `window` (the measured window's mean, median,
p10, p90, misses and layers a step, and the split into device / copy / turnaround /
cut bookkeeping / rest), `dump` (per-step TSV with per-layer misses), `fitcuts` (a, b from
cut rows), `fitsteps` (a, b from the steps given a floor), `predict` and `table` (the user's
form against a window). A step is the gap between two decode fires' `t_ms`; its device time is
this fire's `gpu_cut_ms` plus the next record's `gpu_tail_ms`; per-layer misses come from the
cut rows carrying this fire's number.

Checked against the previous effort's logs:

    resident20-s11204 (0 misses)   step 45.90 mean / 45.66 median; device 30.55, turn 4.45, rest 10.73
    cut-b1-s4000 (last 256)        step 119.04; 134.7 misses in 41.4 layers; device 41.90, copy 62.84
    fitcuts cut-b1-s4000           copy_ms = 0.349 + 0.359 m   (r2 0.984)
    fitcuts cold-s11204            copy_ms = 0.866 + 0.405 m   (r2 0.073)

The user's form with the floor measured and the dense-read a, b (45.66 + 0.346 + 0.1364 MiB):
-8.9% at 4000 seats, -16.7% at 11204. Those are the two errors this effort removes: device
time 41.9 against 30.6 (clock), and a call at 0.87 against 0.35 (read path).

`gpu_tail_ms` reads 0.000 in every record: the final frame's device span is not captured on
this branch, so "device" here is the 47-48 cut frames and the final frame's ~3 ms sits in
`rest`. Noted; consistent across runs, so the model is unaffected, but the clock check reads
the cut frames only.

`macmon` (Homebrew, no sudo) installed 2026-09-17 for a direct GPU-frequency reading.
