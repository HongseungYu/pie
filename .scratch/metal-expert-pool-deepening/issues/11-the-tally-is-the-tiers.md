# 11 The tally is the tier's, not the shell's

Status: resolved
Type: task

The tier's fourteen counters were `pub` one at a time, forwarded by nine
shell methods, and differenced by hand in `enqueue`: a seven-tuple snapshot
before the walk, subtracted field by field after it to build a `FireRecord`,
then the log line, the fire count and the periodic report, all in the shell.

`Tally` (in `tally.rs`) holds the counters, marks where a fire opened and
answers `close(rows, walk) -> FireRecord`. `Tier::begin_fire` opens it;
`Tier::end_fire(rows, walk)` closes it, writes the CSV line, the
`tier-trace` line and the periodic report. The device time of a fire's last
frame reaches it through `note_tail`, as the cut's wait and device time
already did.

Deleted: `Shell::{expert_hits, expert_gpu, expert_bytes, expert_host_time,
expert_cache_report, expert_prediction, gathered_rows, gathered_motion,
gathered_source}` and the `expert_fires`, `expert_report_every` and
`gpu_tail_ns` fields; `Tier::{hits, bytes_read, prediction, host_time,
gpu_ns}`. The tier's interface is 14 methods, down from 19; the shell keeps
the four the serving test reads.

149 lines out of serve.rs. Gate: clippy --all-targets clean, lib tests 20
passed.
