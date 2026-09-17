# 10 Delete what the mapping windows left behind

Status: resolved
Type: task

This branch landed the artifact by uncached `pread` and stopped binding
buffers over the artifact's own mapped pages. The cutting stayed behind.

Deleted: `mapping::{Cut, cut, ceiling, prefault}` and `Mapping::base`;
`Buffer::{mapped, window, is_mapped}`, its `keep` field and the `writable`
guard no buffer could trip; `Context::no_copy`; `Weights::windows` (a
constant 0) and `Shell::weight_windows`; the `window-ceiling`, `prefault`
and `copy-resident` diagnostics, which fed nothing else; the mapping tests
for the cutting, and the window count in the dsv4 serving test.

469 lines out, 6 in. Gate: clippy --all-targets clean, lib tests 20 passed.
