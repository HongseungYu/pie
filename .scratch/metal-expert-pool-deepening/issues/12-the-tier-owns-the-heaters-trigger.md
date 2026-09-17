# 12 The tier owns the heater's trigger

Status: resolved
Type: task

The heater's trigger was the pool's knowledge exported as `Tier::blocking`
so that two callers — the cut in `encode.rs` and the fire's last frame in
`serve.rs` — could each ask it and then arm. Two copies of one rule, in
modules that had no other reason to know about GPU clocks.

The tier arms it now, off the same two moments it is already told about:
`note_wait` (a cut's wait returned) and `note_tail` (a fire's last frame
landed), both through one private `heat()` that checks whether the last
segment read from disk. `Tier::blocking` is gone, and so are both
caller-side checks.

Left as it is, on purpose: the `heater::pause()` in the three commit paths
of `device/ctx.rs`. The heater keeps the device's own clock up, so a frame
stepping it aside as real work reaches the queue is the device's business,
not a leak from the pool; and the module-level state is one heater per
process, which is what a GPU clock is. What was wrong was the trigger, and
that has moved.

Gate: clippy --all-targets clean, lib tests 20 passed. The arming moments
are unchanged, so the numbers in RESULTS.md still describe this build.
