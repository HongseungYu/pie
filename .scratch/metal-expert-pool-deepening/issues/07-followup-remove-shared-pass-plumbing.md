# 07 Follow-up: remove the shared pass plumbing

Status: wontfix
Type: task
Blocked by: 02

Separate change, not this branch. After 02 no engine feeds passes:
vulkan and wgpu pass `&[]` for `run_caps`/`run_passes`; CUDA never had
them. Delete model-exec `Descriptor.run_passes`, `compose::pass_spans`,
walk's `tail_start` and `sink.tail`, `Encode::tail`, and
`Window.pass/passes` + the `tail` override in engine-vulkan/wgpu
window.rs. `run_caps` + `chunk_spans` stay (row cap).

## Comments

2026-09-17: deferred by the user. The plumbing is dead but it lives outside
this project's engine, and touching model-exec or the other engines' windows
buys nothing here. Left as it is; the fact that no engine feeds it is
recorded above for whoever picks it up.
