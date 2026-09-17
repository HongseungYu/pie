# 07 Follow-up: remove the shared pass plumbing

Status: needs-triage
Type: task
Blocked by: 02

Separate change, not this branch. After 02 no engine feeds passes:
vulkan and wgpu pass `&[]` for `run_caps`/`run_passes`; CUDA never had
them. Delete model-exec `Descriptor.run_passes`, `compose::pass_spans`,
walk's `tail_start` and `sink.tail`, `Encode::tail`, and
`Window.pass/passes` + the `tail` override in engine-vulkan/wgpu
window.rs. `run_caps` + `chunk_spans` stay (row cap).
