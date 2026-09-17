# 04 Routes keep expert ids; a seat table says where they sit

Status: resolved
Type: task
Blocked by: 02, 03

ADR-0001. Mechanical: the three seating paths stay, each writes the table
instead of rewriting routes.

- kernels-metal: `Bank.seats: Option<Tensor>` (E x u32). `moe_select`
  (per-row), `matmul_select_bias`/`_quant`, and `route_sort` read
  `seats[e]` where they index the bank; `row_expert`/`tile_expert` carry
  the seat. Drop `id_base`. Restore the `debug_assert!` on the sorted
  stacks and the `experts > ROUTE_SORT_MAX_EXPERTS` fallback (bins are the
  router's count again).
- engine-metal: one persistent table per streamed group allocated at
  `Tier::open`, bound into the group's `WeightRow::Planes` bank by the
  shell. The cut writes `seat_of` into it after seating; the routing vector
  is read (`routing_bytes`) but never written back (`arena.write` goes).
- Delete `Tier::route_ids`, `Tier.on_ring`, `Run.tier`, `Run::route_space`,
  the `(experts, id_base)` tuple parameter; `matmul_select_batched` takes
  `experts` again.

Gate: check/clippy/lib tests, then one harness decode vs RESULTS.md
(kernel change).

## Comments

2026-09-17: done, with one change of plan. The table hangs off the weight
table (`WeightTable.seats`, one entry per weight row, bound from the end of
the store) rather than off `kernels_metal::Bank`: a bf16 expert bank is a
`WeightRow::Dense` and has no `Bank`, yet `found` streams those too, so the
bank could not carry the table for every routed point. The dispatcher asks
the weight table (`Run::seats`), a load-time fact, so the `RefCell`
reach-back into the tier is gone either way. ADR-0001 records this.

Kernels: `route_sort` bins the router's own ids again and writes the seat
into `tile_expert`/`row_expert`; `select_gemv` and the routed `qmv_gptoss`
points take `(seats, seated)` and translate before they address the bank,
the scales, the zero points and the expert bias, all of which stream
together as bands of one group and so share a seat. The batched
`quant_qmm_t` kernel is untouched: it reads the seat out of `tile_expert`.

Engine: `Tier::say_seats` writes its group's table at every cut (E u32s,
2 KiB for this model) and the routing vector is no longer rewritten.
Deleted `Tier::route_ids`, `Tier.on_ring`, `Run.tier`, `Run::route_space`,
the `(experts, id_base)` tuple, the `id_base` kernel argument and the two
silent fallbacks; the sorted-stack check is a `debug_assert` again.

Gate: clippy --all-targets clean, lib tests 20 passed, and a smoke run
(32 teacher-forced tokens, 4000 seats) decoded the recorded tokens exactly.
Untested path: the bf16 `select_gemv` point, which no imported model uses.

2026-09-17, harness gate (clean mode, 4000 seats, heater on, 1k prompt,
1024 teacher-forced tokens): every forced-token assertion passed, so the
indirection decodes exactly what rewriting the routes did.

Timing, corrected 2026-09-17 (see issue 13): `T(1024)` 133.35 s against
issue 02's 133.30 s, i.e. no measurable change end to end. The per-step
figures first reported here (119.78 ms mean, 96.70 ms last half) are the
harness's difference of separate requests and carry run-to-run prefill
variance, not a speed-up; the claim that the batched point stopped falling
back is true of the code, but this run does not measure it.
