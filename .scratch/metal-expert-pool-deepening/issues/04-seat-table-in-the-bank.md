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
