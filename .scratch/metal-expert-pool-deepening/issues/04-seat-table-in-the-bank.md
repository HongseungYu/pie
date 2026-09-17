# 04 Routes keep expert ids; the bank carries a seat table

Status: claimed
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
