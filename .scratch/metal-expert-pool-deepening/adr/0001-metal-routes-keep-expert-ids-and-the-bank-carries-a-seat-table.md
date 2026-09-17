# ADR-0001: Metal routes keep expert ids; the bank carries a seat table

Date: 2026-09-17
Status: accepted

## Context

engine-metal streams routed experts through a shared LRU pool. Until now the
cut rewrote each routing vector in the arena from expert ids to the seat
numbers the experts were landed in, so the MoE kernels could index the pool
directly. That made "which id space does this vector name" a fact produced
as a side effect in the tier and needed in the Run, the dispatcher and
kernels-metal. The collapse bug (40e24bb6) and its fixes (0f1155a6) touched
all of them, added a RefCell reach-back from the dispatcher into the tier, an
`id_base` kernel argument, and two silent per-row fallbacks. Real pools have
2000-8000 seats while `route_sort` bins at most 1024 ids, so pool-seated
multi-row fires were already falling to the per-row kernel. engine-cuda
never rewrites routes: it indirects through a device-side address table.

## Decision

Routing vectors keep the router's expert ids on every fire. Each streamed
group owns one persistent table `seat_of[expert] -> seat` (E x u32) on the
device, attached to its bank as `Bank.seats: Option<Tensor>` and rewritten in
place by the cut after seating. Kernels read `seats[e]` where they index the
bank; the sort still bins by expert. A resident (non-streamed) bank has no
table and indexes by id as before.

## Consequences

- The id space of a routing vector is always the router's expert count; the
  bound the batched kernel needs is derived from the value it bins.
- `route_ids`, `on_ring`, `id_base`, `Run.tier`, `route_space` and both
  runtime fallbacks are deleted; the sorted-stack check is a debug
  assertion again.
- The cut reads routes and never writes them back; the seat table is the
  only thing it writes to the device.
- One extra u32 load per row in the per-row kernel and per row in
  `route_sort`; tables total experts x groups x 4 bytes (48 KiB for 48 x 256).
- A future architecture review should not propose returning to seat-id
  rewriting to save that load.
