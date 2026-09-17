# Metal expert pool: deepen the seating module

Branch: `metal-expert-cache`. Settled 2026-09-17 by grilling the two review
reports (thermo-nuclear quality review + architecture deepening scan). This
file is the design tree; `issues/` holds one ticket per commit.

Vocabulary: module / interface / implementation / depth / seam / adapter /
leverage / locality (codebase-design). Engine terms: tier, pool, seat, group
(one router's mixture), cut, segment, fire, routing vector, prefill ring.

## What is wrong (verified)

- One fact, "which id space does this routing vector name", lived in five
  modules because the cut rewrote route ids to seat numbers as a side effect
  on the arena. The collapse bug (40e24bb6) and its two fixes (0f1155a6)
  touched Tier, Run, dispatch, kernels-metal and a debug_assert-to-fallback.
- Real pools have 2000-8000 seats; route_sort bins at most 1024 ids, so a
  pool-seated multi-row fire silently falls to the per-row kernel.
- `Plan::under` reads env (`PIE_EXPERT_CACHE_PREFILL`) and diag although
  `Plan::of` is documented environment-free; six env vars are read at ten
  sites, one of them per fire.
- Three seating paths (`segment_rows`, `pass_at`, `ring_at`) each with its
  own read, validation, error string, pin reset and rewrite loop.
- `place()` marks residency before bytes land; a failed pread leaves a
  phantom resident.
- experts.rs 1300 -> 2316 lines, seven concerns, one file.

## Decisions

| # | Decision | Why |
|---|----------|-----|
| Q1 | Routes keep expert ids. Each streamed group owns a `seat_of[expert]` table (E x u32) the kernels read before indexing the bank. | Deletes `route_ids`, `on_ring`, `id_base`, `Run.tier`, `route_space`, both kernel fallbacks. Bins always by expert count. Same shape as CUDA's address table. ADR-0001. |
| Q2 | Seating is `decide` (pure: routes + pool state -> Seating{jobs, evictions, holds}) then `apply` (arena/bytes). Pool state commits only after bytes land; inflight commits at join. | Testable without a device; no phantom residents. |
| Q3 | The prefill ring is not reserved seats past the pool; it is 2E seats **borrowed** from the pool at fire start (coldest first, free seats first), held for the fire, returned at `end_fire`. Floor = 2E when prefill is enabled, E when `PIE_EXPERT_CACHE_PREFILL=0`. | Working-set damage fixed at 2E regardless of layer count; decode gets 2E more seats between prefills. ADR-0002. |
| Q3 | With floor >= E (2E), `pass_at` is provably dead: one segment never routes to more distinct experts than fit. Delete Metal's pass feeding and the `slots / top_k` row cap; keep the explicit `stream_rows_per_cut` tuning. | Dormant modes are the spaghetti the review flagged. |
| Q4 | No new tests. | User decision. Verification is the harness (Q19). |
| Q5 | First commit splits experts.rs by **ownership** (main unchanged / replaced / new), names by content: `trace.rs`, `source.rs` (main unchanged), `plan.rs` (replaced), `tally.rs` (new). Pool+Group+Ring+Tier stay together until after decide/apply (Q16). | Move-only commit reviews easily; later diffs shrink. |
| Q6/Q7/Q15 | Knobs this branch added (`PIE_EXPERT_CACHE`, `_HEADROOM`, `_NOCACHE`, `_PREFILL`, `_REPORT`, `_LOG`, heater's two) are resolved once in api.rs into one value passed to `Plan::beside/under` and `Tier::open`. `Plan.uncached()` replaces the three `nocache()` readers. Env stays; main-era diag fields stay out of scope. Structure of the value (one struct vs two) deferred (Q8). | `Plan::under` and `Tier::open` become deterministic functions of their arguments. |
| Q9 | Seat table attaches as `Bank.seats: Option<Tensor>` bound by the shell per streamed weight row; one persistent E x u32 table per group, updated in place at the cut. The cut only reads routes; `arena.write` of routes goes. | Dispatcher untouched; same timing guarantee as today's arena rewrite (the cut waits on the previous segment's matmuls). |
| Q10 | Tier gets `begin_fire(&lane_rows) -> Mode` / `end_fire()` called from serve.rs enqueue (which already brackets the walk). Tier decides whole-vs-plain itself. Delete `Shell.ring_min`, `Prepared.whole`, `Cuts.whole`, `segment`'s `whole` arg. | Borrow and mode come from the same fact (rows this fire). |
| Q11 | Returned ring seats keep their `(layer, expert)` identity as valid residents at the cold end of the LRU. | Free hits for the last two layers; Pool already keys by (group, expert). |
| Q12 | `pinned: Vec<bool>` becomes a per-seat hold kind: none / segment (released at the next cut) / fire (released at `end_fire`) / inflight (released at join). `victim()` skips any hold. | "Release all" can no longer drop a fire-long borrow. |
| Q14 | Prefetch stays and is the same decide/apply path with an async apply (spawn, join at the next cut). | Two paths would mean two hold rules. |
| Q16 | Split names: `experts/{trace,source,plan,tally}.rs` first; `pool.rs`/`tier.rs` after decide/apply. | |
| Q17 | Pass deletion is Metal-only in this branch. The shared plumbing (model-exec `Descriptor.run_passes`, `pass_spans`, walk `tail_start`/`Encode::tail`; vulkan/wgpu `Window.pass/passes`) is fed by no engine (vulkan/wgpu pass `&[]`) and goes in a follow-up. | Metal expert work is its own project, not an upstream PR. |
| Q18 | Commit order: 01 split -> 02 floor+pass deletion -> 03 knobs -> 04 seat table+kernels -> 05 decide/apply+borrowed ring+holds+fire hooks -> 06 pool/tier split. | 02 must carry the floor with the deletion; 03 before 05 settles `Tier::open`; 04 before 05 keeps the kernel diff separate. |
| Q19 | Gate per commit: `cargo check/clippy` + `cargo test -p engine-metal --lib`. After 02, 04, 05: one qwen38-profile harness decode (1-2k prefill, 1024 decode, last 512 measured) compared with `scratch/qwen38-profile/RESULTS.md` hit rate and step ms. | 04 changes kernels; 05 should raise hit rate (returned layers). |
| Q20 | The seat-space fix was committed first (0f1155a6), text-completion separately (4f1f649d); both are removed/kept by later commits with a bisectable history. | |
| — | **Deviation, issue 04**: the seat table hangs off the weight table, not `kernels_metal::Bank`, because a bf16 expert bank is a `WeightRow::Dense` with no `Bank` and `found` streams those too. ADR-0001 records it. |
| — | **Deviation, issue 05**: `segment_rows` and `ring_at` were not folded into one function. They share the pool, the holds, the seat table and the fire hooks, but their rhythms differ: one seats what a segment names, the other holds a whole layer filled a layer ahead. `ring_has`/`ring_join`/`ring_ready`/`ring_fill` stay for the same reason. |
| Q21 | Records live inside `pie/` and are tracked in git: this spec, `issues/`, `adr/0001`, `adr/0002`, all under `.scratch/metal-expert-pool-deepening/`. `/docs/` is gitignored and `.gitignore` stays untouched. | Workspace has no tracker yet. |

## Facts the decisions rest on

- Ownership map of experts.rs vs origin/main (function-level body hash):
  38 unchanged, 26 same-name-different-body, 63 new, 1 gone (`Slab`).
  Unchanged: `found`, `fan_out`, `weight_of`, `cuts`, `pass_group`,
  `GroupPlan`, `GroupResidency`, all of `Source`/`Bytes`, Plan accessors,
  `Tier::{hint_for,read_rows,join_inflight,motion,hits,prediction,host_time,source,source_kind}`.
  Replaced: `Plan`/`BandPlan`/`Band`, `Plan::{of,beside,slots,source_bytes}`,
  `Tier` + `open/segment/segment_at/segment_rows/pass_at/predict/prefetch/prefetch_group/seat/flush/evict/residency/note_wait/drop`.
  New: `Policy` + env readers, `RegionPlan`, `Plan::under` + helpers, `Group`,
  `Pool`, `Ring`, `CacheReport`/`FireRecord`/`fire_log`, `ring_*`, `place`,
  `touch`, `route_ids`, tally accessors, `mod tests`.
- Passes were born on main in 3d81f42b (2026-09-02). Vulkan and wgpu never
  feed them. CUDA has none.
- Ring fill is asynchronous (next layer into the other half while the
  current computes); holds survive a fire but every fire starts at layer 0,
  so cross-fire reuse never happens.
- Prefetch (`route_prefetch`, default on) is main-era.
- `Bank` comes from the shell's weight table (`WeightRow::Planes`); kernels
  index `bank + e * out_width * in_width`.
