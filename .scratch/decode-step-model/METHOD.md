# Measuring a decode step, and modelling it

A manual. Follow it to produce, for one (machine, model, batch), a step-latency model of the form

    step_ms = floor + sum over layers with m_l >= 1 of ( a + b * m_l * B )

`floor` is **measured** at zero misses, never fitted (§3). `m_l` is layer `l`'s misses (experts read off
disk), `B` one expert's MiB, `a` what one layer's read call costs before any bytes, `b` ms a MiB. Only
`a` and `b` are fitted. Layers that miss nothing contribute nothing.

- **Repeating it on this box** (M4 Pro 48 GB / Qwen3.8-Flash-Next-4bit / batch 1): run §0's checklist, then
  Appendix A's **Reproduce** block (`sm_sweep3.sh` ~1 h, then `validate.py`), then read §4 to grade the
  output. §1's conditioning is already done and its answers are in Appendix A. §2 and §6 are what to check
  when a number looks wrong.
- **Repeating it elsewhere** (other machine, model or batch): **§5 first — it is a patch list, not an
  appendix.** Apply §5.1 / §5.2 / §5.3 to the harness and re-read the load line for `groups` and `MiB`
  before running anything (`stepmodel.py:83-84` and `validate.py:114` drop every cut row with
  `group >= LAYERS` silently). Then §0, then §1 in full — every number in Appendix A is one box's answer,
  not a constant, except that §1.1 may fall out on its own gate at batch > 1 (read §1.1's opening note).
  Then §2–§4.

Paths used throughout: `$H` = `/Users/yecl/hongseung/scratch/qwen38-profile` (harness),
`$P` = `/Users/yecl/hongseung/pie` (engine, branch `metal-expert-cache`),
`$R` = `$P/.scratch/decode-step-model` (records: `spec.md`, `RESULTS.md`, `issues/01..10`).

---

## 0. Pre-flight (checklist)

Run this before the first measured run and after any reboot. Everything is gate-or-stop.

| # | check | command | pass |
|---|---|---|---|
| 1 | engine has the instrumentation | the loop below | `7` — the §8d minimum |
| 2 | binaries built | `cargo build --release -p pie --bin pie --features metal` in `$P`; `cargo build --manifest-path $P/tests/inferlets/Cargo.toml -p text-completion --target wasm32-wasip2 --release` | both exit 0 |
| 2b | inferlet version matches | `grep ^version $P/tests/inferlets/text-completion/Pie.toml; grep -n 'INFERLET =' $H/profile_run.py` | the two are equal. **Today they are not** (`Pie.toml` `0.3.0` vs `profile_run.py:38` `text-completion@0.3.1`), which survives here only as a stale `~/.pie/programs/text-completion/0.3.1.wasm`. Edit one to match before the first run |
| 3 | no other server | `pgrep -f "pie.*serve"` | no output (`sm_run.sh:18` exits 3 otherwise) |
| 4 | machine quiet | close browsers, other agent sessions, GUI apps; `sudo mdutil -a -i off; sudo tmutil disable` if you have sudo | no sudo → sample their activity per run and discard windows in which it moved |
| 5 | thermal nominal | `/Users/yecl/hongseung/scratch/thermal` | prints `nominal` (build from `scratch/thermal.m` on a new box, or stub it) |
| 6 | forced sequence exists — **once per model** | `cd $H && rm -f out/forced_tokens.json && ./sm_run.sh preflight greedy <a pool that boots> 1024 0 1 PIE_METAL_HEATER=on` | `out/preflight.log` reads `greedy: 1024 tokens in Ns -> .../out/forced_tokens.json` **and** its `tail:` is this model's own text; `grep -m1 '^launch:' out/preflight.log` names the binary under `$P` |
| 7 | the load lines appear | the two greps below, on `out/preflight.server.log` | all six `engine-metal:` lines present, none in its wrong form; `seated=` equals your batch |

```bash
for h in e949089b 0a08e1a9 e7c12238 fba613c0 31a96628 75452854 eabe2fd6; do
  git -C $P merge-base --is-ancestor $h HEAD 2>/dev/null && echo $h
done | wc -l     # pass: 7
```

If the branch has been squashed or rebased into `main` these hashes are gone. Fall back to the capability
check — it is what those commits buy: `PIE_EXPERT_CACHE_CUT_LOG` writes a `.cuts.csv` with a `gpu_ms`
column, `gpu_tail_ms` is non-zero in the fire log, `PIE_EXPERT_CACHE_FORCE_MISS=all:1` plants exactly
`#layers` misses a step, and the heater knobs of §8b are read.

```bash
grep -nE 'expert cache|landed|weight store|heater|n-gram rows|F_NOCACHE' $H/out/preflight.server.log
grep -n 'admission:' $H/out/preflight.server.log        # read `seated=`; it must equal your batch
```

Each line and what it proves (real example: `$H/out/sm7-c2-s13200-a.server.log:6-11`, and `:13` for the
admission WARN):

| line (prefix `engine-metal: ` unless marked) | proves | wrong if |
|---|---|---|
| `expert cache: 13200 seats x 2.64 MiB = 33.99 GiB shared by 48 groups over 24576 (layer, expert) pairs, 1024 of which a prefill fire borrows for its ring from 103 rows a lane [PIE_EXPERT_CACHE=13200]` | the policy actually applied (brackets), seats, **B**, **layer count** (`groups`), **ring size** | reads `expert cache off (...)` — nothing streams, no cut rows, no model |
| `landed 1739 plane(s), 4.94 GiB, in 0.82 s (6.47 GB/s, uncached preads)` | this drive's real bandwidth | GB/s far below spec → thermal/SSD problem, stop |
| `weight store in 2 chunk(s); layer.0.experts_gate_up seats 0..20 in chunk 0..0; ... layer.0.experts_down seats 23..33 in chunk 1..1` | the store's chunk layout, and which band sits in which chunk | nothing — a band split across chunks is **expected** once the store exceeds `maxBufferLength` (`$P/crates/engine-metal/src/device/ctx.rs:113`; the store is cut in `weight_store.rs:23`, so chunks ≈ ceil(store bytes / maxBufferLength), and that shrinks with RAM). Record the layout and the pool it flipped at. It matters only if the **1-miss call steps up** at that same pool — then check `8a98f324` is in and `PIE_STORE_SERIAL_CHUNKS` is unset, and see §7 (why: issues 09, 10) |
| `heater on: 1024 threads x 8192 FMAs x 2 in flight, armed at every gap, logged to ...` | the heater shape you asked for is the one running | `heater not started:` / line absent → the knob never arrived (§6, unsplit variable) |
| `cpu heater on: one thread spinning for the load's life` | no per-call CPU wake-up | `cpu heater off` → `a` is gap-dependent, un-modellable (why: issue 03) |
| ``n-gram rows: 32768 seats of `ple.table`, read by uncached pread, prefetched as the fire opens`` | PLE off the page cache | says `the artifact's mapping` (why: issue 08) |
| **WARN** `runtime::bootstrap: admission: more lanes than the state pool seats; capping ... requested=512 seated=1 seat_cost=2 state_slots=2` | how many sequences this server will actually run **at once**: `seated = max_state_slots / seat_cost` | `seated` < your batch — every extra request queues instead of batching, silently, with `rows=1` on every fire and `forced_ok: true`. Raise `[engine] max_state_slots` to `seat_cost × batch` (this model: `2 × batch`). At batch 1 the line is expected and harmless |

Also fatal: `F_NOCACHE on the artifact was refused; expert reads go through the page cache`.

---

## 1. Condition the machine

Four conditions, each as run → pick → gate → if it fails. Order 1.1 → 1.4.

- **§1.2–§1.4 must run under §1.1's winner.** Export it once: `export HEAT="<winner>"`. Three scripts carry
  a hard-coded heater string instead of reading it — patch them first (§5.1).
- **§1.1 needs a pool before §1.4 has picked one.** Any pool that boots and whose short `all_in_mem`
  resident pass reports `0.0 misses` will do; halve it until it does. It need not be your final ceiling.

### 1.1 Hold the GPU clock (why: issue 02)

> **Test the gate before you run the grid.** The heater exists because a mostly-idle GPU downclocks.
> Measure the two anchors with the heater off first: `device(D0)` (`all_in_mem`, big pool, zero misses)
> and `device(Dmax)` (`steps`, floor pool, max misses). If `|device(Dmax) − device(D0)| ≤ 6%` already,
> §1.1 is done — record both numbers and go to §1.2. At batch 1 on this box it is not (30.6 → 43.1 ms,
> saturating past ~150 misses a step); at batch 4 it is (84.10 against 84.08 across 3.7× the misses —
> four sequences keep the device busy enough that the clock never sags; why: pool-deepening issue 15).

| | |
|---|---|
| **run** | First patch the pool and window baked into `sm_sweep1.sh:7,12-28` and `sm_sweep2.sh:6-11`: D0 = `all_in_mem <largest pool that boots> <provisional W>` (provisional `W` = §3's `0.6 × (N − 2·ring) / u`), Dmax = `steps <ring> 512`, the ring read off §0's load line. Then `cd $H && ./sm_sweep1.sh && ./sm_sweep2.sh && .venv/bin/python sweep1_table.py` (~40 min). Grid covered: arm `reads` vs `always` × kernel `mem` (several MiB) / `alu` (several iteration depths) / `spin` × inflight 1 vs 2 |
| **gate (D0)** | before reading the table: `stepmodel.py window <any d0 tag> --last <W>` reads `0.0 misses in 0.00 layers` for **every** candidate. It does not ⇒ shorten the D0 window or raise its pool and rerun. `sweep1_table.py` prints no D0-misses column, so a contaminated anchor is invisible in the table you are about to read |
| **pick** | minimise `device(Dmax) − device(D0)` subject to (i) `device(D0)` ≤ heater-off D0 + 0.5 ms, (ii) `Dmax copy` no more than ~2% above the best copy in the table, (iii) macmon median MHz at the device max at both ends |
| **gate** | `\|device(Dmax) − device(D0)\| ≤ 6% of device(D0)` (`device(D0)` is `C` up to the misses you are removing; the real `C` is measured later, §3/c3) **and** `device(D0)` ≤ the `hoff` row + 0.5 ms — the **pick** tolerance is the gate. Then 3 reps of the winner at D0: that spread is your rep tolerance for the rest of the effort |
| **fail** | wide kernel every gap → `D0 device` explodes; too shallow → the host commits ~30k buffers/s and **Dmax copy** inflates; too deep → it blocks frames and `Dmax device` rises; `spin` → the GPU does not see the host's flag mid-kernel, times are µs or the full bound. Widen the **depth** grid before widening the shape grid |
| **trap** | MHz alone proves nothing: *1578 MHz in every sample at Dmax and 1566 median at D0 — both at or beside the device max — while device time swung 30.1 → 43.4*. Always compare device time |

Read the table as `| id | env | D0 device | D0 step | D0 rest | Dmax device | Dmax step | Dmax misses/L | Dmax copy | Dmax rest | MHz D0/Dmax |`.
The winner's env is `HEAT` for every later run.

### 1.2 Keep the host read path warm (why: issue 03)

| | |
|---|---|
| **run** | Build and sweep the probe outside the engine — same call shape as the engine (per expert one large + one small `F_NOCACHE` pread at random offsets of the real artifact, over ≤16 spawned threads). The artifact is the `checkpoint=` field of the load log's `serving the sku the config named` line: `export ZT=$(grep -m1 checkpoint $H/out/preflight.server.log \| sed -E 's/.*checkpoint[^"]*"([^"]*)".*/\1/')`. Then `cc -O2 -o $H/tools/ssd_gap $H/tools/ssd_gap.c -lpthread && $H/tools/ssd_gap "$ZT" cpu` (modes `spawn\|pool\|keeper\|cpu`). It sweeps gap G ∈ {0, 0.5, 1, 2, 5, 10, 30, 100} ms × m ∈ {1, 4}. `$ZT` serves §7 step 4 and `tools/probe_beside_server.sh:6` too |
| **pick** | the cheapest knob holding ms/call flat from G=0 to G ≥ the largest gap a real decode leaves between calls (read that off `<tag>.cuts.csv`; *2–10 ms here*). Order: nothing → one spinning CPU thread → a tiny keeper read → a thread pool |
| **gate** | the probe flat within ~10% across G; then **in-engine**: `stepmodel.py fitcuts` gives one `a` and one `b` across a dense-read pool, a mid pool and the large pool |
| **fail** | reject anything that oversubscribes cores (16 spin-waiting readers → 10–13 ms a call). A keeper read that does not hold it flat past 2 ms proves the **drive** is not what sleeps. If no CPU-side knob works, report the cost as a regime; do not model `a` as a function of read density |
| **knob** | `PIE_METAL_CPU_HEATER=1` (`$P/crates/engine-metal/src/device/spinner.rs:22`) — default off; `sm_run.sh:12` forces it on for every run |

### 1.3 Take the page cache out of the step (why: issue 08, symptom in issue 05)

| | |
|---|---|
| **ask** | which host cost inside the step reads memory the OS may evict? Here: the n-gram (PLE) table |
| **run** | `cd $H && ./sm_ple_ab.sh` — three `all_in_mem` runs at the same pool: `PLE_SOURCE=mmap PREFETCH=0`, `pread PREFETCH=0`, `pread PREFETCH=1`. Compare the **prime** pass (first touch, `--request 0`) with the **resident** pass (repeat) |
| **gate** | `\|first-touch − repeat\| ≤ 0.1 ms a step` **and** `ple_prefetch_misses = 0` in both passes |
| **fail** | move the rows to uncached pread at offsets already known from the load (never copy the table into a second file) and prefetch from host-computed ids as the fire opens. A wrong host guess must cost a read, never a wrong row — count the misses |
| **knobs** | `PIE_PLE_SOURCE=pread`, `PIE_PLE_PREFETCH=1` (both defaults); expert reads `PIE_EXPERT_CACHE_NOCACHE` (default on) |

### 1.4 Find the pool ceiling (why: issues 05, 08, 09, 10)

The largest pool you measure is the largest at which **nothing outside the model moves** — not the largest
that boots.

```bash
cd $H && ./sm_probe_seats.sh && grep 'probe s' out/sm_runs.log | tail -5 && cat out/seats_max.txt
cd $H && ./sm_sweep5.sh planted && ./sm_sweep5.sh natural \
      && .venv/bin/python sweep5_table.py     # the full sweep above the quick pick
```

**Read the `probe sN:` lines before trusting `out/seats_max.txt`.** If no candidate printed `misses/step`
under 0.5, the file holds the script's default (`pick=8192` at `sm_probe_seats.sh:8`) and nothing was
measured — that is what happened on this box, because `sm_probe_seats.sh:10` probes with a 128-token
window and the pool holds 40. Shorten the probe window (`:10` and `--last` at `:12`) until the largest
candidate's resident pass is clean, then re-run. The candidate list at `:9` and the default at `:8` are
this box's; set both from your RAM first.

| pick the largest N at which all hold | check |
|---|---|
| 1-miss call within run-to-run spread (~±5%) of the small-pool value | `sweep5_table.py`, `1-miss call` |
| floor within ±2% of the small-pool floor | `sweep5_table.py`, `floor (0-miss steps)` |
| n-gram join < 0.5 ms | `sweep5_table.py`, `ple join` |
| resident pass of the floor window misses exactly 0.0 | `stepmodel.py window <tag> --last <W>` |
| host recovers: the `after:` free figure back above ~5% of RAM, `swapouts/s` at zero | `out/sm_runs.log` memory line; `<tag>.vmstat.log` for the trend |

A `least free during: 0.0–0.1 G` dip while the pool fills is **normal** — every accepted 8192-seat run here
shows it — and the compressor runs at several thousand pages/s throughout (7052/s at 8192 seats, 13952/s at
13200, swapouts 0 at both). Judge on swapouts, and on whether `ple_ms` and the read call moved.

**Do not cap at the first failing N.** Decide by §7 whether the cause is the host or the engine
(why: issues 08, 10 — both apparent caps here were removable).

Memory accounting: `pinned GB = N × B_MiB × 1.048576 / 1024 + FIXED_GB`, `FIXED_GB` = planes + arena + KV +
tables (`validate.py:42`, *6.5 GB here*, ~1.5 GB of other resident memory beside). Recompute it from the
load log on any other model or config toml.

---

## 2. Take the measurement

| rule | do this | gate | on fail |
|---|---|---|---|
| **teacher forcing** | one `--mode greedy` run writes `out/forced_tokens.json`; every later run replays it, so routing is identical across seat counts and modes | `"forced_ok": true` for every measured request in `out/<tag>.requests.json` — **but** that only proves the engine echoed the ids it was handed (`profile_run.py:179`), and it is `true` for another model's token file. The real gate is that `out/forced_tokens.json` was written by a `--mode greedy` run **of the model under test**: check its mtime against the model change. Second gate: `decode_hit_rate` in the same file matches the pool you asked for | the run is void — the miss pattern is a different experiment. Re-run. When the model changes, `mv out/forced_tokens.json out/forced_tokens.<oldmodel>.json` (or pass `--forced out/forced_<model>.json` everywhere): a stale file is undetectable downstream. At batch > 1 forcing still gives identical routing; a near-tied `picked` may differ for one row because of its tile position in the fire — that is the batch's shape, not a race (why: pool-deepening issue 14) |
| **warm-up** | `steps` runs replay the whole 1024-token sequence as their warm-up whatever `decode` is (c2 decodes 1024; c1 and the mid points decode 512 — `sm_run.sh <tag> steps <seats> <decode> 1024 1`); `all_in_mem` runs use their own prime pass | the window's first steps not systematically dearer than its last | lengthen the warm-up — a cold prime pass runs ~6× a steady call (why: issues 02, 05) |
| **window** | last **256** steps of a `steps` request; last **32–40** of an `all_in_mem` resident pass. `--skip 1` drops the post-prefill step | report mean **with** median, p10, p90 | never use differenced harness figures (`(T(1024)−T(512))/512`) as a step measurement |
| **reps** | two fresh-**server** runs per `steps` configuration, three for the floor | reps agree within §1.1's spread (*±1.5–2%*) | a rep outside it is a drive-state or heater-state run — diagnose it, do not average it in |
| **drop filter A** — n-gram faults | a step counts only if `ple_ms < PLE_MAX` | pick `PLE_MAX` between the cached and the faulting value on this box (*cached 0.02–0.1, faulting 5–7 → 0.5*); pass it to `validate.py --ple-max` | no clean separation ⇒ §1.3 is not done |
| **drop filter B** — drive stalls | `validate.py` does this for you: `--stall-factor 1.5` against the cut-level curve it fits in the same run, falling back to `validate.py:126`'s `0.387 + 0.357 m` when no planted grid sits under the prefix (that fallback is **this SSD's**, §5.1) | report clean-step **and** all-step error, plus the dropped count | never drop silently: *one run lost 23/256 → −0.25% clean vs −14.17% all* |
| **rerun rule** | clean steps ≥ ~80% of the window | the `steps (clean)` column of `validate.py` | rerun the configuration; keep the original in `out/`, out of the fit, and named in the record |

One run, all sidecar logs (macmon 250 ms, `vm_stat` 2 s, `iostat -d -w 2`, heater CSV, thermal before/after,
one summary + one memory line appended to `out/sm_runs.log`):

```bash
cd $H && ./sm_run.sh <tag> <mode> <seats> <decode> <warmup> <reps> [KEY=VAL ...]
# modes: greedy | steps | all_in_mem | kernel | clean
# EXTRA_ARGS="--config $H/pie-qwen38-big.toml" for a different pool budget
```

Read one run before trusting a batch:

```bash
cd $H && .venv/bin/python stepmodel.py window sm-c3-n40-a --last 40
# sm-c3-n40-a: seats 8192, request 1 of 2, 37 steps measured  [PIE_METAL_CPU_HEATER=1 PIE_METAL_HEATER=on ...]
#   step ms   mean   38.40  median   38.33  p10   38.05  p90   38.74   (26.04 tok/s)
#   per step     0.0 misses in  0.00 layers, hit rate 1.000, 0.000 MiB a miss
#   split     device  33.29  gpu_cut  30.24  gpu_tail   3.05  copy   0.00  turn   3.55  cut_host   0.20
#             ple   0.03  enc   1.00  tail   3.38  rest   1.36
```

Gates on that output: the bracketed env shows the heater knobs as **separate words**; `device ≈ gpu_cut +
gpu_tail`; `misses` is what you planted or expected. `NO CUT LOG: L unknown` ⇒ the cut log was off and the
run cannot be fitted. `tok/s` is **fires** a second (`stepmodel.py:164`) — at batch N divide by N (§8c).

---

## 3. The five configurations

Run c3 first and gate `W` on it; a wrong `W` spoils c4's 13 runs and c3's 3 reps together, and the damage
only shows at §4.

```bash
cd $H && export HEAT="<winner from §1.1>" SEATS_MAX=<N from §1.4> WIN=<W>
./sm_sweep3.sh c3
.venv/bin/python stepmodel.py window sm-c3-n$WIN-a --last $WIN    # must read 0.0 misses in 0.00 layers
./sm_sweep3.sh c4 c1 c2 mid                                       # ~1 h
```

| # | what | how | what it pins down | gate |
|---|---|---|---|---|
| c1 | **max misses** | `steps` at the floor pool = smallest that boots (= ring size); plus a smaller one with `PIE_EXPERT_CACHE_PREFILL=0` | the `a × L` extreme | `L/step ≈ #layers`; hit rate ≤ ~0.35; the **second** request's prefill must not fail (§6) |
| c2 | **min misses** | `steps` at `SEATS_MAX`, ×2 reps | the extreme where the floor dominates | hit rate ≥ ~0.85, `ple < PLE_MAX`, floor unchanged |
| c3 | **zero misses** | `all_in_mem` at `SEATS_MAX`, `W` tokens, ×3 reps | **the floor** (C and F), measured | resident misses/step exactly `0.0`; 3 reps within ±2% |
| c4 | **planted misses** | `all_in_mem` at a primed pool + `PIE_EXPERT_CACHE_FORCE_MISS=<layers>:<k>`. Cover: L-sweep at k=1 (`all:1, every:4:1, every:12:1, every:48:1, 0-23:1`); k-sweep at L=12 (`every:4:{1,2,4,5,10}`); equal-m/different-L pairs (`all:1` vs `every:4:4` vs `every:8:8`; `all:5` vs `every:2:10`) | the call term `a` at known (L, m); makes "same m, different L" a direct experiment | the cut log reads exactly the plant (`all:1` → 48.0 misses in 48.00 layers); hit rate unchanged; device time within 1 ms of the floor's |
| mid | **interpolation** | `steps` at 3–4 seat counts spanning the miss range | fit targets + interpolation check | each within §4's gate |

**Window length `W` for c3/c4.** The prime pass touches `u` new (layer, expert) pairs a step and the
resident pass's own prefill borrows `2 × ring` of the coldest seats. Get `u` from the prime pass:
`stepmodel.py window <any all_in_mem tag> --request 0 --last <W>` → `per step 131.1 misses in 40.57 layers`
(u ≈ 131 here); `ring` is the load line's `1024 of which a prefill fire borrows`. Start at
`W ≈ 0.6 × (N − 2·ring) / u` and verify. If only the **earliest** steps miss, the ring borrow took their
seats → shorten `W` (*64 → 48 → 40 at 8192 seats here*). Do not buy a longer window with `--forced-offset`:
a repeated token in a different context routes elsewhere and the union does not shrink (why: issue 05).

**`WIN=128` is reserved.** `validate.py:84` drops every `sm-c3-n*` tag containing `n128` from the floor and
`validate.py:129` hands it to the natural-miss fit — it is the long-window record run written by
`sm_sweep3.sh:21`, not a floor rep. If your `W` lands on 128, use 120 or 130, or change both lines together.

`--mode all_in_mem` fires the same forced request twice per rep: the prime pass seats every expert the
second will route to, the resident pass decodes with nothing to read. `floor` = mean step over the
zero-miss, ple-clean steps; `C` = `gpu_cut + gpu_tail`; `F` = `floor − C`, itemised as turnaround /
seating / n-gram rows / encode / after-walk. If `gpu_tail_ms` reads 0.000 in every record, this build does
not count the final frame and `C` is wrong.

---

## 4. Fit and validate

```bash
cd $H && .venv/bin/python validate.py --fit-on minimax \
  --seats <N from §1.4> --win <W from §3> --last 256 --ple-max <PLE_MAX from §2> \
  --prefix <tag family> --name <box>_<model>_np<batch>_v<n> | tee out/validate-<name>.txt
```

- `--fit-on` defaults to `all` (step-level least squares) — **pass `minimax` explicitly** for the reported model.
- `--seats` defaults to `8192` (`validate.py:67`) and is a **filter, not a label**: c2/c3/c4 runs at any
  other pool are dropped (c1 and mid are exempt). Pass §1.4's N. `StatisticsError: fmean requires at least
  one data point` from `validate.py:95` means the filter ate your c3 reps, not that the runs are bad.
  `--seats any` pools every pool into one floor — use it only with `--fixed-floor/-a/-b`, never for a fit.
- `--name` defaults to `mac_m4pro_qwen38flash_np1_v2` and the output config is written unconditionally:
  an unnamed run overwrites another box's — or this box's current — sim config.
- `--prefix` defaults to `sm`. Name your prefix per regime and never reuse a `validate-*.txt` across an
  engine change: this box's three validations are `sm` → `out/validate-final.txt` (**superseded**,
  pre-issue-08, floor 37.69), `sm4` → `out/validate-sm4.txt` (the reported floor 37.05), `sm7` →
  `out/validate-sm7.txt` (the 11264/13200 re-validation). `tee` to your own file, not over a record.

| step | what to look at | gate |
|---|---|---|
| floor block | `## Floor (C + F)`: floor, C (cut frames + final frame), F itemised, per-rep values | 3 reps within ±2%; set-aside count small |
| drive's own curve | `## Read call, cut level: copy_ms = a_cut + b_cut * m (r2 ...)` with the per-m table | r2 not collapsed (a run at r2 ~0.25 against neighbours' ~0.97 is a drive-slow run, §6) |
| fitted `a`, `b` | the `**Model**` line, `(minimax over ['c1','c2','c3','mid'])` | minimax `(a, b)` lands on the cut-level curve within a few % — if not, the step pays something beyond its reads: find that term, do not absorb it into `b` |
| the pick is interior | the `**Model**` line's a and b against the search box at `validate.py:231,233` | `a` strictly inside (0.20, 0.80) **and** `b` strictly inside (0.100, 0.180). On a boundary the answer is the grid's edge, not a fit: widen `validate.py:230-233` and refit before believing any error figure or opening §7 |
| step-level OLS cross-check | `step level: step - floor = A*L + B*MiB ...; with intercept ...` | a large intercept means a missing constant term |
| the table | one row per tag: `steps (clean)`, misses/step, L/step, device, copy, ple, measured, predicted, error, error-all-steps | **the parenthesised figure ≤ ~4%**: `validate.py`'s closing line reads `worst ... X% (Y% outside the planted grid)` and **Y** is the gate — X includes the c4 rows, which the next row expects one-sided by up to ~5%. Every configuration represented. This box: 1.75% (`sm`), 3.96% (`sm4`, a drive-slow c1 rep), 1.77% (`sm7`). A Y much above ~4% is a missing term, not noise → §7 |
| planted rows | c4 rows | expect the **measured** window up to ~5% **above** the model on the multi-miss plants. `error` is `100 × (pred − meas) / meas` (`validate.py:180`), so those rows read **negative** (−4.08 / −4.54 / −5.17% here) while the k=1 plants straddle zero. Keep them out of the objective either way (why: issues 04, 07). Structural checks only: equal total m at different L must differ by ≈ `a × ΔL`; device time unchanged within 1 ms |
| output config | `wrote out/sim/configs/<name>.json` | contains floor, C, F's parts, `read_model`, variants, worst error |

Fit on `minimax`, not the `all` default: the call cost is not exactly linear in m, so a straight line is a
compromise and least squares hands that compromise to whichever pool contributed the most steps; minimax
bounds the worst pool (why: issue 07). Keep the planted grid out of the objective — `--include-planted`
only to see the cost of letting it in.

Re-validate a later regime against fixed parameters, no new fit:

```bash
cd $H && .venv/bin/python validate.py --prefix sm7 --seats any \
  --fixed-floor 37.05 --fixed-a 0.368 --fixed-b 0.1360 --ple-max 2.0 --win 40 --last 256
```

**Report apart, do not model:** if your simulator fires a long prefill and then decodes, measure the first
seconds after the burst separately and report them as their own regime (numbers in Appendix A).

Optional — `C` split by layer type and module (needs its own run; never mix with a step-time run, since
`kernel-profile` puts each kernel in its own command buffer):

```bash
cd $H && EXTRA_ARGS="--diag kernel-profile=2" ./sm_run.sh sm-kern-s8192 all_in_mem 8192 48 0 1 $HEAT
.venv/bin/python kernelsplit.py sm-kern-s8192 --rows 1 --frame-ms <measured zero-miss device ms> --last 32
# gate: "sum check X ms against the frames' X" and "unclassified 0.000 ms"
```

`--frame-ms` is the **measured** `C` from §3, never the profiled sum: the per-kernel command buffers
put that sum 11% over the real frame at batch 1 (and 1% under it at batch 4), so the split is a share
of a measured total, not a total of its own. `--rows` is the batch: a decode fire at batch N has N
rows and `blocks()` keeps only fires of exactly that width.

**At batch > 1, measure it per batch — do not reuse batch 1's.** `sm_kern_batch.sh N` runs it (N
copies of one prompt, the c3 recipe) and `layer_table.py` puts the batches side by side and writes
one config:

```bash
cd $H && for N in 2 4 8; do ./sm_kern_batch.sh $N; done
.venv/bin/python layer_table.py --batch 1:sm-kern-s8192:<C1> --batch 2:kern-b2-s8192:<C2> ... \
  --base out/sim/configs/<the batch config>.json --name <box>_<model>_batch_v<n>
```

A shape carries the batch as a trailing dimension (`[2560,16384,N]`) or a leading one
(`[N,4,10240]`); `norm_shape()` rewrites it into the batch-1 form so `classify()` keeps one set of
rules. Add a rule only if `unclassified` is non-zero. On this box the split moved a long way with
the batch — `hyper_connection` 0.125 → 0.132 → 0.485 → 0.534 ms a layer, on a kernel boundary at
batch 4, not a slope (why: issue 12). Expect to have to measure every batch you will simulate.

---

## 5. Porting

### 5.1 Another machine

| do | detail |
|---|---|
| `$H/pie-qwen38.toml:16,22`, `pie-qwen38-big.toml:16,22` | `device_weight_budget` 40GiB / 44GiB and `gpu_mem_utilization` 0.90 / 0.96 are **48-GB numbers**. Set the budget above the largest pool you intend to measure plus `FIXED_GB`, and `gpu_mem_utilization` so the two fit under the unified memory. Nothing in §0–§4 boots until these are right, and `sm_sweep5.sh:6` pins the big toml unconditionally |
| redo **all** of §1 | none of Appendix A transfers: clock behaviour, wake-up cost, drive curve and pool ceiling are all this box's |
| `$H/sm_sweep1.sh:7,12-28`, `sm_sweep2.sh:6-11` | D0 pool `11204` (28.9 GiB of seats) and Dmax pool `1024` are hard-coded on every run line — set them **before running §1.1**, or `smoke()` at `sm_sweep1.sh:6-10` fails its first run and exits 1 |
| `$H/sm_probe_seats.sh:7`, `sm_ple_ab.sh:3`, `sm_sweep5.sh:5` | each hard-codes a heater string instead of reading `$HEAT`; `sm_probe_seats.sh:7` is this box's §1.1 winner verbatim. Replace with `: "${HEAT:?set HEAT to the §1.1 winner}"` and substitute `$HEAT` at every call site |
| `$H/sm_probe_seats.sh:8,9,10,12`, `sm_sweep5.sh:11`, `sm_sweep3.sh:12,30` | the fallback `pick`, the candidate pools, the probe window (128 — too long for this box's own pool), the floor pool and the mid points: all RAM-dependent |
| `$H/sm_run.sh:10`, `$H/profile_run.py:31,35-37` | `PIE_PROFILE_ROOT` is hard-coded to this box's clone at `sm_run.sh:10` and **defaults to a second clone** (`/Users/yecl/hongseung/pie-profile`) at `profile_run.py:31`, from which `PIE`, `WASM` and `MANIFEST` are derived. Set both to your `$P`. A run against the wrong clone writes a different cut-log header, with inert flight-log knobs and no error |
| `$H/profile_run.py:38` | `INFERLET = "text-completion@0.3.1"` against `Pie.toml`'s `0.3.0` — §0 gate 2b |
| `$H/sm_run.sh:19,33` | `/Users/yecl/hongseung/scratch/thermal` — build from `scratch/thermal.m` or stub it |
| `$H/sm_run.sh:20,27` | `f*16/1048576` assumes a **16 KiB page** (Intel Mac = 4 KiB ⇒ every memory figure 4× wrong); `macmon pipe -i 250` — no macmon, no `MHz` column |
| `$H/validate.py:67` | `--seats` default `8192` — a filter, not a label (§4) |
| `$H/validate.py:126` | `a_cut, b_cut = 0.387, 0.357` fallback drive curve — **this SSD's**; used for the stall flag |
| `$H/validate.py:186-187,200-206,230-233` | `fitknee` defaults, the knee grid `30.0+0.25k` GB and the minimax grid `A=0.20+0.003i`, `B=0.100+0.0005j` — a much faster/slower drive, or a memory edge elsewhere, puts the optimum outside the box **silently** |
| `$H/validate.py:42`, `sweep5_table.py:9` | `FIXED_GB = 6.5` — recompute from the load log |
| `$H/tools/probe_beside_server.sh:6-7` | artifact `.zt` path and `PIE` binary path |

### 5.2 Another model

| do | detail |
|---|---|
| **decide the form applies, before anything else** | (a) **no misses?** No MoE, or the pool holds every expert ⇒ `m_l ≡ 0`: only c3 is meaningful — measure the floor, report it, stop. (b) **one call a layer?** Compare `max m_l` over c1/c4 against `SEAT_THREADS = 16` (`$P/crates/engine-metal/src/experts/tier.rs:98`; overridable with `--diag seat-threads=<n>`). If a layer needs more than one round of readers the call term is `a·ceil(m_l/16)`, not `a` — plant `PIE_EXPERT_CACHE_FORCE_MISS=all:<k>` at k = 15, 16, 17, 20 and look for a step in `copy_ms` at the thread count before fitting any line. (c) **are `a` and `b` separable?** `stepmodel.py dump` each configuration's `L` and `misses·B`. If `L` is pinned at `#layers` everywhere, `a·L` is a constant: fold it into the floor and report `floor' + b·m·B`, two terms not three. c4's equal-m/different-L pairs (`all:1` vs `every:4:4` vs `every:8:8`) settle it — if they do **not** differ by ≈ `a·ΔL`, the per-call term is not there |
| re-bootstrap the forced sequence | retokenise the prompt, `mv out/forced_tokens.json out/forced_tokens.<oldmodel>.json`, then `--mode greedy` again (`$H/profile_run.py:134-135`). A stale file replays another tokenizer's ids with `forced_ok: true` (§2) |
| `$H/stepmodel.py:32` | `LAYERS = 48` → read off the load line: `... shared by 48 groups`. Cut rows with `group >= LAYERS` are dropped silently (`stepmodel.py:83-84`, `validate.py:114`) |
| `$H/stepmodel.py:33` | `MIB = 2.637` (**B**) → read off the load line: `13200 seats x 2.64 MiB`, or `bytes_read / misses` |
| `$H/validate.py:230-233` | the minimax grid is **model-dependent**: `b` is ms a **MiB** (`validate.py:119` divides by `sm.MIB`). Halve the expert and the per-MiB figure roughly doubles — a ~1 MiB expert sits near 0.36 ms/MiB and the search clamps at 0.180 with no message. Rescale by `2.637 / B_new` before the first fit |
| `$H/kernelsplit.py:19` | `GDN, FULL, ALL = 36, 12, 48` — used both as launch-count fingerprints **and** as divisors; wrong values mis-assign kernels *and* mis-scale per-layer ms |
| `$H/kernelsplit.py:22-42` | `norm_shape()` — which dimension of a shape is the batch; a wrong guess silently mis-normalises |
| `$H/kernelsplit.py:44-63` | `classify()` entrypoint prefixes and shape strings — a stale rule shows up as non-zero `unclassified` |
| `$H/tools/ssd_gap.c:20-21`, `ssd_align.c:25-27`, `ssd_metal.m:21-22` | `GU 1843200`, `DN 921600` = one expert's band bytes (**three copies**, change together); `ROW 96` = one n-gram row |
| `$H/validate.py:63` | `--ple-max 0.5` — set high for a model with no PLE table |
| `$H/profile_run.py:38-39` | `INFERLET`, `CONFIG` toml; `$H/*.toml:8` `port` must match `profile_run.py:41 PORT` |
| `$H/sm_sweep3.sh:24` | the 13 `layers:k` plant specs assume 48 layers (`every:12`, `0-23`) |
| config toml | `device_weight_budget`, `gpu_mem_utilization`, `total_pages`, `max_model_len` set what fits and what the n-gram table becomes (§5.1's first row for the sizing rule) |

### 5.3 Another batch size

> **Batch > 1 is measured; issue 11 has the results and this section is the recipe it used.**
> `profile_run.py --seqs N [--prompts a,b,...]`, `stepmodel.py --batch N`, `validate.py --batch N`
> and `pie-qwen38-b{2,4,8}.toml` exist now; `sm_sweep_batch.sh <N> [groups]` runs the whole
> procedure. What follows was written before that and still names everything that needs care.
>
> **Superseded:** `profile_run.py` / `sm_run.sh` fire one
> sequence and take no concurrency flag. `batched.py --seqs N` is the only thing that launches several,
> and it is a probe, not a measurement rig: it sets `PIE_EXPERT_CACHE_LOG` (`batched.py:29`) but **not**
> `PIE_EXPERT_CACHE_CUT_LOG`, so it writes no `cuts.csv` and therefore no `m_l`. Two minimum edits before
> you can measure at all:
> 1. `batched.py:29` — add `env["PIE_EXPERT_CACHE_CUT_LOG"] = f"{OUT}/{tag}.cuts.csv"`; or give
>    `profile_run.py` a `--seqs N` that launches N processes concurrently and keep `sm_run.sh`'s sidecars.
> 2. `batched.py:11` — `ROOT` defaults to `/Users/yecl/hongseung/pie-profile`. Export `PIE_PROFILE_ROOT=$P`
>    or you read a cut log with a different header off a different build.
>
> And check §0's admission line: `max_state_slots ≥ seat_cost × batch` (this model: `2 × batch`), or the
> server serves your N requests one at a time and you measure batch 1 believing it is batch N.

| do | detail |
|---|---|
| the form is unchanged — check first | `m_l` is still that layer's misses, but re-read §5.2's "decide the form applies": at a batch where one layer routes to more than `SEAT_THREADS = 16` distinct experts the call term becomes a staircase `a·ceil(m_l/16)` |
| `m_l` grows **super-linearly** | 4 sequences over 4 unrelated 1k documents cost **5.3× the misses at 4× the rows**: they route to different experts and share a pool sized for one, so the hit rate falls 0.705 → 0.569 and reads go from 54.6% to 72.3% of a step (why: pool-deepening issue 14). Budget seats ≈ N × the batch-1 pool before concluding anything about `a` and `b`; N sequences over one document overlap far more, so record the prompt set |
| **which MoE kernel the decode takes** | the routed matmul batches only when `pairs = rows × top_k ≥ experts × moe_batch_min_per_expert` (default 2) — `$P/crates/kernels-metal/src/linear/moe.rs:35-43` (`should_batch`), `$P/crates/kernels-metal/src/tuning.rs:55,81`. For this model that is 1024 pairs = **103 sequences**; at batch 4–16 the decode stays on the per-row point. State which side you are on in the record and never fit across the boundary (why: pool-deepening issue 14) |
| forcing the batched point | `[engine.tuning] moe_batch_min_pairs = <pairs>` states a lower bound instead (default 0 = off, `kernels-metal/src/tuning.rs:57,82`; `pie-qwen38-batch.toml` uses 40, `-batch-perrow.toml` uses 100000 to pin the per-row point for an A/B). Measured at 80–360 pairs only; the scheduler at 1024 pairs is untested |
| re-measure the floor | **`C` is a step function in batch, not a slope**: batch 4 costs 2.2× batch 1's device time (38 → 84 ms) because the dense projections switch kernel (`affine_qmv_fast` → `affine_qmv_rows_r_4` + `dense_gemm_t_bm_8`) while the routed matmul stays per-row and simply gets 3× dearer (why: pool-deepening issue 15). Never reuse a batch-1 floor |
| re-measure the **split** too | the same switch moves the split much further than it moves `C`: `hyper_connection` 0.125 → 0.132 → 0.485 → 0.534 ms a layer at batch 1/2/4/8, because four nodes leave `dense_gemv_t_ksplit` for `dense_gemm_t_bm_8` at batch ≥ 4 and 59% of the batch 2 → 4 step in `C` is that one switch (why: issue 12). Never reuse a batch-1 `layer_types`, and never interpolate one between measured batches |
| re-measure `a` and `b` | a call at batch N reads more experts at once; the per-call fixed part is amortised differently, so the (a, b) split moves even if total bytes/ms does not |
| `$H/stepmodel.py:57,71` | `batch=1` is a Python default with **no CLI flag** — add `--batch` and thread it through `requests()` / `steps()`, or a decode fire of N rows is read as a prefill (`stepmodel.py:61`) and every step vanishes. Verify with `stepmodel.py window <tag>`: it must report the expected number of decode steps, not "no decode fires" |
| `$H/validate.py:39,113,159` | the three `.steps(...)` call sites drop `batch` — all three must pass it, or the fit stays at batch 1 whatever `stepmodel.py` says |
| `$H/validate.py:265-267` | `"batch": 1` and the `"description"` string are hard-coded — the emitted sim config otherwise claims batch 1 on an M4 Pro running Qwen3.8-Flash-Next |
| `$H/stepmodel.py:164` | `1e3 / mean_step_ms` is **fires** a second; at batch N divide by N for tokens a second a sequence |
| re-pick `W` | `u` (new pairs a step) rises with N, so the zero-miss window shortens ~N× |
| re-pick the pool ceiling | two caps, not one: **seats** (RAM, §1.4, now ~N× the working set) and **`max_state_slots`** (what actually bounds concurrency, §0's admission line; `total_pages` / `max_model_len` only bound context length). Recompute `FIXED_GB` from the load log after changing either |
| configs | `$H/pie-qwen38-batch.toml`, `-batch4.toml` (`max_state_slots = 8`, `total_pages = 512`), `-batch-perrow.toml` are starting points |

**Does each configuration still work?** (answered by measurement in issue 11)

| config | at batch > 1 | why |
|---|---|---|
| c1 max misses | **unchanged** | the floor pool is still 2E: a segment names at most one layer's experts however many rows it carries, and the ring keys off per-lane rows, so N one-row lanes stay on the pool (pool-deepening issue 14) |
| c2 min misses | unchanged in form, **new `SEATS_MAX`** | §1.4 again; the hit-rate gate (≥ ~0.85) needs a much larger pool |
| c3 zero misses | **the one at risk** | `u` rises with N, so `W ≈ 0.6 × (N_seats − 2·ring) / u` shrinks ~N×. If `W` falls below ~16 steps the floor is not measurable this way: raise the pool, or use one document across all N sequences so the union stops growing |
| c1 max misses (measured) | **the floor pool is outside the model at batch > 1**: 1024 seats hold less than one fire's working set, so the pool misses everything and the drive saturates (`b` +10%). Use `c1b`, a pool of `1024 x batch` (issue 11) | |
| c4 planted misses | **unchanged, plant included** | the plant is per (fire, layer), not per row — `all:1` is still 48.0 misses in 48.00 layers at any batch (`$P/crates/engine-metal/src/experts/plan.rs:79-84`). This is the cleanest way to get `a` and `b` at batch N |
| mid | unchanged | |

---

## 6. Traps

| symptom | cause | what to do |
|---|---|---|
| GPU at ~900 MHz, device time far above D0, no `heater on` line | a shell variable of knobs reached `profile_run.py --env` **unsplit** — the engine read the whole word and turned the heater off. Nothing errors and `forced_ok` is true | launch only through `sm_run.sh` / `sm_sweep*.sh`, where `$HEAT` word-splits. Keep `sm_run.sh:13-15`'s guard (exit 4 on an env word containing a space) in any new launcher. Post-hoc: the bracketed env in `stepmodel.py window`, and `stepmodel.py macmon <tag>` (why: issue 07) |
| `validate.py` dies in `statistics.fmean`, "requires at least one data point" | `--seats` (default 8192) matched none of the c3 runs, so the floor window is empty | pass `--seats <SEATS_MAX>`; `--seats any` reads every pool (and pools them — fixed parameters only) |
| every call ~20% dearer at **every** m; cut-fit r2 collapses (*0.24 vs 0.97*); host traces flat | the drive is in a slow state for the whole run | compare its cut curve to neighbours' at m = 1, 2, 4, 8; rerun as `-b`, keep the original in `out/` and out of the fit, name it in the record |
| a burst of steps with `copy_ms > 1.5×` the curve; all-step error diverges from clean-step error | drive stall, seconds inside a run | set aside, count, show both error columns; rerun if clean < ~80% |
| the effect tracks the session's age, not the swept variable | ordering confound | **reverse-order control**: measure the largest size first, then the smallest; accept only if both ends reproduce |
| `swapouts/s` non-zero, `enc` several ms a step, `ple_ms ~6`, the reads themselves slower | host memory pressure at the pool ceiling | cap at §1.4's ceiling or remove the dependency the pressure exposes. **Never fold an OS cost into F** — it depends on what else the machine holds and it drifts. (A `least free during` dip and a busy compressor alone are **not** this — §1.4) |
| the second request dies: *"one segment of this fire routes to more than N distinct experts ... every seat is held"* | a floor-sized pool: the last decode segment of a fire has no next cut to release its holds. Single-request runs never hit it | release `Hold::Segment` at `begin_fire` after joining the prefetch and returning the ring (why: issue 06). Always exercise a floor pool with warm-up request **then** measured request |
| a **step** (not a slope) in the call's *fixed* part while floor, device time and the n-gram join are flat; every host probe comes back flat | engine self-serialisation at large allocations (bands stop sharing a Metal chunk) | §7; the engine's own layout log names the structure that flipped |
| a first pass much dearer than a repeat; a host term that moves with pool size but not with misses | page-cache dependence somewhere in the step | §1.3 |
| `.flights.csv` / `.kernels.csv` never appear | `PIE_FLIGHT_LOG` / `PIE_KERNEL_LOG` are inert on this build | never build a step model on a flight log that is not there, and make anything parsing a cut log assert its header (§5.1's `PIE_PROFILE_ROOT` row) |

---

## 7. When a cost does not fit the model

A term that will not fit is the host's, the drive's, or the engine's own structure, and you cannot tell
which from inside the engine. Falsify every host hypothesis first (each must come back flat); accept the
engine only on a probe that reproduces **both** regimes quantitatively.

1. **Localise** it in the step's split (`device / copy / turn / cut_host / ple / enc / tail`) and show every
   other term flat across the sweep. A term that moves alone is one mechanism.
2. **Shape it**: sweep the suspected variable over ≥5 points; decide **step vs slope**. A step at a threshold
   implicates a structure that changes there; a smooth slope implicates a resource being consumed.
3. **Reverse the sweep order** and confirm both ends reproduce — otherwise it is session drift.
4. **Rebuild the call outside the engine**: same bytes, thread count, flags, file (`$ZT`), offsets
   (`tools/ssd_gap.c`, `ssd_align.c`, `ssd_metal.m`). The probe must land within a few % of the engine's
   number in the **unaffected** regime before it may say anything about the affected one.
5. **Falsify the host, one hypothesis per probe run**: memory pressure (a wired balloon), destination size
   and kind (anonymous vs Metal `StorageModeShared`), virgin vs touched pages, offset/destination alignment.
   Also measure **beside a real idle server** holding the same allocation
   (`tools/probe_beside_server.sh <seats>`) — the engine's memory state without its activity. Record every
   flat number; they are the evidence.
6. **Imitate the engine's structure in the probe** (e.g. the same bytes as one threaded call vs two). Accept
   only on a quantitative match of both regimes.
7. **Make the engine state the structure** (log the layout that flips at the threshold) and put the old
   behaviour behind a knob (`PIE_STORE_SERIAL_CHUNKS=1`), so both orders run from one binary.
8. **A/B at the same sizes.** Accept when the anomalous points return to baseline and the flat model meets
   §4's gate. If the fix is not taken, model the anomaly as a ramp continuous with the flat model at its foot
   and flat past its top (`validate.py --fit-on fitknee`), and say plainly it describes **this engine today**,
   not the host.

---

## 8. Tool and knob reference

### 8a. Tools (all under `$H`)

| tool | answers | invocation |
|---|---|---|
| `sm_run.sh` | one measured run + every sidecar log | `./sm_run.sh <tag> <mode> <seats> <decode> <warmup> <reps> [KEY=VAL ...]`; `EXTRA_ARGS="--config ...\|--diag ..."` |
| `profile_run.py` | the run itself (boots `pie serve`, teacher-forces, writes `out/<tag>.*`) | `--mode greedy\|steps\|all_in_mem\|kernel\|clean --seats --decode --warmup --reps --tag --env K=V --config --diag --forced --forced-offset --prompt --with-logs` |
| `stepmodel.py` | one run's window, split, fits | `window\|dump\|fitcuts\|fitsteps\|floor\|predict\|table\|heater\|macmon <tag>...` + `--last --skip --request` |
| `validate.py` | floor, a, b, every configuration's error, the sim config | `--win --last --prefix --seats --fixed-floor/-a/-b --fit-on --ple-max --stall-factor --exclude --name --include-planted` |
| `batched.py` | do N sequences share one fire, and which MoE point does it take? | `PIE_PROFILE_ROOT=$P ./.venv/bin/python batched.py --seqs N --config pie-qwen38-batch4.toml` — **probe only**: writes no cut log as shipped (§5.3) |
| `sm_sweep1.sh` / `sm_sweep2.sh` + `sweep1_table.py` | which heater setting makes device time miss-independent | `./sm_sweep1.sh && .venv/bin/python sweep1_table.py` |
| `sm_probe_seats.sh` | the quick pool-ceiling gate | `./sm_probe_seats.sh` → `out/seats_max.txt` (read the `probe sN:` lines too, §1.4) |
| `sm_sweep3.sh` | the five configurations at one conditioned setting | `HEAT="..." SEATS_MAX=8192 WIN=40 ./sm_sweep3.sh c3 c4 c1 c2 mid` |
| `sm_sweep5.sh` + `sweep5_table.py` | pools above the quick pick: planted call, floor, natural window, host state, in one row a pool | `./sm_sweep5.sh all` (forces `pie-qwen38-big.toml`) |
| `sm_ple_ab.sh` | mapping vs pread vs pread+prefetch for the n-gram rows | `./sm_ple_ab.sh` |
| `sweep7_table.py` | store-chunk layout + 1-miss call, old order vs new | `.venv/bin/python sweep7_table.py` |
| `kernelsplit.py` | C by layer type and module, one batch | `.venv/bin/python kernelsplit.py <tag> --rows <batch> --frame-ms <C> --last 32` |
| `sm_kern_batch.sh` + `layer_table.py` | the same split at every batch, side by side, into one sim config | `./sm_kern_batch.sh <N>`; `layer_table.py --batch N:<tag>:<C> ... --name <config>` |
| `tools/ssd_gap.c` | is the rare-read penalty the drive, the threads, or the CPU? | `cc -O2 -o tools/ssd_gap tools/ssd_gap.c -lpthread`; `./tools/ssd_gap "$ZT" cpu` |
| `tools/ssd_align.c` | alignment / host pressure / destination size | `cc -O2 -o tools/ssd_align tools/ssd_align.c -lpthread`; `./tools/ssd_align "$ZT" <balloon_gb> <dest_gb> 16 1` |
| `tools/ssd_metal.m` | same, destination a Metal `StorageModeShared` buffer, optional virgin pages | `clang -O2 -o tools/ssd_metal tools/ssd_metal.m -framework Metal -framework Foundation`; `./tools/ssd_metal "$ZT" 40 16 1 8` |
| `tools/probe_beside_server.sh` | the read cost beside a real idle server at N seats | `./tools/probe_beside_server.sh 13200` (edit artifact/binary paths first) |

### 8b. Engine knobs (all read **once**, at the load's edge — setting one after `pie serve` starts does nothing)

| knob | values | default | effect |
|---|---|---|---|
| `PIE_EXPERT_CACHE` | `off\|full` / `auto\|on` / `<n>` | the config's `device_weight_budget`, else `auto` = free RAM at load, less a 4 GiB headroom (`PIE_EXPERT_CACHE_HEADROOM`), less the dense planes — tens of GiB on a big box | `<n>` = exactly n seats (the measurement arm); `off/full` = everything resident, **no misses, no cut log** |
| `PIE_EXPERT_CACHE_LOG` | `<path>` | unset | the per-fire CSV — **the step clock**. Required |
| `PIE_EXPERT_CACHE_CUT_LOG` | `<path>` | unset | the per-cut CSV — **the per-layer miss vector**. Required |
| `PIE_EXPERT_CACHE_FORCE_MISS` | `<layers>:<k>`, layers = `all` / `every:<n>` / list with `lo-hi` | unset | the first `k` distinct experts each named layer routes to **in a decode segment** are re-read from disk whether seated or not. Changes no result, only where the bytes come from |
| `PIE_EXPERT_CACHE_PREFILL` | `<rows>` / `0` | `max(32, ceil(2*experts/top_k))` | the prefill ring; `0` drops it, so a prompt evicts the decode working set |
| `PIE_EXPERT_CACHE_NOCACHE` | switch | **on** | `F_NOCACHE` + `F_RDAHEAD=0`: reads come off SSD, nothing enters the page cache |
| `PIE_EXPERT_CACHE_COLD` | switch | off | every prefill fire starts from an empty pool — a cold measurement without restarting the server |
| `PIE_METAL_HEATER` | `off` / `on\|auto` / `<MiB>` | **off** | `on` ⇒ ALU kernel + `arm=always`; `<MiB>` ⇒ `mem` scale kernel + `arm=reads` |
| `PIE_METAL_HEATER_ARM` | `reads` / `always` | follows the above | `always` = at every cut wait and fire tail |
| `PIE_METAL_HEATER_KERNEL` | `mem` / `alu` / `spin` | follows the above | `mem` bandwidth-bound, `alu` compute-bound and narrow, `spin` polls a host flag |
| `PIE_METAL_HEATER_ALU_ITERS`, `_ALU_THREADS`, `_INFLIGHT`, `_LOG` | `<n>`, `<n>`, `<n>`, `<path>` | 8192, 1024, 2, unset | kernel depth/width, queue depth, per-window CSV (`mean_ms` is a direct clock readout) |
| `PIE_METAL_CPU_HEATER` | switch | **off** | one thread spinning for the load's life, so a miss call pays no CPU wake-up. `sm_run.sh:12` forces `=1` |
| `PIE_PLE_SOURCE` / `PIE_PLE_PREFETCH` | `pread`/`mmap`, switch | `pread`, **on** | n-gram rows by uncached pread at known offsets, prefetched as the fire opens |
| `PIE_STORE_SERIAL_CHUNKS` | switch | **off** | `on` restores the pre-fix pass-a-chunk-at-a-time read order, for A/B only |

`--diag` words worth knowing: `kernel-profile[=1|2]` (per-kernel split; **never** in a step-time run — each
kernel gets its own command buffer), `seat-threads=<n>` (reader threads per call, default 16), `tier-trace`
(one stderr line a fire: the cheap sanity check), `route-prefetch=off` / `keepalive=off` (A/B the host-side
helpers, both default on). An unknown word is a hard error that lists the valid words.

### 8c. CSV columns

`out/<tag>.fires.csv` — one row per fire:
`seq,rows,cuts,copies,hits,misses,bytes_read,cut_ms,copy_ms,wait_ms,walk_ms,gpu_cut_ms,gpu_tail_ms,t_ms,ple_ms,ple_reads,ple_prefetch_misses`

| column | meaning |
|---|---|
| `rows` | **`rows > batch` ⇒ prefill fire** — this is how decode runs are split out (`stepmodel.py:61`) |
| `bytes_read / misses` | **B**, the MiB an expert |
| `cut_ms` / `copy_ms` / `wait_ms` / `walk_ms` | host: cuts (read+seat) / the reads alone / blocked on the device / whole trace walk |
| `gpu_cut_ms` / `gpu_tail_ms` | device: this fire's cut frames / final frames landed since the previous record — in steady decode the tail belongs to the **previous** fire |
| `t_ms` | when this fire **opened**, ms since the load's first fire — the step clock |
| `ple_ms`, `ple_reads`, `ple_prefetch_misses` | n-gram row gather: host time, rows read, rows the prefetch had not landed |

`out/<tag>.cuts.csv` — one row per layer per fire:
`fire,group,rows,misses,hits,copies,bytes_read,copy_ms,wait_ms,gpu_ms,cut_ms`. `fire` joins to `seq`;
`group` **is** the layer index `l`; `misses` **is** `m_l`; `copy_ms` is where `a` and `b` are visible directly.

Derivation, one line: `step_ms[k] = fires[k+1].t_ms − fires[k].t_ms` over consecutive decode fires;
`m_l` = misses of cut rows with `fire == fires[k].seq` bucketed by `group`; `L = #{l : m_l ≥ 1}`.
Reference implementation `$H/stepmodel.py:72-110`.

> At batch N a decode fire carries N rows and advances N sequences by one token each, so **one step is
> still one fire**, but it buys N tokens: divide `1e3 / step_ms` by N for tokens a second a sequence, and
> compare `step_ms` against a batch-1 step only after dividing. Measured here: batch 4 steps at 385.75 ms
> median against batch 1's 116.06 — 3.3× the step for 4× the tokens, so 10.37 tok/s total but 2.59 a
> sequence (pool-deepening issue 14). Check `rows` is the same on every measured fire before averaging:
> the scheduler batches whatever is admitted, so a fire can carry fewer rows than your batch.

Other sidecars: `<tag>.requests.json` (`forced_ok`, `decode_hit_rate`, `pie`, `config`),
`<tag>.macmon.jsonl` (`gpu_freq_mhz`), `<tag>.vmstat.log` / `.iostat.log` (free/wired/compressed/swap
counters, drive throughput), `<tag>.heater.csv` (`opened_ms,closed_ms,kernels,busy_ms,mean_ms,min_ms,max_ms`),
`<tag>.server.log` (the load lines), `out/sm_runs.log` (one line a run: env, extra args, rc, thermal
before→after, memory before / least-free / after — **every** run, including spoiled ones; this is what later
tells a drive run from an engine run).

### 8d. Minimum engine commits

`e949089b` (the pool) + `0a08e1a9` (frames report device time) + `e7c12238` (`t_ms`, the step clock) +
`fba613c0` (the cut log) + `31a96628` (planted misses, final frame counted) + `75452854` / `eabe2fd6`
(heater knobs, CPU heater) — the seven §0 gate 1 tests. The rest of `metal-expert-cache` removes confounds
rather than enabling a column; full list: `git -C $P log --oneline main..HEAD -- crates/engine-metal`, and
the experiments are in `$R/issues/01..10`.

---

## Appendix A — what this box measured

**One machine's answers, 2026-09-18.** Apple M4 Pro 48 GB, mlx-community/Qwen3.8-Flash-Next-4bit
(48 layers = 36 GDN + 12 full attention, 512 experts, top-k 10), batch 1, 1k prompt, 1024-token forced
sequence. **Do not** carry any of them to another box.

| term | value | how it was got |
|---|---|---|
| B | 2.637 MiB | `bytes_read / misses`; load line `13200 seats x 2.64 MiB` |
| floor (C + F) | **37.05 ms** | 3 zero-miss reps 37.34 / 36.89 / 36.95, `out/validate-sm4.txt:3,7` (post-issue-08 `sm4-*` runs) |
| C | 32.86 ms | cut frames 29.83 + final frame 3.04, same file |
| F | 4.19 ms | turnaround 2.69, seating 0.21, n-gram rows 0.02, encode 0.96, after the walk 0.31 (issue 08), same file |
| a | **0.368 ms a call** | minimax over c1, c2, c3, mid |
| b | **0.1360 ms a MiB** (= 0.359 ms an expert) | same |
| drive's own cut curve | `copy_ms = 0.387 + 0.357 m` (r2 0.843, 9568 cuts) | 15 planted-miss runs — the minimax pick lands on it |
| worst clean-window error | **1.77%** at 11264 / 13200 seats | `out/validate-sm7.txt:22`. On the fitted `sm4` set: 3.96% (a drive-slow c1 rep) |

    step_ms = 37.05 + sum over layers with m_l >= 1 of ( 0.368 + 0.1360 * m_l * 2.637 )   valid to 13200 seats

| conditioning | this box's answer |
|---|---|
| heater | `PIE_METAL_HEATER=on PIE_METAL_HEATER_ARM=always PIE_METAL_HEATER_KERNEL=alu PIE_METAL_HEATER_ALU_ITERS=8192` — cut frames 29.7 ms at 0 misses vs 31.5 at 331 |
| host read path | `PIE_METAL_CPU_HEATER=1` — 0.53–0.58 ms a 1-expert call at every gap 0–30 ms (bare: 0.71 at G=0, 1.44 at 30 ms, 2.34 only at 100 ms) |
| page cache | `PIE_PLE_SOURCE=pread PIE_PLE_PREFETCH=1` — prime/resident 0.01 / 0.19 ms, 0 prefetch misses; floor 37.69 → 37.05 |
| pool | `seats_max = 8192` for the four configurations — note this is `sm_probe_seats.sh`'s **fallback default**, not a measured pick: every candidate failed `misses/step < 0.5` because the probe's 128-token window is three times what the pool holds. It was confirmed afterwards by c3 at `W = 40` and by `sm_sweep5.sh`. After the issue-08 and issue-10 fixes the flat model holds to **13200 seats ≈ 42 GB pinned** |
| window | `WIN = 40` tokens zero-miss at 8192 seats (64 and 48 still missed in their first steps); `u ≈ 131` new pairs a step |
| floor pool | 1024 seats (= the ring) with the ring on; 512 with `PIE_EXPERT_CACHE_PREFILL=0` |
| mid points | 1536 / 3072 / 4000 / 6144 seats |
| PLE_MAX | 0.5 (cached 0.02–0.1 vs faulting 5–7); 2.0 for the `sm4`/`sm5`/`sm7` reruns |
| FIXED_GB | 6.5 GB, plus ~1.5 GB of other resident memory |
| rep spread | ±1.5% on a 48-step window → windows are 256 steps |
| chunk layout | 1 chunk at 8192 seats, 2 at 13200 (`gate_up` in chunk 0, `down` in chunk 1) — harmless with `8a98f324` in: 0.747 ms a 1-miss call at 13200 against 0.706 at 8192 |
| fifth regime, apart | reads within seconds of a 55 GiB prefill burst: 0.60 ms a miss vs 0.53, a 128-token window reads −8% |

**Reproduce.** `out/validate-final.txt` is the **superseded** pre-issue-08 run (floor 37.69, worst
5.17% / 1.75% outside the planted grid); it is kept as a record and is not what this table reports.

```bash
cd /Users/yecl/hongseung/scratch/qwen38-profile
export HEAT="PIE_METAL_HEATER=on PIE_METAL_HEATER_ARM=always PIE_METAL_HEATER_KERNEL=alu PIE_METAL_HEATER_ALU_ITERS=8192"
SEATS_MAX=8192 WIN=40 ./sm_sweep3.sh c3 c4 c1 c2 mid            # ~1 h, writes sm-* tags
# the reported floor, C and F (post-issue-08 sm4-* runs):
.venv/bin/python validate.py --prefix sm4 --ple-max 2.0 --win 40 --last 256 \
  --fixed-a 0.368 --fixed-b 0.1360 --name mac_m4pro_qwen38flash_np1_v6
# the 11264 / 13200-seat re-validation against fixed parameters:
.venv/bin/python validate.py --prefix sm7 --seats any --ple-max 2.0 --win 40 --last 256 \
  --fixed-floor 37.05 --fixed-a 0.368 --fixed-b 0.1360 --name mac_m4pro_qwen38flash_np1_v6_large
# a fresh fit of your own sm-* sweep (never --name ..._v5: that is issue 10's config, cited by RESULTS.md):
.venv/bin/python validate.py --seats 8192 --win 40 --last 256 --fit-on minimax \
  --name mac_m4pro_qwen38flash_np1_v6 | tee out/validate-v6.txt
```

Full write-ups: `$R/spec.md` (the form and 14 decisions), `$R/RESULTS.md` (the tables),
`$R/issues/01..10-*.md` (one experiment each), `$P/.scratch/metal-expert-pool-deepening/issues/14,15`
(batch and kernel evidence), `$H/out/sm_runs.log` (231 runs with env, thermal and memory).
