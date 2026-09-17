# 16 What a step costs, layer by layer

Status: resolved
Type: research

For the step-time simulator: a step is the kernels, a per-cut fixed cost,
and the host blocked reading this step's misses. `PIE_EXPERT_CACHE_CUT_LOG`
prices the last term per layer, and `--mode all_in_mem` measures the first
two directly instead of extrapolating to them.

## A layer's read

One row a cut is one layer's own step. Fitting `copy_ms ~ a + b*m` over the
cuts of a batch-1 decode at 4000 seats, 21234 of them:

    copy_ms = 0.346 + 0.360 * misses      r2 0.983
    a miss = 0.360 ms = 0.1364 ms a MiB at 2.637 MiB an expert (7.33 GiB/s)

Cuts that miss nothing read 0.000 ms, so the `m >= 1` condition is right.
The coefficients are the same layer to layer (b within 0.352-0.365 over all
48, a within 0.31-0.39); what differs is how often a layer misses, from 4.98
misses a step at layer 0 to 1.44 at layer 40.

**How many layers call matters as much as how many misses there are.** At
4000 seats a step misses in 41.47 of its 48 cuts; at 11204 seats, in 14.26.
The call is a third of the read at the first and more than half at the
second.

The per-call cost is not a constant of the drive. At 11204 seats, where 70%
of cuts read nothing, the fit is `0.833 + 0.441 * m`: a cut with one miss
costs 1.275 ms against 0.712 ms at 4000 seats. A read path that is busy
amortises; one that is idle pays to wake up.

## A step with nothing to read

`--mode all_in_mem` fires the same forced request twice: the first seats
what the second routes to. At 11204 seats, 20 tokens, the second request hit
1.000 of its routes and read 0.00 ms.

    first step after the prefill   74.52 ms   (discard it)
    the other 17                   45.66 ms median, 45.29-48.23, 21.9 tok/s
    of which device                30.59 ms
    host waiting on the device     36.50 ms   (5.9 ms of turnaround over 48 cuts)

So the floor is **45.7 ms a step** on this box for this model at batch 1.
The 58.69 ms the pooled fits called "the step at no misses" is 13 ms high,
because it extrapolates from runs whose device time was already inflated by
their own idling (issue 15).

## The model that fits what was measured

    step = (45.9 + 0.037 * m) + a * L + b * m

      m = misses a step, L = layers that miss (the SSD calls)
      a, b = 0.346, 0.360 where reads are frequent; 0.833, 0.441 where they
             are rare; b = 0.1364 ms a MiB x the expert's MiB
      the first bracket is compute and the per-cut fixed cost, which climbs
      with the miss rate and saturates near 56

Against the sweep at batch 1: 264 misses 168.2 modelled against 168.27
measured, 142 -> 115.4 against 117.74, 96 -> 96.4 against 98.63, 20 -> 67.4
against 69.20, 0 -> 45.9 against 45.90. Within 3%.

## Where the tools are

`scratch/qwen38-profile/` (outside git): `coldsteps.py` reads a cold
request's own steps, `cuts.py` prices a layer's misses, `kernels.py` reads
`--diag kernel-profile` blocks, `resident.py` reads an `all_in_mem` run,
`batched.py` drives concurrent sequences. The engine side is committed:
`t_ms` on the fire record (e7c12238), the per-cut log (fba613c0), and
`PIE_EXPERT_CACHE_COLD` (699e2f87).
