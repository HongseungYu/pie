# Decode-step model: results

Branch `metal-expert-cache`, 2026-09-17. Batch 1, mlx-community/Qwen3.8-Flash-Next-4bit
(`qwen38-flash-next-u4g64-kv-bf16`), M4 Pro 48 GB, 1k prompt, the teacher-forced 1024-token
sequence of `out/forced_tokens.json`. Reader: `scratch/qwen38-profile/stepmodel.py`;
pipeline: `validate.py`; runs `out/sm-*` (`out/sm_runs.log` lists every run with its
thermal state and memory before, during and after).

## The form

    step_ms = C + F + sum over layers with m_l >= 1 of ( a + b * m_l * B ),   B = 2.637 MiB

## What had to be conditioned first

| effect | symptom | cure | knob |
|---|---|---|---|
| GPU heater's own kernels beside the frames | cut frames 30.6 ms at no misses, 43.4 at 331, clock at 1578 MHz throughout (issue 02) | a narrow ALU kernel at every gap | `PIE_METAL_HEATER=on PIE_METAL_HEATER_ARM=always PIE_METAL_HEATER_KERNEL=alu` |
| host asleep between reads | a call 0.35 ms when dense, 0.83 when rare (issue 03) | one thread spinning | `PIE_METAL_CPU_HEATER=1` |
| n-gram (PLE) rows faulting from disk | 6-7 ms a step at pools >= 9216 that page the host, or on a sequence's first pass (issue 05) | pool <= 9216 seats; warm the sequence first | (measurement protocol) |

## Parameters

(filled by `validate.py`)

## The four configurations

(filled by `validate.py`)

## C by layer type and module

(from `sm-kern-*`, `--diag kernel-profile=2`)

## How to reproduce

    HEAT="PIE_METAL_HEATER=on PIE_METAL_HEATER_ARM=always PIE_METAL_HEATER_KERNEL=alu PIE_METAL_HEATER_ALU_ITERS=8192"
    SEATS_MAX=9216 WIN=48 ./sm_sweep3.sh c3 c4 c1 c2 mid     # ~1 h
    .venv/bin/python validate.py --win 40 --last 256
