# 05 The largest pool is the largest the host can spare

Status: claimed
Type: research

Configuration 2 asks for the largest pool that boots, about 11204-12000 seats on this 48 GB
box. Measured at 11204 seats (28.85 GiB of seats, ~35 GiB wired with the planes), the host
between cuts — the walk's own time outside every wait, read and seating, `enc` in
`stepmodel.py` — costs 7.2 ms a step at no misses, against 1.2-1.7 ms at 1536-6144 seats,
and it decays over a long run (7.8 -> 4.4 ms over 1000 steps in `cold-s11204`). It does not
follow the miss count (r2 0.00 against L within a run) and it is not the cut log (7.19 ms
with the log off, `sm-e1-nocut`).

`vm_stat` during an 11204-seat run: **5231 pages free (80 MB)**, 714k pages in the
compressor, 1.25 GB swapped. The host's other memory (8-9 GB compressed before any run) has
nowhere to go once 35 GiB are wired, so the encode path — the Metal driver, the walk — runs
against a compressor, and so do the reader threads: a one-miss call costs 0.99 ms here
(`sm-g1-all1`) against 0.71 with dense reads at 4000 seats.

That cost is real, but it is the operating system's, not the engine's or the model's: it
depends on what else the machine holds, and it drifts. A model with one F cannot carry it,
and a simulator should not have to. So configuration 2 is the largest pool that leaves the
host its memory: `sm_probe_seats.sh` tries 8192 / 9216 / 10240 / 11204 with `all_in_mem`
128 tokens and takes the largest whose run ends with 2 GB or more free and whose resident
pass misses nothing.

Every run's memory before and after is now on `out/sm_runs.log`.
