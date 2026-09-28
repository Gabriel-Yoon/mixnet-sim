# Consolidated results for the revision (final, 2026-09-28; simulation campaign complete)

All htsim numbers: L4 seq4096 task graphs, paper routing matrices, `-a2a-symmetric` (dispatch sized
like combine), makespan = stderr "finished one iter". Thermal: ANSYS OIO3D stack (Coenen params),
1 ms fixed step, per-device power schedules built from the simulated compute timelines.
Design point: intra-wafer 256 GB/s per row/column link (12 waveguides x 32 λ x 32 Gb/s = 1.5 TB/s
per die), per-reticle CPO port 800 GB/s (`-inter-mode port`).

## T1. Iteration time, stall-free (ms) — iso-injection 1.5 TB/s per GPU
| fabric | Mixtral 8x7B (256 GPU) | LLaMA-MoE 6.7B (128) | Qwen-MoE 14.3B (512) |
|---|---|---|---|
| fat-tree, non-blocking | 841.3 | 427.4 | 195.4 |
| flat (direct all-to-all) | 1152.7 | 1560.1 | 4872.4 |
| wafer, per-wafer-pair gateway 200 GB/s (legacy) | 1954.9 | 493.1 | 1452.1 |
| wafer, per-reticle CPO port 200 GB/s | 890.9 | 495.7 | 807.4 |
| wafer, port 400 GB/s | 873.7 | 490.4 | 678.6 |
| wafer, port 800 GB/s (design point) | 865.3 | 494.4 | 661.3 |
| wafer, port 1600 GB/s | 863.9 | 491.4 | 673.4 |
EP block: Mixtral tp4 x ep8 = 32 GPU (2 wafers); LLaMA 16 (1 wafer); Qwen 64 (4 wafers).
Knee at 400 GB/s per port; 800 saturates. LLaMA flat across port rates (never leaves a wafer).

## T2. Thermo-optic stall penalty at the design point (Job J, corrected round semantics)
| model | no stall | with stall | penalty | stall map GBf/AGGf/GBb/AGGb (ms) |
|---|---|---|---|---|
| Mixtral | 865.3 | 1370.5 | 1.58x | 50.8 / 26.7 / 26.3 / 7.2 |
| LLaMA | 494.4 | 1395.6 | 2.82x | 61.2 / 47.9 / 24.5 / 29.0 |
| Qwen | 661.3 | 1179.4 | 1.78x | 20.9 / 11.2 / 6.7 / 11.5 |
Legacy gateway model, same semantics not run; old-semantics H runs: Mixtral +54% (1800) / +120% (256),
LLaMA +1.4% / +22% (superseded).

## T3. Thermal traces (aPIC / MRR, 1 ms ANSYS, steady state)
| trace | swing (K) | mean (C) | stall per round (ms, mean over rounds) |
|---|---|---|---|
| Mixtral port800 | 11.4 | 61.6 | 27.7 |
| LLaMA port800 median / cold | 9.4 / 10.2 | 59.8 / 60.0 | 19.6 / 38.9 |
| Qwen i1800 median / cold | 3.3 / 4.6 | 55.4 / 55.4 | 11.2 / 10.1 |
Old synthetic schedule (350 ms a2a): 9.1–9.7 K swings, mean window 48.7/46.8/47.4 ms (coarse
ANSYS grid); 1 ms rerun of the same schedule: 51.1 / 17.1 / 35.6 ms.
Cold-plate sensitivity (B5): 1 mm Si = no change; 3 mm Cu swing −45% (LLaMA) / −17% (Mixtral);
10 mm Cu −34% / −23%, operating floor +7.7 K.
Closed loop: iteration 2 within ~1–10% of iteration 1 (converged).

## T4. Controller / mitigation sensitivity (stall model on real timelines, R_ctrl base 0.0625 K/ms)
- Stalls vanish for R_ctrl >= 0.5 K/ms (8x); at 0.2 K/ms per-round stall <= 3 ms.
- Feed-forward from previous iteration: zero stall with exact timing; with a2a timing jitter of
  2 / 5 / 10 / 20 ms → Mixtral 3 / 13 / 31 / 47 ms per round at base R_ctrl, 0 at 10x.
- Lorentzian graded model (no hard stall): 8–85% effective bandwidth loss during comm windows.
- Heater bias to cover a 0.73–0.97 nm swing: 1.5–3.9 mW/ring, 1.1–2.9 W/die (750 rings).

## T5. Wavelength reassignment (uniform → demand-aware, stall-free unless noted)
| case | Mixtral | LLaMA | Qwen |
|---|---|---|---|
| gateway model, intra 256, expert-matrix demand | −12.3% | −9.0% | −10.4% (no sym) |
| gateway model, intra 256, stale matrix | −4.0% | +39.8% | −5.0% (no sym) |
| gateway model, intra 256, measured demand (floor 0.5) | −36.0% | −8.0% | — |
| gateway model + stall (H3→H5) | −43.9% | −8.6% | — |
| port 800, expert-matrix demand | +0.8% | −10.5% | — |
| port 800, measured demand floor 0.5 | −3.4% | — | — |
| port 800 + stall (J) | −2.9% (measured) | +0.9% | +1.3% |
Reading: reassignment is a lever when the intra-wafer mesh / gateway is the bottleneck; it does not
shorten the stall-dominated serialized round chain. Receiver ring over-provisioning: 1.41x (cap 2x).

## T6. H100 cross-validation (Job C, one SXM node, NVML 20 ms)
- Package time constant 6–7 s (heat) / 9 s (cool); 60 s on/off swing 34 C.
- Die-sensor per-cycle swing 13 C for 300 ms on / 350 ms off (420 W peaks), ~4 C for 35 ms bursts (176 W).
- ANSYS stack for the same burst lengths: 10–12 K → consistent within ~2x on the fast timescale;
  slow timescale = package R·C, absent from the bare stack, absorbed by the programmable setpoint.
- Real 8xH100 MoE fwd+bwd (bf16): a2a 44% of a 1.59 s iteration; fwd dispatch 2.7 ms, fwd combine 47 ms,
  bwd dispatch 98 ms at equal bytes.

## T7. Ferroelectric / athermal (analytic, literature)
- Xu et al. device: ~1 nm range, 6–16 levels (60–170 pm steps), 10 us writes, 65 fJ, >1e7 cycles, <15% loss <90 C.
- Bare LN ring 25–50 pm/K → 110–460 pm over the workload swing; athermal TiO2-clad LN <= 1 pm/K → 4–9 pm < 19.4 pm budget.
- Tracking with a ferroelectric tuner: ~1,200 writes/iteration (Mixtral) → 1e7 endurance in ~1e4 iterations. Not viable.
- Programmable setpoint roles: per-device offset (hot vs cold device mean differs 0.2–0.9 K in the old schedule;
  port-800 timelines equalise duty), slow package drift (7–9 s), routing-epoch wavelength reassignment.

## Provenance corrections vs the submitted manuscript
1. 350 ms per all-to-all round assumed; real rounds 0.4–26 ms (gateway) / 0.4–5 ms (port 800).
2. ANSYS AUTOTS quantised stall windows to 14.6 ms steps; 175.0 ms startup window = one max step.
3. FlexFlow export gave GROUP_BY (dispatch) all-to-all 0 bytes; all network results re-run with symmetric sizing.
4. Table 7 "our design" ran at 1800/200 GB/s, not 400 Gbps; replaced by iso-injection comparison.
5. Simulated iteration = 4 micro-batch ids (not batch 128 / mb 8 as Table 2 implies).
6. Per-wafer-pair single gateway replaced by per-reticle CPO ports.
