# Ferroelectric tuning section — analytic rework (no hardware)

Numbers feeding the revised Sec. 2.5 / 4.8. Stall-model inputs from stall_model_v2_L4seq4096.json
(interim coarse traces; refresh after the 1 ms ANSYS reruns).

## 1. What the cited ferroelectric device actually provides (Xu et al., Nat Commun 16:8329, 2025)
- Stack: z-cut LNOI ring (600 nm film, R = 150 um) with FeFET gate: HZO 8 nm / IGZO 5 nm / ITOx 3 nm.
- Tuning range: ~1 nm total resonance shift, in 6 non-volatile states per FeFET (16 with three stacked
  FeFETs) -> step size ~170 pm (6 states) or ~60 pm (16 states).
- Write: -6 V / +3 V pulses, optimized pulse width 10 us (RC-limited), 65 fJ/state.
- Retention: ~28 h measured, 10-year extrapolation; <15 % polarization loss below 90 C.
- Endurance: > 1e7 read/write cycles, minimal degradation to 1e5.
- States are written by partial polarization switching (pulse-energy controlled).
- NOT reported: thermo-optic drift of the ring. A plain LNOI ring drifts ~26 pm/K.
Implications for the manuscript:
  (a) The manuscript's "5.4 ns switching" comes from a different HZO capacitor paper
      (Si et al. APL 2019); the photonic device switches in ~10 us. Still 3-4 orders below the
      ms-scale stall, so the latency argument survives, but the citation must be corrected.
  (b) Range (~1 nm) covers the 0.73-0.99 nm workload swing, but the *resolution* (60-170 pm)
      is 3-9x coarser than the 19.4 pm acceptable-detuning budget. The published device cannot
      hold a resonance inside the budget while the substrate drifts; it would need >= 50 levels
      or a hybrid coarse (ferroelectric) + fine (EO/thermal) trim.

## Citations to add to refs.bib
- ln_athermal: J. Ling, Y. He, R. Luo, M. Li, H. Liang, Q. Lin, "Athermal lithium niobate microresonator,"
  Opt. Express 28(14), 2020, doi 10.1364/OE.398363 — TiO2-clad x-cut LN ring, first-order TOC nulled at 20 C,
  residual second-order 0.37 pm/K^2.
- ln_athermal_broadband: P. Han, L. Yang, L. Xu, X. Zhou, L. Cai, A. M. Agarwal, J. Michel, L. C. Kimerling,
  L. Zhang, "Broadband Athermal Lithium Niobate Microresonators Across C and L Bands," J. Lightwave Technol.
  42, 3246-3250, 2024 (IEEE Xplore 10415485) — LN/TiO2 hybrid, |dlambda/dT| <= 1 pm/K over 1480-1630 nm.
- (MDPI sources are NOT to be used — user rule.) Bare-LN-ring thermo-optic sensitivity must come from
  Ling et al. OE 2020 (uncladded reference device) or another Optica/IEEE/APS source.

## 2. Residual thermal drift is not removed by changing the actuator
- LN ring thermo-optic sensitivity: ~26 pm/K (LNOI ring, literature) vs 80 pm/K for Si.
- Over the steady-state workload swing (9.1 K Mixtral/LLaMA, 4.4 K Qwen):
  plain LNOI ring drift = 237 pm / 115 pm  >> 19.4 pm budget  -> tracking still required.
  => "removes the continuous thermal-tuning requirement" (current text) is not supported;
     the mechanism replaces a slow actuator with a fast, non-volatile one.
- Athermalized LN ring (TiO2 cladding, residual ~ +/-1 pm/K, literature): drift = 9.1 pm / 4.4 pm
  < 19.4 pm budget -> NO tracking needed during training; ferroelectric tuner becomes a
  one-time non-volatile trim for fabrication/aging offsets.

## 3a. CORRECTION (from Job A a2a data): the fbuf has 8 micro-batches per iteration (128 a2a
rounds = 4 layers x 2 dirs x 2 ops x 8 mb; batch 128 / dp 2 = 64 sequences per replica / 8 =
8 micro-batches), not 16. All "x16" factors below become x8: stall-window switches per
iteration 47 / 24 / 31 (Mixtral / Qwen / LLaMA), analog tracking writes ~2,400 / 1,170 / 2,400.
Conclusions unchanged (still >= 1e7 well before 1e6 iterations for tracking; 4.7e7 / 2.4e7 /
3.1e7 for the stall-window count at 1e6 iterations).
Also from Job A (llama): real a2a rounds are 0.37 ms mean / 0.84 ms max, vs 350 ms assumed —
the thermal schedule is being rebuilt from the simulated per-device timeline
(gen_power_schedule_from_htsim.py), so every swing/stall number here will be refreshed.

## 3. Endurance, recounted
Per TRUE training iteration (batch 128 / microbatch 8 = 16 microbatches; the thermal "iteration"
in the current paper is one microbatch through one layer):
| model  | swing | analog tracking writes/iter (2*swing/eps per burst x 4 bursts x 16 mb) | 1e6 iters | paper-style "1 switch per stall window" x16 | 1e6 iters |
| Mixtral| 728 pm| ~4800 | 4.8e9 | 85  | 8.5e7 |
| Qwen   | 355 pm| ~2340 | 2.3e9 | 43  | 4.3e7 |
| LLaMA  | 728 pm| ~4800 | 4.8e9 | 56  | 5.6e7 |
Against the cited endurance (1e7 cycles at 125 C from hzo_endurance; >1e7 for the photonic
device): analog tracking of thermal drift with a ferroelectric tuner exhausts endurance in
~2e3-4e3 iterations; even the paper's own (under-counted) accounting exceeds 1e7 within 1e6
iterations once the x16 microbatch factor is applied. The endurance conclusion of Sec. 4.8 reverses.
The athermal-ring + static-trim architecture needs only O(1) writes per device lifetime (plus
re-trim on aging), so endurance is a non-issue there.

## 4. Power
- Thermo-optic: heater bias to cover the swing (0.73 nm at ~0.4 nm/mW typical local-heater
  efficiency) ~1.8 mW per ring -> ~1.4 W static per die for 750 rings; up to 2.9 mW/ring (2.2 W/die)
  at 0.25 nm/mW. Plus the FSR-hop hazard when the bias runs out (heaters only heat).
- Ferroelectric non-volatile: 65 fJ per write; even 5e3 writes/iteration is ~0.3 nJ/iteration -> zero
  static power. Athermal + static trim: zero tuning power in operation.

## 5. Proposed reframing (needs author decision)
Current thesis: ferroelectric EO actuator tracks workload thermal drift with negligible latency.
Problems: resolution (60-170 pm steps vs 19 pm budget), residual LN drift, endurance (1e9-1e10 writes).
Defensible thesis: athermalized LN ring (residual <= 1 pm/K -> < 10 pm over the workload swing)
+ non-volatile ferroelectric trim for static offsets. This removes the stall by removing the drift,
removes static tuning power, and keeps write counts at O(1). The system-level result then reads:
thermo-optic Si MRR + feedback loop of rate R_ctrl -> stall per round (Sec. 4); required loop rate
to avoid stalls >= 0.5 K/ms (interim); athermal-LN + ferroelectric trim -> no stall by construction.
Alternative mitigations to compare in the same table: faster thermo-optic loop (>= 0.5 K/ms),
schedule-aware feed-forward (works if all-to-all timing jitter < ~2-5 ms), lower-Q rings (wider budget,
higher power), MZI-based DWDM (athermal but large).
