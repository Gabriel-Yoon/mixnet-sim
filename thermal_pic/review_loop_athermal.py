#!/usr/bin/env python3
"""E. linear loop-bandwidth model (type-1 / first-order tracking, no slew limit) on the ANSYS traces
   F. slew limit AND bandwidth combined
   G. athermal LN ring with second-order residual a*(T-Tath)^2, a = 0.37 pm/K^2 (Ling 2020):
      residual excursion during the workload swing vs placement of the athermal point
"""
import json, math
import numpy as np
import rom_fulldepth as R
from derive_stall_physical import DLAM_DT_PM_PER_K
from stall_model_v2 import per_a2a_round_stall, load_a2a_windows, FWHM_PM, EPS_MAX_PM

KEYS = ["mixtral_median", "llama_median", "llama_cold", "qwen_median"]
A2 = 0.37  # pm/K^2


def loop_eps(tg, dlam, fc, slew_pm_per_ms=None):
    k = 2 * math.pi * fc * R.DT
    k = 1 - math.exp(-k)
    c = 0.0
    eps = np.zeros(len(dlam))
    for i in range(len(dlam)):
        step = k * (dlam[i] - c)
        if slew_pm_per_ms is not None:
            lim = slew_pm_per_ms * R.DT * 1e3
            step = max(-lim, min(lim, step))
        c += step
        eps[i] = abs(dlam[i] - c)
    return eps


def stall_from_eps(tg, eps, wins, budget=EPS_MAX_PM):
    v = [x for x, _, _ in per_a2a_round_stall(list(tg), list(eps), budget, wins, R.DT)]
    return float(np.mean(v)), float(np.max(v))


out = {}
for key in KEYS:
    rows, tg, Tg, p, T0 = R.load_pair(key)
    wins = load_a2a_windows(R.PAIRS[key][0].replace(".csv", "_a2a_windows.csv"))
    dlam = (Tg - Tg[0]) * DLAM_DT_PM_PER_K
    E = {}
    for fc in (1, 3, 10, 30, 100, 300, 1000):
        E[fc] = stall_from_eps(tg, loop_eps(tg, dlam, fc), wins)
    F = {fc: stall_from_eps(tg, loop_eps(tg, dlam, fc, 5.0), wins) for fc in (100, 1000)}
    m = tg >= 1.6
    Tmin, Tmax, Tmean = float(Tg[m].min()), float(Tg[m].max()), float(Tg[m].mean())
    G = {}
    for off in (0, 1, 2, 3, 5, 10, 20, 34):
        Tath = Tmean + off
        lam = A2 * (Tg - Tath) ** 2
        exc = float(lam[m].max() - lam[m].min())
        # static setpoint centred on the mid excursion: worst detuning is half the excursion
        G[off] = dict(excursion_pm=exc, worst_detuning_pm=exc / 2, within_budget=bool(exc / 2 <= EPS_MAX_PM))
    out[key] = dict(E_bandwidth=E, F_bw_plus_slew=F, G_athermal=G, T=(Tmin, Tmean, Tmax))
    print(f"\n== {key}  T min/mean/max {Tmin:.1f}/{Tmean:.1f}/{Tmax:.1f} C; max |dT/dt| {np.abs(np.diff(Tg[m])).max()/R.DT/1e3:.2f} K/ms")
    print("  E loop bandwidth Hz -> stall/round mean (max) ms: " + ", ".join(f"{f}: {a:.1f} ({b:.0f})" for f, (a, b) in E.items()))
    print("  F with 5 nm/s slew limit: " + ", ".join(f"{f} Hz: {a:.1f}" for f, (a, b) in F.items()))
    print("  G athermal point offset from mean (K) -> excursion pm / worst detuning pm: " +
          ", ".join(f"{o}: {v['excursion_pm']:.0f}/{v['worst_detuning_pm']:.0f}{'' if v['within_budget'] else '*'}" for o, v in G.items()))
json.dump(out, open("review_loop_athermal.json", "w"), indent=1, default=float)
