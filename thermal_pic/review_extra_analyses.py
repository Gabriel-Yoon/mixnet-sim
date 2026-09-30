#!/usr/bin/env python3
"""Extra analyses requested by the mock reviews (all local, from existing ANSYS traces + ROM):
 A. detuning-budget sweep up to 30% FWHM (1 dB Lorentzian penalty = 25% FWHM)
 B. GPU power smoothing: hold a dummy load during waits (idle floor raised), via the ROM
 C. swing sensitivity: temperature excursion scaled 0.5x / 2x; idle power 12% -> 6% / 25%
 D. event-triggered feed-forward: heater pre-driven from the known power command with a timing error
"""
import json
import numpy as np
import rom_fulldepth as R
from derive_stall_physical import DLAM_DT_PM_PER_K
from stall_model_v2 import track, per_a2a_round_stall, load_a2a_windows, FWHM_PM, R_CTRL_BASE

KEYS = ["mixtral_median", "llama_median", "llama_cold", "qwen_median"]
g = R.fit("mixtral_median")


def stall(tg, Tg, wins, frac=0.10, r=R_CTRL_BASE, ff=None):
    t = list(tg)
    dlam = [(x - Tg[0]) * DLAM_DT_PM_PER_K for x in Tg]
    eps = track(t, dlam, r * DLAM_DT_PM_PER_K, feedforward=ff)
    v = [x for x, _, _ in per_a2a_round_stall(t, eps, frac * FWHM_PM, wins, R.DT)]
    return float(np.mean(v))


out = {}
for key in KEYS:
    rows, tg, Tg, p, T0 = R.load_pair(key)
    wins = load_a2a_windows(R.PAIRS[key][0].replace(".csv", "_a2a_windows.csv"))
    o = {}
    o["A_budget"] = {f: stall(tg, Tg, wins, frac=f) for f in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)}
    base_rom = stall(tg, R.predict(g, p, T0), wins)
    o["rom_base"] = base_rom
    B = {}
    for floor in (84, 175, 350, 525, 700):
        p2 = np.maximum(p, floor)
        T2 = R.predict(g, p2, T0)
        m = tg >= 1.6
        B[floor] = dict(stall_ms=stall(tg, T2, wins), swing_K=float(T2[m].max() - T2[m].min()),
                        mean_power_W=float(p2.mean()), extra_W=float(p2.mean() - p.mean()))
    o["B_smoothing"] = B
    C = {}
    for sc in (0.5, 1.0, 2.0):
        C[f"swing_x{sc}"] = stall(tg, T0 + sc * (Tg - T0), wins)
    for idle_frac in (0.06, 0.25):
        p3 = np.where(p <= 84.0 + 1e-6, idle_frac * 700.0, p)
        C[f"idle_{idle_frac}"] = stall(tg, R.predict(g, p3, T0), wins)
    o["C_sensitivity"] = C
    # D: feed-forward from the power command through the ROM, with timing error and gain error
    D = {}
    Trom = R.predict(g, p, T0)
    for jit_ms in (0, 1, 2, 5, 10):
        n = int(round(jit_ms / 1e3 / R.DT))
        pred_T = np.concatenate([np.full(n, Trom[0]), Trom[:len(Trom) - n]]) if n else Trom
        for gerr in (0.0, 0.1):
            ff = [float((x - Tg[0]) * (1 - gerr) * DLAM_DT_PM_PER_K) for x in pred_T]
            D[f"late{jit_ms}ms_gainerr{int(gerr*100)}"] = stall(tg, Tg, wins, ff=ff)
    o["D_event_ff"] = D
    out[key] = o
    print(f"\n== {key} (ANSYS base {o['A_budget'][0.10]:.1f} ms, ROM base {base_rom:.1f} ms)")
    print("  A budget %FWHM -> stall/round ms: " + ", ".join(f"{int(f*100)}%: {v:.1f}" for f, v in o["A_budget"].items()))
    print("  B idle floor W -> stall ms / swing K / extra W: " + ", ".join(f"{k}: {v['stall_ms']:.1f}/{v['swing_K']:.1f}/{v['extra_W']:.0f}" for k, v in B.items()))
    print("  C: " + ", ".join(f"{k}: {v:.1f}" for k, v in C.items()))
    print("  D: " + ", ".join(f"{k}: {v:.1f}" for k, v in D.items()))
json.dump(out, open("review_extra_analyses.json", "w"), indent=1)
