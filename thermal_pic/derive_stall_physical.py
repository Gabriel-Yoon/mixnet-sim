#!/usr/bin/env python3
"""Re-derive the thermal-tuning stall duration via a proper physical chain:

    T(t) -> Delta-lambda_thermal(t) = (dlambda/dT) * (T(t) - T(0))
         -> controller tracks Delta-lambda_thermal(t) subject to a MAXIMUM
            SLEW RATE (a real closed-loop wavelength-locking speed, not an
            idealized "instant except above some K/ms threshold" proxy)
         -> residual detuning eps(t) = |Delta-lambda_thermal(t) - lambda_track(t)|
         -> the link is "stalled" whenever eps(t) exceeds an acceptable-penalty
            detuning budget defined from the ring's resonance linewidth (Q).

This replaces the previous proxy (delay = time for dT/dt to fall below an
ad hoc K/ms rate) with a delay defined by an actual optical-domain criterion,
answering the reviewer's core objection: a fast cooling *rate* is not the same
thing as a communication outage.

Parameters (all cited, not invented):
  dlambda/dT = 80 pm/K            -- standard Si microring thermo-optic coefficient
  R_ctrl     = 0.0625 K/ms          -- the manuscript's existing thermo-optic tracking-
                                       loop figure [PID], converted to a wavelength
                                       slew rate via dlambda/dT (= 5 pm/ms = 5000 pm/s)
  Q          = 8000                -- representative loaded Q for a DWDM ring
                                       modulator, consistent with the <1 nm optical
                                       bandwidth already reported in Table 1
  FWHM       = lambda / Q           -- resonance full width at half maximum
  eps_max    = 0.10 * FWHM          -- acceptable-detuning budget (10% of FWHM is a
                                       conservative, commonly used design margin
                                       bounding the additional insertion-loss
                                       penalty to roughly 1 dB for a Lorentzian
                                       resonance)
"""
import re

DLAM_DT_PM_PER_K = 80.0          # pm/K
# Controller tracking rate: keep the manuscript's existing thermo-optic tracking-loop
# figure (0.0625 K/ms, from [PID]) and convert it into a wavelength-domain slew rate
# via dlambda/dT, rather than importing a different reported number. This keeps the
# already-cited assumption but now runs it through a proper T->lambda->penalty chain
# instead of comparing a raw dT/dt to a raw K/ms threshold.
R_CTRL_K_PER_MS = 0.0625
R_CTRL_PM_PER_MS = R_CTRL_K_PER_MS * DLAM_DT_PM_PER_K
R_CTRL_PM_PER_S = R_CTRL_PM_PER_MS * 1000.0
LAMBDA_NM = 1550.0
Q_FACTOR = 8000.0
FWHM_PM = (LAMBDA_NM * 1000.0) / Q_FACTOR   # nm->pm
EPS_MAX_PM = 0.10 * FWHM_PM


def parse_prvar(path):
    times, temps = [], []
    with open(path) as f:
        started = False
        for line in f:
            line = line.strip()
            if line.startswith("TIME"):
                started = True
                continue
            if started:
                m = re.match(r"^([\d.Ee+-]+)\s+([\d.Ee+-]+)$", line)
                if m:
                    times.append(float(m.group(1)))
                    temps.append(float(m.group(2)))
    return times, temps


def simulate_residual(t, T):
    """Rate-limited wavelength tracker; returns (eps_pm[], dlam_thermal_pm[])."""
    T0 = T[0]
    dlam = [(Ti - T0) * DLAM_DT_PM_PER_K for Ti in T]
    lam_track = [0.0] * len(t)
    for i in range(1, len(t)):
        dt_ms = (t[i] - t[i - 1]) * 1000.0
        if dt_ms <= 0:
            lam_track[i] = lam_track[i - 1]
            continue
        desired = dlam[i] - lam_track[i - 1]
        max_step = R_CTRL_PM_PER_MS * dt_ms
        step = max(-max_step, min(max_step, desired))
        lam_track[i] = lam_track[i - 1] + step
    eps = [abs(dlam[i] - lam_track[i]) for i in range(len(t))]
    return eps, dlam


def find_stall_windows(t, eps, eps_max):
    """Return list of (start_s, end_s, duration_ms) where eps > eps_max."""
    windows = []
    in_stall = False
    start = None
    for i in range(len(t)):
        if eps[i] > eps_max and not in_stall:
            in_stall = True
            start = t[i]
        elif eps[i] <= eps_max and in_stall:
            in_stall = False
            windows.append((start, t[i], (t[i] - start) * 1000.0))
    if in_stall:
        windows.append((start, t[-1], (t[-1] - start) * 1000.0))
    return windows


if __name__ == "__main__":
    print(f"dlambda/dT = {DLAM_DT_PM_PER_K} pm/K, R_ctrl = {R_CTRL_PM_PER_S} pm/s "
          f"[PID, converted via dlambda/dT], Q = {Q_FACTOR:.0f}, FWHM = {FWHM_PM:.1f} pm, "
          f"eps_max = {EPS_MAX_PM:.2f} pm (10% FWHM)")
    print()
    for name, path in [("Mixtral-8x7B", "mixtral8x7b_pic_transient_v2.csv"),
                        ("LLaMA-MoE-6.7B", "llamamoe_pic_transient_v2.csv")]:
        t, T = parse_prvar(path)
        eps, dlam = simulate_residual(t, T)
        windows = find_stall_windows(t, eps, EPS_MAX_PM)
        durations = [w[2] for w in windows if w[2] > 1.0]
        print(f"=== {name} ===")
        print(f"  thermal detuning range: [{min(dlam):.1f}, {max(dlam):.1f}] pm "
              f"(peak-to-peak {max(dlam)-min(dlam):.1f} pm = "
              f"{(max(dlam)-min(dlam))/FWHM_PM*100:.1f}% of FWHM)")
        print(f"  peak residual (post-controller) detuning: {max(eps):.1f} pm")
        print(f"  time to slew full swing at R_ctrl: "
              f"{(max(dlam)-min(dlam))/R_CTRL_PM_PER_MS:.1f} ms")
        print(f"  number of stall windows > 1ms: {len(durations)}")
        if durations:
            print(f"  stall durations (ms): {[f'{d:.1f}' for d in durations[:12]]}")
            print(f"  mean={sum(durations)/len(durations):.1f} ms, "
                  f"max={max(durations):.1f} ms, min={min(durations):.1f} ms")
        print()
