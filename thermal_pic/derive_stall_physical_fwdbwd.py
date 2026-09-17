#!/usr/bin/env python3
"""Same physical stall-derivation chain as derive_stall_physical.py, applied to
the new fwd+bwd (full-iteration) ANSYS transient traces instead of the
forward-only v2 traces."""
from derive_stall_physical import (
    parse_prvar, simulate_residual, find_stall_windows,
    DLAM_DT_PM_PER_K, R_CTRL_PM_PER_S, Q_FACTOR, FWHM_PM, EPS_MAX_PM,
)

if __name__ == "__main__":
    print(f"dlambda/dT = {DLAM_DT_PM_PER_K} pm/K, R_ctrl = {R_CTRL_PM_PER_S} pm/s "
          f"[PID, converted via dlambda/dT], Q = {Q_FACTOR:.0f}, FWHM = {FWHM_PM:.1f} pm, "
          f"eps_max = {EPS_MAX_PM:.2f} pm (10% FWHM)")
    print()
    for name, path in [("Mixtral-8x7B", "mixtral8x7b_pic_transient_fwdbwd.csv"),
                        ("LLaMA-MoE-6.7B", "llamamoe_pic_transient_fwdbwd.csv")]:
        t, T = parse_prvar(path)
        eps, dlam = simulate_residual(t, T)
        windows = find_stall_windows(t, eps, EPS_MAX_PM)
        durations = [w[2] for w in windows if w[2] > 1.0]
        # exclude the first window: it is the one-time startup transient from
        # ambient (T0) and does not repeat during steady-state training, unlike
        # every subsequent per-iteration stall.
        steady = durations[1:]
        print(f"=== {name} ===")
        print(f"  thermal detuning range: [{min(dlam):.1f}, {max(dlam):.1f}] pm "
              f"(peak-to-peak {max(dlam)-min(dlam):.1f} pm = "
              f"{(max(dlam)-min(dlam))/FWHM_PM*100:.1f}% of FWHM)")
        print(f"  peak residual (post-controller) detuning: {max(eps):.1f} pm")
        print(f"  number of stall windows > 1ms: {len(durations)} (incl. 1 startup transient)")
        if durations:
            print(f"  stall durations (ms): {[f'{d:.1f}' for d in durations[:12]]}")
            print(f"  startup-transient window: {durations[0]:.1f} ms (excluded from mean below)")
            print(f"  steady-state: n={len(steady)}, mean={sum(steady)/len(steady):.1f} ms, "
                  f"max={max(steady):.1f} ms, min={min(steady):.1f} ms")
        print()
