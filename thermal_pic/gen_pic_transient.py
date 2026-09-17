#!/usr/bin/env python3
"""Generate an ANSYS MAPDL transient thermal .inp for the ICCAD2026 wafer-scale
OIO3D-style GPU/EIC/aPIC 3D stack, driven by a real (not idealized square-wave)
per-model GPU power schedule, to find the actual PIC/MRR thermal fluctuation
rate (dT/dt) -- used to determine a physically-grounded thermal-tuning-delay
instead of an arbitrary sweep value.

Stack + boundary conditions from Coenen et al., "Benchmarking the Thermal Impact
of 2.5D/3D Co-Packaged Optics on Si Photonic Devices," IEEE TCPMT 2026:
  - OIO3D layer order (top->bottom): XPU -> hybrid bond -> EIC -> hybrid bond ->
    aPIC -> D2W bond -> passive PIC (pPIC) interposer. Cooling is applied at the
    XPU TOP side (Si microchannel cold plate); the interposer bottom only sees a
    weak 0.1 K/W path to the package substrate ("majority of heat removed from
    the top side cooler").
  - HTC (top, Si microchannel cold plate) = 7e4 W/m^2K (Fig. 11, conservative ref).
  - Hybrid-bond effective k = t/R, t=2.4um pitch=10um: active(via)=0.48, dummy
    (no-via)=0.94 mm^2-K/W (Table I) -> k_active=5.0 W/mK, k_dummy=2.55 W/mK.
    XPU-EIC and EIC-aPIC bonds are dense/via-populated (active); the aPIC-to-
    passive-interposer D2W bond is evanescent-coupling only, no dense vias
    (dummy).
  - EIC power density 0.426 W/mm^2, aPIC power density 0.113 W/mm^2 (Sec. II-B);
    applied as CONSTANT background self-heating (the paper gives no temporal
    breakdown for these), over the full 25x25mm OIO3D tile footprint (Sec. IV,
    "each tile is 25 x 25 mm^2").
  - XPU (GPU) power: the REAL time-varying schedule from the ICCAD2026 workload
    profiling (thermal_schedule_three_models/*_power_schedule_s.csv), rescaled
    so its peak phase matches Coenen's baseline XPU input power (700 W) --
    the schedule's absolute magnitudes were placeholder/manual guesses; only
    the RELATIVE per-phase shape (from real FlexFlow+htsim timing) is used,
    now anchored to a citable peak power.

All "effective"/assumed values (not directly stated in the paper for OIO3D
specifically) are flagged inline: EIC/aPIC lateral extent = full tile footprint,
XPU die thickness, coolant inlet temperature.
"""
import csv
import sys

MAT_SI = 1        # XPU, EIC, pPIC interposer: bulk silicon
MAT_BOND_ACTIVE = 2   # hybrid bond, via-populated (XPU-EIC, EIC-aPIC)
MAT_APIC = 3       # aPIC: Si-photonics effective (Si core + SiO2 clad)
MAT_BOND_DUMMY = 4    # D2W bond, no dense vias (aPIC-interposer)

W = 0.025          # 25x25 mm OIO3D tile footprint (Coenen, Sec. IV)
ND = 20             # lateral elements/side (matches the project's proven "light mesh" convention)

T_XPU = 700e-6      # XPU die thickness [m] (ASSUMED: typical compute-die thickness)
T_BOND = 2.4e-6     # hybrid/D2W bond thickness [m] (Coenen Table I)
T_EIC = 100e-6      # EIC die thickness [m] (ASSUMED: thin logic/driver die)
T_APIC = 50e-6      # aPIC thickness [m] (matches the project's established PIC-layer convention)
T_INTP = 300e-6     # passive pPIC interposer thickness [m] (ASSUMED: wafer-scale carrier)

TILE_AREA_MM2 = 25.0 * 25.0   # 625 mm^2
EIC_POWER_W = 0.426 * TILE_AREA_MM2   # 266.25 W, constant (Coenen Sec. II-B density)
APIC_POWER_W = 0.113 * TILE_AREA_MM2  # 70.625 W, constant

HCP_TOP = 70000.0     # W/m^2K, Si microchannel cold plate (Coenen Fig. 11)
# Bottom BC: Coenen's 0.1 K/W lumped resistance, converted to an equivalent film
# coefficient over the tile footprint: h = 1 / (R_K_per_W * Area_m2).
R_BOTTOM_K_PER_W = 0.1
TILE_AREA_M2 = (25e-3) * (25e-3)
H_BOTTOM = 1.0 / (R_BOTTOM_K_PER_W * TILE_AREA_M2)  # ~16000 W/m^2K

T_COOLANT_C = 45.0   # ASSUMED liquid-cooling inlet temp (paper doesn't state one explicitly)

XPU_PEAK_TARGET_W = 700.0  # Coenen's baseline XPU input power (Sec. II-B, III-B)

DT_MAX_S = 1e-3   # maximum transient time step [s]; must be << the ~16 ms XPU thermal time constant


def load_schedule(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["start_s"]), float(r["end_s"]), float(r["power_w"])))
    return rows


def generate(csv_path, out_inp, out_csv, monitor="apic"):
    rows = load_schedule(csv_path)
    peak_w = max(p for _, _, p in rows)
    scale = XPU_PEAK_TARGET_W / peak_w

    z = {}
    z["xpu0"] = 0.0
    z["xpu1"] = z["xpu0"] + T_XPU
    z["b1_1"] = z["xpu1"] + T_BOND
    z["eic1"] = z["b1_1"] + T_EIC
    z["b2_1"] = z["eic1"] + T_BOND
    z["apic1"] = z["b2_1"] + T_APIC
    z["b3_1"] = z["apic1"] + T_BOND
    z["intp1"] = z["b3_1"] + T_INTP

    lines = []
    a = lines.append
    a("/PREP7")
    a("! ===== ICCAD2026 OIO3D GPU/EIC/aPIC transient (Coenen et al. TCPMT 2026 params) =====")
    a(f"MP,KXX,{MAT_SI},150.0 $ MP,KYY,{MAT_SI},150.0 $ MP,KZZ,{MAT_SI},150.0 $ MP,DENS,{MAT_SI},2330 $ MP,C,{MAT_SI},700")
    a(f"MP,KXX,{MAT_BOND_ACTIVE},5.0 $ MP,KYY,{MAT_BOND_ACTIVE},5.0 $ MP,KZZ,{MAT_BOND_ACTIVE},5.0 $ MP,DENS,{MAT_BOND_ACTIVE},2500 $ MP,C,{MAT_BOND_ACTIVE},800")
    a(f"MP,KXX,{MAT_APIC},80.0 $ MP,KYY,{MAT_APIC},80.0 $ MP,KZZ,{MAT_APIC},80.0 $ MP,DENS,{MAT_APIC},2300 $ MP,C,{MAT_APIC},700")
    a(f"MP,KXX,{MAT_BOND_DUMMY},2.55 $ MP,KYY,{MAT_BOND_DUMMY},2.55 $ MP,KZZ,{MAT_BOND_DUMMY},2.55 $ MP,DENS,{MAT_BOND_DUMMY},2500 $ MP,C,{MAT_BOND_DUMMY},800")
    a("ET,1,SOLID70")
    a(f"ESIZE,{W}/{ND}")

    layers = [
        ("xpu0", "xpu1", MAT_SI),
        ("xpu1", "b1_1", MAT_BOND_ACTIVE),
        ("b1_1", "eic1", MAT_SI),
        ("eic1", "b2_1", MAT_BOND_ACTIVE),
        ("b2_1", "apic1", MAT_APIC),
        ("apic1", "b3_1", MAT_BOND_DUMMY),
        ("b3_1", "intp1", MAT_SI),
    ]
    for lo, hi, mat in layers:
        a(f"BLOCK,0,{W},0,{W},{z[lo]},{z[hi]} $ VSEL,S,LOC,Z,({z[lo]}+{z[hi]})/2 $ VATT,{mat},,1 $ VMESH,ALL $ ALLSEL")
    a("ALLSEL $ NUMMRG,NODE,1e-9 $ NUMCMP,NODE")

    vol_xpu = W * W * T_XPU
    a(f"VOL_XPU={vol_xpu}")
    # constant EIC / aPIC self-heating (Coenen power densities, Sec. II-B)
    vol_eic = W * W * T_EIC
    vol_apic = W * W * T_APIC
    a(f"ESEL,S,MAT,,{MAT_SI} $ ESEL,R,CENT,Z,({z['b1_1']}+{z['eic1']})/2 $ BFE,ALL,HGEN,1,{EIC_POWER_W}/{vol_eic} $ ALLSEL")
    a(f"ESEL,S,MAT,,{MAT_APIC} $ BFE,ALL,HGEN,1,{APIC_POWER_W}/{vol_apic} $ ALLSEL")

    # boundary conditions: top = liquid cold plate on XPU, bottom = weak path to package substrate
    a(f"ASEL,S,LOC,Z,{z['xpu0']} $ SFA,ALL,,CONV,{HCP_TOP},{T_COOLANT_C} $ ALLSEL")
    a(f"ASEL,S,LOC,Z,{z['intp1']} $ SFA,ALL,,CONV,{H_BOTTOM:.4f},{T_COOLANT_C} $ ALLSEL")

    # monitor node: aPIC layer center (where the MRR sits)
    apic_mid = f"({z['b2_1']}+{z['apic1']})/2"
    a(f"NMON=NODE({W}/2,{W}/2,{apic_mid})")
    a("FINISH")
    a("")
    a("/SOLU")
    a("ANTYPE,TRANS")
    a("TRNOPT,FULL")
    a("OUTRES,ALL,ALL")
    a("TIMINT,OFF")
    a(f"ESEL,S,MAT,,{MAT_SI} $ ESEL,R,CENT,Z,({z['xpu0']}+{z['xpu1']})/2 $ BFE,ALL,HGEN,1,{rows[0][2]*scale}/{vol_xpu} $ ALLSEL")
    a("TIME,1e-3 $ KBC,1 $ SOLVE")
    a("TIMINT,ON")
    prev_end = rows[0][0]
    for (t0, t1, p) in rows:
        p_scaled = p * scale
        a(f"ESEL,S,MAT,,{MAT_SI} $ ESEL,R,CENT,Z,({z['xpu0']}+{z['xpu1']})/2 $ BFE,ALL,HGEN,1,{p_scaled}/{vol_xpu} $ ALLSEL")
        dt = max(t1 - t0, 1e-4)
        # Fixed step, no AUTOTS: with automatic stepping the long all-to-all load steps were
        # solved at 14.6-175 ms steps, which quantized the derived stall windows to multiples
        # of the step. DT_MAX_S bounds the step everywhere; short compute bursts use dt/8.
        substep = min(dt / 8.0, DT_MAX_S)
        a(f"TIME,{t1} $ KBC,1 $ AUTOTS,OFF $ DELTIM,{substep} $ SOLVE")
    a("FINISH")
    a("")
    a("/POST26")
    a("NUMVAR,200")
    a("NSOL,2,NMON,TEMP")
    a(f"/OUTPUT,{out_csv},csv")
    a("PRVAR,2")
    a("/OUTPUT")
    a("FINISH")

    with open(out_inp, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_inp}  (peak XPU power {peak_w}W -> scaled x{scale:.3f} to {XPU_PEAK_TARGET_W}W; "
          f"{len(rows)} load steps; EIC={EIC_POWER_W:.1f}W aPIC={APIC_POWER_W:.1f}W const; "
          f"H_BOTTOM={H_BOTTOM:.1f} W/m2K)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: gen_pic_transient.py <power_schedule.csv> <out.inp> <out_csv_basename>")
        sys.exit(1)
    generate(sys.argv[1], sys.argv[2], sys.argv[3])
