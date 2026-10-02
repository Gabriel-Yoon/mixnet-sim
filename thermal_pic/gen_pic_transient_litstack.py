#!/usr/bin/env python3
"""Literature-stack variant of gen_pic_transient.py (Job L).

Same solver, mesh rule, boundary conditions, time stepping and schedule handling as
gen_pic_transient.py; ONLY the layer stack and the material constants change, to follow
the published OIO3D stack instead of the assumed one. The original script is untouched.

What changed and why (source checks behind Job L):
  1. Coenen et al., JSTQE 32(2) 2026: in OIO3D the aPIC and EIC silicon SUBSTRATES are
     REMOVED -- only BEOL/FEOL, the device layer and the BOX remain. The original kept
     EIC 100 um and aPIC 50 um of silicon.
  2. A homogeneous aPIC at k = 80 W/mK is not defensible: the 2 um BOX and the BEOL oxide
     (k = 1.38 W/mK) dominate vertical conduction. The aPIC is now resolved into
     BEOL oxide / Si device layer / BOX, and the aPIC self-heating and the monitor node
     both sit in the 0.22 um Si device layer, where the MRR actually is.
  3. Si k at the 330 K operating point is about 130 W/mK (Ho, Powell, Liley 1972), not 150.
     Si c is 712 J/kgK.
  4. XPU 775 um is an unthinned 300 mm wafer; the photonic interposer is a 100 um TSV
     interposer. Both are the literature values.
  The EIC is now an orthotropic FEOL+BEOL block (KZZ 5.0, KXX = KYY 20 W/mK): vertical
  conduction is oxide-limited while the metal stack spreads heat laterally.

Stack, top to bottom, all 25 x 25 mm:
   1 XPU Si                775   um   k 130   rho 2330  c 712   <- time-varying schedule, peak 700 W
   2 hybrid bond           2.4   um   k 5.0   rho 2500  c 800
   3 EIC FEOL+BEOL          10   um   KZZ 5.0 / KXX = KYY 20, rho 2500, c 800  <- 0.426 W/mm2 const
   4 hybrid bond           2.4   um   k 5.0   rho 2500  c 800
   5 aPIC BEOL oxide         3   um   k 1.38  rho 2220  c 745
   6 aPIC Si device layer  0.22  um   k 60    rho 2330  c 712   <- 0.113 W/mm2 const, MONITOR NODE
   7 aPIC BOX                2   um   k 1.38  rho 2220  c 745
   8 D2W bond              2.4   um   k 2.55  rho 2500  c 800
   9 passive PIC oxide/SiN   3   um   k 1.38  rho 2220  c 745
  10 interposer Si         100   um   k 130   rho 2330  c 712

Unchanged from the original: SOLID70, ESIZE W/20 laterally (one element per thin block),
top HTC 7e4 W/m2K at 45 C on the XPU top, bottom h = 1/(0.1 K/W * tile area) = 16000 W/m2K
at 45 C, DT_MAX_S = 1e-3, KBC,1 with AUTOTS off, the TIMINT,OFF steady priming step, and the
peak-rescaling of the XPU schedule to 700 W.

usage: gen_pic_transient_litstack.py <power_schedule.csv> <out.inp> <out_csv_basename>
       [--device-um <thickness>]   (fallback if ANSYS rejects the 0.22 um block)
"""
import csv
import sys

MAT_SI = 1          # XPU and interposer: bulk silicon at 330 K
MAT_BOND_ACTIVE = 2  # hybrid bond, via-populated (XPU-EIC, EIC-aPIC)
MAT_APIC_SI = 3     # aPIC silicon device layer (thin, partially etched -> reduced k)
MAT_BOND_DUMMY = 4  # D2W bond, no dense vias (aPIC - passive interposer)
MAT_OXIDE = 5       # BEOL oxide, BOX, passive PIC oxide/SiN
MAT_EIC = 6         # EIC FEOL+BEOL, orthotropic

W = 0.025           # 25x25 mm OIO3D tile footprint
ND = 20             # lateral elements/side (unchanged mesh rule)

T_XPU = 775e-6      # unthinned 300 mm wafer
T_BOND = 2.4e-6     # hybrid/D2W bond thickness (Coenen Table I)
T_EIC = 10e-6       # FEOL + BEOL only, substrate removed
T_BEOL_OX = 3e-6    # aPIC BEOL oxide
T_DEV_DEFAULT = 0.22e-6   # aPIC Si device layer
T_BOX = 2e-6        # aPIC buried oxide
T_PPIC = 3e-6       # passive PIC oxide/SiN
T_INTP = 100e-6     # TSV photonic interposer

TILE_AREA_MM2 = 25.0 * 25.0
EIC_POWER_W = 0.426 * TILE_AREA_MM2    # 266.25 W, constant
APIC_POWER_W = 0.113 * TILE_AREA_MM2   # 70.625 W, constant

HCP_TOP = 70000.0
R_BOTTOM_K_PER_W = 0.1
TILE_AREA_M2 = (25e-3) * (25e-3)
H_BOTTOM = 1.0 / (R_BOTTOM_K_PER_W * TILE_AREA_M2)   # ~16000 W/m^2K

T_COOLANT_C = 45.0
XPU_PEAK_TARGET_W = 700.0
DT_MAX_S = 1e-3


def load_schedule(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["start_s"]), float(r["end_s"]), float(r["power_w"])))
    return rows


def generate(csv_path, out_inp, out_csv, t_dev=T_DEV_DEFAULT):
    rows = load_schedule(csv_path)
    peak_w = max(p for _, _, p in rows)
    scale = XPU_PEAK_TARGET_W / peak_w

    z = {"xpu0": 0.0}
    z["xpu1"] = z["xpu0"] + T_XPU
    z["b1_1"] = z["xpu1"] + T_BOND
    z["eic1"] = z["b1_1"] + T_EIC
    z["b2_1"] = z["eic1"] + T_BOND
    z["beol1"] = z["b2_1"] + T_BEOL_OX
    z["dev1"] = z["beol1"] + t_dev
    z["box1"] = z["dev1"] + T_BOX
    z["b3_1"] = z["box1"] + T_BOND
    z["ppic1"] = z["b3_1"] + T_PPIC
    z["intp1"] = z["ppic1"] + T_INTP

    lines = []
    a = lines.append
    a("/PREP7")
    a("! ===== JETC Job L: literature OIO3D stack (substrates removed, resolved aPIC) =====")
    a(f"MP,KXX,{MAT_SI},130.0 $ MP,KYY,{MAT_SI},130.0 $ MP,KZZ,{MAT_SI},130.0 $ MP,DENS,{MAT_SI},2330 $ MP,C,{MAT_SI},712")
    a(f"MP,KXX,{MAT_BOND_ACTIVE},5.0 $ MP,KYY,{MAT_BOND_ACTIVE},5.0 $ MP,KZZ,{MAT_BOND_ACTIVE},5.0 $ MP,DENS,{MAT_BOND_ACTIVE},2500 $ MP,C,{MAT_BOND_ACTIVE},800")
    a(f"MP,KXX,{MAT_APIC_SI},60.0 $ MP,KYY,{MAT_APIC_SI},60.0 $ MP,KZZ,{MAT_APIC_SI},60.0 $ MP,DENS,{MAT_APIC_SI},2330 $ MP,C,{MAT_APIC_SI},712")
    a(f"MP,KXX,{MAT_BOND_DUMMY},2.55 $ MP,KYY,{MAT_BOND_DUMMY},2.55 $ MP,KZZ,{MAT_BOND_DUMMY},2.55 $ MP,DENS,{MAT_BOND_DUMMY},2500 $ MP,C,{MAT_BOND_DUMMY},800")
    a(f"MP,KXX,{MAT_OXIDE},1.38 $ MP,KYY,{MAT_OXIDE},1.38 $ MP,KZZ,{MAT_OXIDE},1.38 $ MP,DENS,{MAT_OXIDE},2220 $ MP,C,{MAT_OXIDE},745")
    a(f"MP,KXX,{MAT_EIC},20.0 $ MP,KYY,{MAT_EIC},20.0 $ MP,KZZ,{MAT_EIC},5.0 $ MP,DENS,{MAT_EIC},2500 $ MP,C,{MAT_EIC},800")
    a("ET,1,SOLID70")
    a(f"ESIZE,{W}/{ND}")

    layers = [
        ("xpu0", "xpu1", MAT_SI),            # 1 XPU
        ("xpu1", "b1_1", MAT_BOND_ACTIVE),   # 2 hybrid bond
        ("b1_1", "eic1", MAT_EIC),           # 3 EIC FEOL+BEOL
        ("eic1", "b2_1", MAT_BOND_ACTIVE),   # 4 hybrid bond
        ("b2_1", "beol1", MAT_OXIDE),        # 5 aPIC BEOL oxide
        ("beol1", "dev1", MAT_APIC_SI),      # 6 aPIC Si device layer
        ("dev1", "box1", MAT_OXIDE),         # 7 aPIC BOX
        ("box1", "b3_1", MAT_BOND_DUMMY),    # 8 D2W bond
        ("b3_1", "ppic1", MAT_OXIDE),        # 9 passive PIC oxide/SiN
        ("ppic1", "intp1", MAT_SI),          # 10 interposer
    ]
    for lo, hi, mat in layers:
        a(f"BLOCK,0,{W},0,{W},{z[lo]},{z[hi]} $ VSEL,S,LOC,Z,({z[lo]}+{z[hi]})/2 $ VATT,{mat},,1 $ VMESH,ALL $ ALLSEL")
    a("ALLSEL $ NUMMRG,NODE,1e-9 $ NUMCMP,NODE")

    vol_xpu = W * W * T_XPU
    vol_eic = W * W * T_EIC
    vol_dev = W * W * t_dev
    a(f"VOL_XPU={vol_xpu}")
    # Constant EIC and aPIC self-heating. Each now has its own material id, so no z-filter
    # is needed (the XPU and the interposer still share MAT_SI and are filtered by z).
    a(f"ESEL,S,MAT,,{MAT_EIC} $ BFE,ALL,HGEN,1,{EIC_POWER_W}/{vol_eic} $ ALLSEL")
    a(f"ESEL,S,MAT,,{MAT_APIC_SI} $ BFE,ALL,HGEN,1,{APIC_POWER_W}/{vol_dev} $ ALLSEL")

    # boundary conditions (unchanged): top cold plate on the XPU, weak path at the bottom
    a(f"ASEL,S,LOC,Z,{z['xpu0']} $ SFA,ALL,,CONV,{HCP_TOP},{T_COOLANT_C} $ ALLSEL")
    a(f"ASEL,S,LOC,Z,{z['intp1']} $ SFA,ALL,,CONV,{H_BOTTOM:.4f},{T_COOLANT_C} $ ALLSEL")

    # monitor node: centre of the aPIC Si device layer, where the MRR sits
    dev_mid = f"({z['beol1']}+{z['dev1']})/2"
    a(f"NMON=NODE({W}/2,{W}/2,{dev_mid})")
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
    for (t0, t1, p) in rows:
        p_scaled = p * scale
        a(f"ESEL,S,MAT,,{MAT_SI} $ ESEL,R,CENT,Z,({z['xpu0']}+{z['xpu1']})/2 $ BFE,ALL,HGEN,1,{p_scaled}/{vol_xpu} $ ALLSEL")
        dt = max(t1 - t0, 1e-4)
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
    print(f"wrote {out_inp}  (peak XPU {peak_w}W -> x{scale:.3f} to {XPU_PEAK_TARGET_W}W; "
          f"{len(rows)} load steps; EIC={EIC_POWER_W:.1f}W in a {T_EIC*1e6:.0f}um orthotropic block; "
          f"aPIC={APIC_POWER_W:.1f}W in a {t_dev*1e6:.3f}um Si device layer; "
          f"H_BOTTOM={H_BOTTOM:.1f} W/m2K; total stack {z['intp1']*1e6:.2f}um)")


if __name__ == "__main__":
    args = [x for x in sys.argv[1:]]
    t_dev = T_DEV_DEFAULT
    if "--device-um" in args:
        i = args.index("--device-um")
        t_dev = float(args[i + 1]) * 1e-6
        del args[i:i + 2]
    if len(args) != 3:
        print("usage: gen_pic_transient_litstack.py <power_schedule.csv> <out.inp> <out_csv_basename> "
              "[--device-um <thickness>]")
        sys.exit(1)
    generate(args[0], args[1], args[2], t_dev=t_dev)
