#!/usr/bin/env python3
"""Per-device power schedules from the expert-routing weight matrix.

The baseline schedule (gen_power_schedule_fwdbwd.py) gives every device the mean
expert share. Real routing is skewed (Fig. 1 of the paper): a device hosting a hot
expert runs its expert GEMMs for load_factor x longer, a cold one shorter. The
load factor of expert j is (tokens routed to j) / (mean tokens per expert), taken
from the same ep x ep weight matrix fed to htsim (mixnet-htsim/test/wm_ep{E}.txt).

Emits three schedules per model (coldest / median / hottest device) plus the ANSYS
.inp for each, so the collective stall can be evaluated at the worst device.

  python3 gen_power_schedule_device.py   (run from thermal_pic/)
"""
import ast
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_power_schedule_fwdbwd import phase_sums, build_schedule, write_csv  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = {
    "mixtral8x7b": dict(graph="taskgraph/mixtral8x7B_paper_dp2tp4pp4_ep8top2_L4_seq4096_mb8_H100.txt",
                        n_layers=4, n_experts=8, wm="mixnet-htsim/test/wm_ep8.txt"),
    "qwen_moe_14.3b": dict(graph="taskgraph/qwenMoE_paper_dp2tp1pp4_ep64top4_L4_seq4096_mb8_H100.txt",
                           n_layers=4, n_experts=64, wm="mixnet-htsim/test/wm_ep64.txt"),
    "llama_moe_6.7b": dict(graph="taskgraph/llamaMoE_paper_dp2tp1pp4_ep16top2_L4_seq4096_mb8_H100.txt",
                           n_layers=4, n_experts=16, wm="mixnet-htsim/test/wm_ep16.txt"),
}


def load_factors(wm_path):
    rows = ast.literal_eval(open(wm_path).read().strip())
    n = len(rows)
    col = [sum(r[j] for r in rows) for j in range(n)]
    mean = sum(col) / n
    f = sorted(c / mean for c in col)
    return {"coldest": f[0], "median": f[n // 2], "hottest": f[-1]}


if __name__ == "__main__":
    for label, cfg in MODELS.items():
        durations = phase_sums(os.path.join(ROOT, cfg["graph"]), cfg["n_layers"], cfg["n_experts"])
        factors = load_factors(os.path.join(ROOT, cfg["wm"]))
        print(f"=== {label}: expert load factors {factors}")
        for cls, fac in factors.items():
            d = dict(durations)
            d["experts"] = (durations["experts"][0] * fac, durations["experts"][1] * fac)
            rows = build_schedule(f"{label}-{cls}", d)
            csv_out = f"{label}_power_schedule_fwdbwd_L4seq4096_{cls}.csv"
            write_csv(rows, csv_out)
            short = {"mixtral8x7b": "mixtral8x7b", "qwen_moe_14.3b": "qwenmoe", "llama_moe_6.7b": "llamamoe"}[label]
            subprocess.run([sys.executable, "gen_pic_transient.py", csv_out,
                            f"pic_transient_{short}_L4seq4096_{cls}_dt1ms.inp",
                            f"{short}_pic_transient_L4seq4096_{cls}_dt1ms"], check=True)
