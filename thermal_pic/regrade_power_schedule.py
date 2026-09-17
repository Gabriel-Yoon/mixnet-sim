#!/usr/bin/env python3
"""Replace the manually-guessed per-phase GPU power values in a workload power
schedule (thermal_schedule_three_models/*_power_schedule_s.csv) with values
derived from each phase's COMPUTE REGIME, instead of a flat linear rescale of
the original (unsourced) guesses.

Rationale (documented, not measured -- no real GPU power trace is available
for this project; see attention_analytical_cost.md for the FLOP/timing model
this reasoning is anchored to):

  attention, experts   : large, well-parallelized GEMMs (H=4096+ dims) that
                          saturate the SM array -> GPU runs at TDP.
                          TDP = 700 W (Coenen et al. TCPMT 2026 baseline XPU
                          input power).
  gate                  : thin GEMM ([S x H] -> [S x num_experts], num_experts
                          << H) -- output dimension too small to fill the SM
                          array, so achieved utilization is well below peak
                          even though the kernel is compute-type. Assigned a
                          "medium utilization" fraction.
  alltoall_*            : pure network communication; no GEMM/tensor-core
                          kernel is running on the GPU during this phase, only
                          the NIC/memory-copy engines are active -> GPU
                          compute cores are effectively idle. Assigned a
                          board-idle-power fraction (typical for high-TDP
                          datacenter GPUs, which still draw a baseline idle
                          power).
  addnorm               : elementwise layernorm + residual add; HBM-bandwidth
                          bound, not tensor-core bound -> low-to-moderate
                          power (memory subsystem active, compute mostly not).

Fractions of TDP (documented assumption, not measurement):
  GEMM-saturated (attention/experts): 100%
  thin GEMM (gate):                    40%
  idle/comm-bound (alltoall):          12%
  memory-bound elementwise (addnorm):  20%
"""
import csv
import sys

TDP_W = 700.0
PHASE_FRACTION = {
    "attention": 1.00,
    "experts": 1.00,
    "gate": 0.40,
    "addnorm": 0.20,
    # alltoall_1 / alltoall_2 handled by prefix match below
}
ALLTOALL_FRACTION = 0.12


def phase_power(phase):
    if phase in PHASE_FRACTION:
        return TDP_W * PHASE_FRACTION[phase]
    if phase.startswith("alltoall"):
        return TDP_W * ALLTOALL_FRACTION
    raise ValueError(f"unrecognized phase: {phase}")


def regrade(in_csv, out_csv):
    with open(in_csv) as f, open(out_csv, "w", newline="") as g:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(g, fieldnames=reader.fieldnames)
        writer.writeheader()
        counts = {}
        for row in reader:
            old_w = float(row["power_w"])
            new_w = phase_power(row["phase"])
            row["power_w"] = f"{new_w:.4f}"
            writer.writerow(row)
            counts.setdefault(row["phase"], (old_w, new_w))
        print(f"wrote {out_csv}")
        for phase, (old_w, new_w) in sorted(counts.items()):
            print(f"  {phase:>12}: {old_w:>6.1f} W (old, manual guess)  ->  {new_w:>6.1f} W (new, compute-regime)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: regrade_power_schedule.py <in_power_schedule_s.csv> <out.csv>")
        sys.exit(1)
    regrade(sys.argv[1], sys.argv[2])
