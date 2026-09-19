"""Job C part 2: synthetic burst/idle GEMM patterns on one GPU with 20 ms NVML telemetry.

Busy = back-to-back fp32 GEMMs (A 4096x4096 @ B 4096x16384, TF32 off) with a
synchronize after each, until the busy budget is used. Idle = sleep.
Writes telemetry.csv and phases.csv (pattern,cycle,phase,start_s,end_s) to OUT.
"""
import os
import sys
import time

import torch

from c_telemetry import Sampler

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device("cuda:0")
A = torch.randn(4096, 4096, device=dev, dtype=torch.float32)
B = torch.randn(4096, 16384, device=dev, dtype=torch.float32)

PATTERNS = [  # name, busy_s, idle_s, cycles
    ("a_35on_350off", 0.035, 0.350, 20),
    ("b_35on_35off", 0.035, 0.035, 20),
    ("c_300on_350off", 0.300, 0.350, 20),
    ("d_60son_60soff", 60.0, 60.0, 3),
]
SETTLE_S = 60.0

phases = open(os.path.join(OUT, "phases.csv"), "w")
phases.write("pattern,cycle,phase,start_s,end_s,n_gemm\n")


def busy(seconds):
    torch.cuda.synchronize()
    t0 = time.time()
    n = 0
    while time.time() - t0 < seconds:
        torch.mm(A, B)
        torch.cuda.synchronize()
        n += 1
    return t0, time.time(), n


def idle(seconds):
    t0 = time.time()
    time.sleep(seconds)
    return t0, time.time()


# warm up kernels before telemetry starts
for _ in range(5):
    torch.mm(A, B)
torch.cuda.synchronize()
t0 = time.time()
torch.mm(A, B)
torch.cuda.synchronize()
print(f"single GEMM {1e3 * (time.time() - t0):.2f} ms", flush=True)

sampler = Sampler(os.path.join(OUT, "telemetry.csv"), period_s=0.02)
sampler.start()
s, e = idle(SETTLE_S)
phases.write(f"baseline,0,idle,{s:.4f},{e:.4f},0\n")
for name, b_s, i_s, cycles in PATTERNS:
    for c in range(cycles):
        s, e, n = busy(b_s)
        phases.write(f"{name},{c},busy,{s:.4f},{e:.4f},{n}\n")
        s, e = idle(i_s)
        phases.write(f"{name},{c},idle,{s:.4f},{e:.4f},0\n")
    s, e = idle(SETTLE_S)
    phases.write(f"{name},{cycles},settle,{s:.4f},{e:.4f},0\n")
    phases.flush()
    print(f"done {name}", flush=True)
sampler.stop()
phases.close()
print(f"telemetry samples: {sampler.n_samples}, memory temp field: {sampler.mem_temp_ok}", flush=True)
