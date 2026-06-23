# FlexFlow MoE task-graph (`.fbuf`) generation — scripts

These scripts generate **MoE training task graphs** (`.fbuf`) with FlexFlow's `moe` example.
The `.fbuf` files are the input (`-flowfile`) to **mixnet-htsim** for the packet-level fabric
study (glass-FB vs Fat-tree / MixNet / NVL72). Generation runs on **one GPU** (`-ll:gpu 1`);
the logical GPU layout is set purely by the parallelism flags.

| script | purpose |
|--------|---------|
| `profile_flexflow_models.sh` | **main** — interactive (apptainer) generator for the paper's models, memory-aware, with proxies. Use this. |
| `gen_fbuf_slurm.sh` | non-interactive SLURM batch variant (sbatch) for a 128/256/512-GPU sweep. |
| `gen_fbuf_models.sh` | older per-model loop (superseded by `profile_flexflow_models.sh`). |

---

## 0. Prerequisite: a built `moe` binary

`moe` must already be built for the target GPU. On PACE Phoenix (H100/H200 = Hopper sm_90):

```bash
module load cuda/11.7.0 cudnn/8.5.0.96-11.7-cuda
salloc -A gts-syu334-ece -qinferno -N1 --gres=gpu:H100:1 -t2:00:00
apptainer exec --cleanenv --nv /storage/project/r-syu334-0/shared/images/flexflow-cuda-11.8.sif bash
cd <FF>/mixnet-flexflow                      # FF repo root
cmake -S . -B build_h100 -DFF_GPU_BACKEND=cuda -DFF_CUDA_ARCH=90 \
      -DCMAKE_CUDA_ARCHITECTURES=90 -DFF_MAX_DIM=5     # CLEAN dir; FF_MAX_DIM=5 is required
cmake --build build_h100 -j8 --target moe             # ~1-2 h (Legion full build)
```
- **`FF_MAX_DIM=5`** is mandatory (moe uses 5-D tensors; default 4 -> link error
  `undefined reference TensorAccessorW<...,5>`). Must be a **fresh** build dir
  (`Legion_MAX_DIM` is `CACHE STRING` and won't update on in-place reconfigure).
- H100 & H200 are both **sm_90** -> one `arch 90` build serves both.
- If a fresh `scratch` checkout is missing files (60-day purge), restore with
  `git checkout -- .`; flatbuffers headers must be **v24.3.25** under `fbuf/include/`,
  spdlog under `spdlog/include/` (clone `gabime/spdlog v1.11.0`); add `#include <cstdint>`
  to `include/flexflow/utils/hash_utils.h`; add `#!/usr/bin/env python3` + `chmod +x` to
  `deps/legion/bindings/python/setup.py`.

---

## 1. Allocate resources (PACE Phoenix limits: 8 cores/node, 128 GB/core -> **1 TB max/node**)

`moe`'s host RSS scales with the task-graph size and is **cgroup-OOM-killed** (`EXIT=137`,
`dmesg`: "Memory cgroup out of memory: Killed process moe") if it exceeds the requested memory.
`free -g` shows the **node** total (2 TB), not your job's cgroup limit — don't trust it.

Request the most memory the model needs (interactive desktop or salloc):

| model | experts | logical GPUs (dp1 tp1) | est. host memory | fits 1 TB node? |
|-------|---------|------------------------|------------------|-----------------|
| mixtral8x22B (validation) | 8 | 128 | ~130 GB | yes |
| qwen3_235B | 128 | 128 | ~0.3–0.6 TB | yes |
| deepseekV3 | 256 | 256 | ~0.6–1.2 TB | edge (use levers) |
| kimiK2 | 384 | 384 | ~1.5–2 TB | **no -> use `kimi_proxy64`** |

Recommended request: **1 GPU (H100), 8 cores, 128 GB/core (= 1 TB), 2 h.**

---

## 2. Run

Inside the apptainer shell:

```bash
cd <FF>/mixnet-flexflow            # or wherever; script cd's to $FF itself

# validate the toolchain first (proven mixtral8x22B config)
bash <repo>/scripts/profile_flexflow_models.sh

# one or more target models
bash <repo>/scripts/profile_flexflow_models.sh qwen3_235B
bash <repo>/scripts/profile_flexflow_models.sh deepseekV3
bash <repo>/scripts/profile_flexflow_models.sh kimi_proxy64     # Kimi > 1 TB -> reduced-expert proxy

# comm-intensity sweep (more memory per run)
MBS="8 16 32" bash <repo>/scripts/profile_flexflow_models.sh qwen3_235B

# preview the exact command without a GPU
DRYRUN=1 bash <repo>/scripts/profile_flexflow_models.sh deepseekV3
```

Known model labels: `mixtral8x22B qwen3_235B deepseekV3 kimiK2 deepseek_proxy64 kimi_proxy64`.

### What the script bakes in (and why)
- **Memory levers**: `--train_dp 1 --num-layers 1 --budget 1`, 1 micro-batch, `seq 1024`.
  (`-ll:csize` does **not** cap the blow-up — it's FlexFlow graph/simulator malloc, not a Legion pool.)
- **Low TP (`tp 1`)** on the targets -> the workload is **a2a-dominant**, which is what reveals
  the fabric difference (and is fair to MixNet; TP-heavy = all-reduce-dominant hides it).
- **Logical GPUs = dp × tp × pp × ep** (ep = experts). e.g. deepseekV3 dp1·tp1·pp1·256 = 256.
- Output goes to `<FF>/results/<tag>.fbuf` (+ `.txt`). On failure it prints the OOM line and a hint.

### Tunable env vars
`MBS` (microbatch list, default `8`), `NLAYERS` (1), `SEQ` (1024), `BUDGET` (1),
`FSIZE`/`ZSIZE`/`SIMWS` (Legion `-ll:fsize`/`-ll:zsize`/`--simulator-workspace-size`),
`FF`, `MOE`, `OUT`, `DRYRUN`.

---

## 3. Troubleshooting

- **`EXIT=137` / "Killed"** = cgroup OOM. Check actual peak: `dmesg | grep -i "killed process" | tail`.
  Mitigate: request more memory (≤ 1 TB/node), or use `*_proxy64`, or lower `--batch-size`.
- **Kimi/DeepSeek won't fit 1 TB** even with levers -> use `deepseek_proxy64` / `kimi_proxy64`
  (64 experts, same hidden/topk). They preserve the a2a-dominant regime at ~1/4–1/6 memory;
  cite as "DeepSeek/Kimi-scaled MoE" in the paper.
- **Binary won't run (no kernel image)** -> you're not on a Hopper GPU; `moe` is sm_90. Check `nvidia-smi -L`.

---

## 4. Next step

Feed the generated `.fbuf` to mixnet-htsim and compare fabrics (glass-FB / Fat-tree / MixNet /
NVL72), normalized to non-blocking Fat-tree (no infinite-BW "ideal"), per the MixNet-paper method.
