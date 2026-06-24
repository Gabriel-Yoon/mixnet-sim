#!/bin/bash
# Fully-measured (NO analytical) mixtral8x22B microbatch sweep — comm-intensity anchor.
# Run INSIDE the apptainer container (H200 or H100), after building moe with FF_FULLMEASURE support.
#   FSIZE=120000 bash scripts/profile_mixtral_sweep.sh           # H200 (default mb 8 16 32 64)
#   FSIZE=70000  bash scripts/profile_mixtral_sweep.sh           # H100
#   MBS="16 32"  FSIZE=120000 bash scripts/profile_mixtral_sweep.sh
#
# WHY this is fully measurable (unlike qwen3/deepseek/kimi): mixtral8x22B = 8 experts (MoE-routing
# ops measure without crash) AND tp8 -> per-device hidden = 6144/8 = 768 (small enough for cuDNN
# multi-head attention measurement). FF_FULLMEASURE=1 forces measure_operator_cost for ALL ops
# (attention + group_by/aggregate/topk), i.e. no analytical estimate anywhere. This is the proven
# config from the original HPC run. Note: tp8 => allreduce-dominant (conservative anchor regime).
set -uo pipefail
FF="${FF:-/storage/home/hcoda1/8/syoon351/scratch/repos/mixnet-sim/mixnet-flexflow}"
MOE="${MOE:-$FF/build_h100/examples/cpp/mixture_of_experts/moe}"
OUT="${OUT:-$FF/results}"; mkdir -p "$OUT"
FSIZE="${FSIZE:-120000}"   # H200 141GB; use 70000 on H100 (80GB)
GPU_TAG="${GPU_TAG:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -oiE 'H100|H200|B200' | head -1)}"
[ -z "$GPU_TAG" ] && GPU_TAG=gpuUNK
read -r -a MB_ARR <<< "${MBS:-8 16 32 64}"

cd "$FF"
for mb in "${MB_ARR[@]}"; do
  tag="mixtral8x22B_fullmeasure_dp2tp8pp1ep8_mb${mb}_1L_seq1024_${GPU_TAG}"
  echo "=================================================================="
  echo "[mixtral8x22B full-measure] $tag"
  echo "=================================================================="
  FF_FULLMEASURE=1 "$MOE" -ll:gpu 1 -ll:fsize "$FSIZE" -ll:zsize 24000 \
    --budget 1 --only-data-parallel --simulator-workspace-size 12000000000 \
    --batch-size 128 --microbatchsize "$mb" \
    --train_dp 2 --train_tp 8 --train_pp 1 \
    --num-layers 1 --embedding-size 1024 --expert-hidden-size 16384 \
    --hidden-size 6144 --num-heads 32 --sequence-length 1024 --topk 2 --expnum 8
  rc=$?
  if [ $rc -eq 0 ] && [ -f results/taskgraph.fbuf ]; then
    mv results/taskgraph.fbuf "$OUT/${tag}.fbuf"
    [ -f output.txt ] && mv output.txt "$OUT/${tag}.txt"
    echo "OK -> $OUT/${tag}.fbuf  ($(du -h "$OUT/${tag}.fbuf" 2>/dev/null | cut -f1))"
  else
    echo "FAILED rc=$rc (137=OOM). lower FSIZE or check moe output above."
  fi
done
echo "DONE -> $OUT"
