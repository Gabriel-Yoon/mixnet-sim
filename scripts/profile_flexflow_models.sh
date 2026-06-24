#!/bin/bash
# Generate MoE training task-graph .fbuf files (FlexFlow `moe`) for the glass-FB fabric study.
# Run INSIDE the apptainer container, AFTER a GPU allocation. Example:
#
#   salloc -A gts-syu334-ece -qinferno -N1 --gres=gpu:H100:1 -t2:00:00   # PACE: 8 cores x 128GB/core = 1TB max/node
#   apptainer exec --cleanenv --nv /storage/project/r-syu334-0/shared/images/flexflow-cuda-11.8.sif bash
#   bash scripts/profile_flexflow_models.sh                 # default: validate with mixtral8x22B
#   bash scripts/profile_flexflow_models.sh qwen3_235B      # one model
#   bash scripts/profile_flexflow_models.sh qwen3_235B deepseekV3
#   DRYRUN=1 bash scripts/profile_flexflow_models.sh deepseekV3   # print command only (no GPU needed)
#   MBS="8 16 32" bash scripts/profile_flexflow_models.sh qwen3_235B   # comm-intensity sweep
#
# WHY these choices (see notes):
#  - moe host RSS scales with task-graph size and OOMs the cgroup (EXIT=137). PACE caps a node at
#    8 cores x 128 GB/core = 1 TB. Estimated need: mixtral8x22B(8e)~130GB, qwen3(128e)~0.5TB,
#    deepseek(256e)~1TB(edge), kimi(384e)>1TB -> Kimi needs a *_proxy64 (reduced experts) on 1 node.
#  - Memory levers baked in: --train_dp 1, --num-layers 1, --budget 1, 1 micro-batch, seq 1024.
#    (-ll:csize does NOT cap the blow-up; it is FlexFlow graph/simulator malloc, not a Legion pool.)
#  - Targets use LOW TP (tp1) so the workload is a2a-DOMINANT -> reveals the fabric difference and
#    is fair to MixNet (TP-heavy = all-reduce-dominant hides fabric). GPU count = dp*tp*pp*ep.

set -uo pipefail
FF="${FF:-/storage/home/hcoda1/8/syoon351/scratch/repos/mixnet-sim/mixnet-flexflow}"
MOE="${MOE:-$FF/build_h100/examples/cpp/mixture_of_experts/moe}"
OUT="${OUT:-$FF/results}"; mkdir -p "$OUT"
NLAYERS="${NLAYERS:-1}"; SEQ="${SEQ:-1024}"; BUDGET="${BUDGET:-1}"
FSIZE="${FSIZE:-70000}"; ZSIZE="${ZSIZE:-24000}"; SIMWS="${SIMWS:-12000000000}"  # H100 80GB
DRYRUN="${DRYRUN:-0}"
read -r -a MB_ARR <<< "${MBS:-8}"   # microbatch sizes; "8 16 32" for a comm-intensity sweep
# auto-label the fbuf by the GPU it is generated on (compute cost model is GPU-specific;
# same moe binary runs on H100 & H200, so the filename must record which). Override: GPU_TAG=...
GPU_TAG="${GPU_TAG:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -oiE 'H100|H200|B200|A100|V100' | head -1)}"
[ -z "$GPU_TAG" ] && GPU_TAG="gpuUNK"

# model spec (portable, no associative arrays):  "hidden heads expert_hidden experts topk dp tp pp batch"
KNOWN="mixtral8x22B qwen3_235B deepseekV3 kimiK2 deepseek_proxy64 kimi_proxy64"
spec_for(){
  case "$1" in
    mixtral8x22B)     echo "6144 32 16384 8 2 2 8 1 128" ;; # validation = the proven config (tp8)
    qwen3_235B)       echo "4096 64 1536 128 8 1 1 1 8" ;;  # 128 experts -> 128 logical GPU
    deepseekV3)       echo "7168 128 2048 256 8 1 1 1 8" ;; # 256 experts -> 256 logical GPU (~1TB edge)
    kimiK2)           echo "7168 64 2048 384 8 1 1 1 8" ;;  # 384 experts -> 384 logical GPU (>1TB: use proxy)
    deepseek_proxy64) echo "7168 128 2048 64 8 1 1 1 8" ;;  # reduced-expert proxy, a2a-dominant, ~1/4 mem
    kimi_proxy64)     echo "7168 64 2048 64 8 1 1 1 8" ;;
    *) echo "" ;;
  esac
}

run_model(){
  local label="$1"; local s; s="$(spec_for "$label")"
  [ -z "$s" ] && { echo "!! unknown model '$label'. known: $KNOWN"; return 1; }
  read -r hidden heads ehid experts topk dp tp pp batch <<< "$s"
  local gpus=$((dp*tp*pp*experts))
  # batch MUST be >= #devices: --only-data-parallel splits the batch across all devices, so each
  # device needs >=1 sample (else sub_query.dims[2]=0 -> cuDNN BAD_PARAM / zero-cost measures).
  # Use batch = microbatch = #devices: 1 micro-batch, 1 sample/device (minimal memory, valid).
  batch=$gpus
  local mb
  for mb in "$gpus"; do
    local tag="${label}_dp${dp}tp${tp}pp${pp}_ep${experts}_mb${mb}_${NLAYERS}L_seq${SEQ}_${GPU_TAG}"
    echo "=================================================================="
    echo "[$label] $tag   logical GPUs = dp*tp*pp*ep = $gpus"
    echo "=================================================================="
    local cmd=( "$MOE" -ll:gpu 1 -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE"
        --budget "$BUDGET" --only-data-parallel --simulator-workspace-size "$SIMWS"
        --batch-size "$batch" --microbatchsize "$mb"
        --train_dp "$dp" --train_tp "$tp" --train_pp "$pp"
        --num-layers "$NLAYERS" --embedding-size "$hidden" --expert-hidden-size "$ehid"
        --hidden-size "$hidden" --num-heads "$heads" --sequence-length "$SEQ"
        --topk "$topk" --expnum "$experts" )
    if [ "$DRYRUN" = "1" ]; then printf '  %q' "${cmd[@]}"; echo; continue; fi
    "${cmd[@]}"; local rc=$?
    # moe hardcodes the export path to ./results/taskgraph.fbuf (graph.cc) and ./output.txt
    local fb="results/taskgraph.fbuf"
    if [ $rc -eq 0 ] && [ -f "$fb" ]; then
      mv "$fb" "$OUT/${tag}.fbuf"
      [ -f output.txt ] && mv output.txt "$OUT/${tag}.txt"
      echo "OK -> $OUT/${tag}.fbuf  ($(du -h "$OUT/${tag}.fbuf" 2>/dev/null | cut -f1))"
    elif [ $rc -eq 137 ]; then
      echo "FAILED rc=137 = cgroup OOM:"; dmesg 2>/dev/null | grep -i "killed process.*moe" | tail -1
      echo "  -> mitigate: use ${label}_proxy64, lower --batch-size, or request more mem (<=1TB/node)"
    else
      echo "FAILED rc=$rc, no $fb (moe exited without exporting). Check moe output above."
    fi
  done
}

models=("$@"); [ ${#models[@]} -eq 0 ] && models=(mixtral8x22B)
cd "$FF" || { echo "FF dir not found: $FF"; exit 1; }
echo "moe=$MOE  out=$OUT  layers=$NLAYERS seq=$SEQ budget=$BUDGET mbs=[${MB_ARR[*]}] dryrun=$DRYRUN"
for m in "${models[@]}"; do run_model "$m"; done
echo "DONE. fbufs -> $OUT"
