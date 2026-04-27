#!/usr/bin/env bash
# =====================================================================
# ETS on CD-FSOD Benchmark - Main Results Reproduction Script
# ---------------------------------------------------------------------
# Mirrors CDFSOD-benchmark/main_results.sh: trains and evaluates ETS on
# all 6 standard target datasets (ArTaxOr, Clipart1k, DIOR, FISH,
# NEU-DET, UODD) under 1/5/10-shot settings, optionally for both
# variants:
#   - baseline : ETS without ETS-specific mix augmentation
#                (matches benchmark's training regime; for fair compare)
#   - ets      : full ETS pipeline (Mosaic + MixUp + YOLOXHSV + ...)
#
# Usage examples:
#   bash main_results_cdfsod.sh                    # both variants, 4 GPUs
#   bash main_results_cdfsod.sh baseline           # only baseline
#   bash main_results_cdfsod.sh ets                # only ets variant
#   GPUS=8 GPU_IDS=0,1,2,3,4,5,6,7 bash main_results_cdfsod.sh
#
# Each experiment writes to:
#   mmdetection/work_dirs/cdfsod/<variant>/<dataset>_<shot>shot/
# Results are aggregated by `python summarize_results.py`.
# =====================================================================
set -e

# ---- configurable env vars ----
GPUS="${GPUS:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PORT="${PORT:-29508}"
DATASETS="${DATASETS:-artaxor dior fish clipart1k neu-det uodd}"
SHOTS="${SHOTS:-1 5 10}"

# Variants to run (positional args, default = both)
if [ "$#" -eq 0 ]; then
    VARIANTS=(baseline ets)
else
    VARIANTS=("$@")
fi

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${REPO_ROOT}/mmdetection"

echo "============================================"
echo " ETS @ CD-FSOD Benchmark - reproduction run"
echo " GPUs           : ${GPUS} (CUDA_VISIBLE_DEVICES=${GPU_IDS})"
echo " Datasets       : ${DATASETS}"
echo " Shots          : ${SHOTS}"
echo " Variants       : ${VARIANTS[*]}"
echo " Started        : $(date)"
echo "============================================"

run_one() {
    local variant=$1
    local dataset=$2
    local shot=$3
    local cfg="../configs/cdfsod/${dataset}/${shot}shot_${variant}.py"
    local outdir="work_dirs/cdfsod/${variant}/${dataset}_${shot}shot"

    if [ ! -f "$cfg" ]; then
        echo "[skip] missing config: $cfg"
        return
    fi

    echo ""
    echo ">>> [${variant}] ${dataset} ${shot}-shot  $(date '+%H:%M:%S')"
    mkdir -p "$outdir"

    CUDA_VISIBLE_DEVICES=${GPU_IDS} \
    PYTHONPATH="$(pwd)":${PYTHONPATH:-} \
    python -m torch.distributed.launch \
        --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
        --nproc_per_node=${GPUS} --master_port=${PORT} \
        tools/train.py "$cfg" \
        --work-dir "$outdir" \
        --launcher pytorch \
      || echo "WARNING: ${variant}/${dataset}/${shot}shot FAILED"

    # Light cleanup between runs
    sleep 3
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
}

for variant in "${VARIANTS[@]}"; do
    for dataset in ${DATASETS}; do
        for shot in ${SHOTS}; do
            run_one "$variant" "$dataset" "$shot"
        done
    done
done

echo ""
echo "============================================"
echo " All experiments finished : $(date)"
echo " Aggregate results with:"
echo "   cd ${REPO_ROOT} && python summarize_results.py"
echo "============================================"
