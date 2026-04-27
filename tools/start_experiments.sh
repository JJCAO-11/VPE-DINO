#!/usr/bin/env bash
# =====================================================================
# ETS @ CD-FSOD - One-shot Experiment Launcher
# ---------------------------------------------------------------------
# Performs pre-flight sanity checks, summarises the planned workload,
# asks for confirmation, then launches the requested experiments in the
# background with `nohup`.  After launch prints monitoring commands.
#
# Usage examples:
#   bash tools/start_experiments.sh                    # both variants, default GPUs
#   bash tools/start_experiments.sh --variants ets     # only ETS full
#   bash tools/start_experiments.sh --gpus 0,1,2,3
#   bash tools/start_experiments.sh --smoke            # 1 quick smoke test
#   bash tools/start_experiments.sh --mode parallel    # 4 jobs concurrent
#   bash tools/start_experiments.sh --yes              # no confirmation
#   bash tools/start_experiments.sh --dry-run          # show plan only
#
# Default behaviour:
#   - launch BOTH variants (baseline + ets)
#   - all 6 datasets
#   - all 3 shots (1, 5, 10)
#   - 4 GPUs (1,2,3,4)
#   - sequential distributed mode  (calls main_results_cdfsod.sh)
#   - background nohup launch (survives SSH disconnect)
# =====================================================================

set -euo pipefail

# ----------------------------------------------------------------------
# Defaults (overridable via flags below)
# ----------------------------------------------------------------------
VARIANTS="both"           # both | baseline | ets
GPUS_LIST="1,2,3,4"
DATASETS_DEFAULT="artaxor dior fish clipart1k neu-det uodd"
SHOTS_DEFAULT="1 5 10"
DATASETS=""
SHOTS=""
MODE="sequential"          # sequential | parallel
SMOKE=0
YES=0
DRY_RUN=0

# ----------------------------------------------------------------------
# Path helpers
# ----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/runner_logs"
mkdir -p "${LOG_DIR}"
TS="$(date '+%Y%m%d_%H%M%S')"

# ----------------------------------------------------------------------
# Colours (only if stdout is a terminal)
# ----------------------------------------------------------------------
if [[ -t 1 ]]; then
    R='\033[31m'; G='\033[32m'; Y='\033[33m'; B='\033[36m'; D='\033[2m'; N='\033[0m'
else
    R=''; G=''; Y=''; B=''; D=''; N=''
fi

err()  { echo -e "${R}[err]${N} $*" >&2; }
ok()   { echo -e "${G}[ok]${N}  $*"; }
warn() { echo -e "${Y}[warn]${N} $*"; }
info() { echo -e "${B}[..]${N}  $*"; }

# ----------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------
usage() { sed -n '2,28p' "$0"; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variants)  VARIANTS="$2"; shift 2 ;;
        --gpus)      GPUS_LIST="$2"; shift 2 ;;
        --datasets)  DATASETS="$2"; shift 2 ;;
        --shots)     SHOTS="$2"; shift 2 ;;
        --mode)      MODE="$2"; shift 2 ;;
        --smoke)     SMOKE=1; shift ;;
        --yes|-y)    YES=1; shift ;;
        --dry-run|-n) DRY_RUN=1; shift ;;
        -h|--help)   usage ;;
        *) err "unknown flag: $1"; usage ;;
    esac
done

DATASETS="${DATASETS:-${DATASETS_DEFAULT}}"
SHOTS="${SHOTS:-${SHOTS_DEFAULT}}"

# Smoke override
if [[ ${SMOKE} -eq 1 ]]; then
    info "smoke-test mode: only NEU-DET 5-shot ETS"
    VARIANTS="ets"
    DATASETS="neu-det"
    SHOTS="5"
fi

# Variants -> array
case "${VARIANTS}" in
    both)     VARIANTS_ARR=(baseline ets) ;;
    baseline) VARIANTS_ARR=(baseline) ;;
    ets)      VARIANTS_ARR=(ets) ;;
    *) err "--variants must be both|baseline|ets, got: ${VARIANTS}"; exit 2 ;;
esac

# Mode validation
case "${MODE}" in
    sequential|parallel) ;;
    *) err "--mode must be sequential|parallel, got: ${MODE}"; exit 2 ;;
esac

NUM_GPUS=$(echo "${GPUS_LIST}" | awk -F',' '{print NF}')

# ----------------------------------------------------------------------
# Pre-flight: working dir, conda env, python imports
# ----------------------------------------------------------------------
echo
echo "================================================================"
echo " ETS @ CD-FSOD - Experiment Launcher"
echo "================================================================"
echo " repo            : ${REPO_ROOT}"
echo " hostname        : $(hostname)"
echo " datetime        : $(date)"
echo " conda env       : ${CONDA_DEFAULT_ENV:-<none>}"
echo " python          : $(which python)"
echo "----------------------------------------------------------------"

cd "${REPO_ROOT}"

# 1) Conda env
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" == "base" ]]; then
    warn "conda env is '${CONDA_DEFAULT_ENV:-none}'.  Expected: ets"
    warn "Run 'conda activate ets' first, then retry.  (continuing anyway)"
else
    ok "conda env: ${CONDA_DEFAULT_ENV}"
fi

# 2) Python imports
info "checking python imports ..."
if python -c "import torch, mmdet, mmengine; \
    print('   torch', torch.__version__, '| cuda', torch.cuda.is_available()); \
    print('   mmdet', mmdet.__version__); \
    print('   mmengine', mmengine.__version__)"; then
    ok "core packages importable"
else
    err "python imports failed.  fix env before running."
    exit 1
fi

# 3) Datasets present
info "checking datasets ..."
MISSING_DATASETS=()
for d in ${DATASETS}; do
    case "$d" in
        artaxor)   subdir="ArTaxOr" ;;
        clipart1k) subdir="clipart1k" ;;
        dior)      subdir="DIOR" ;;
        fish)      subdir="FISH" ;;
        neu-det)   subdir="NEU-DET" ;;
        uodd)      subdir="UODD" ;;
        *) err "unknown dataset key: $d"; exit 2 ;;
    esac

    test_ann="datasets/${subdir}/annotations/test.json"
    if [[ ! -f "${test_ann}" ]]; then
        MISSING_DATASETS+=("${subdir}")
    fi
done
if [[ ${#MISSING_DATASETS[@]} -gt 0 ]]; then
    err "missing test annotation for: ${MISSING_DATASETS[*]}"
    exit 1
fi
N_DATASETS=$(echo ${DATASETS} | wc -w)
ok "${N_DATASETS} dataset(s) present"

# 4) Leaf configs
info "checking leaf configs ..."
N_CFG=$(find configs/cdfsod -mindepth 2 -name '*shot_*.py' | wc -l)
EXPECTED_CFG=36
if [[ "${N_CFG}" -lt "${EXPECTED_CFG}" ]]; then
    warn "found ${N_CFG} leaf configs (expected ${EXPECTED_CFG})"
    info "regenerating ..."
    python tools/gen_cdfsod_configs.py
    N_CFG=$(find configs/cdfsod -mindepth 2 -name '*shot_*.py' | wc -l)
fi
ok "${N_CFG} leaf configs"

# 5) GPU sanity
info "checking GPUs ..."
if ! command -v nvidia-smi >/dev/null; then
    err "nvidia-smi not found"
    exit 1
fi
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
ok "${GPU_COUNT} GPUs visible to nvidia-smi"

# Check requested GPUs aren't already loaded with our training
RUNNING_TRAIN=$(pgrep -af 'tools/train.py' | wc -l || true)
if [[ "${RUNNING_TRAIN}" -gt 0 ]]; then
    warn "${RUNNING_TRAIN} mmdet train processes already running:"
    pgrep -af 'tools/train.py' | head -5 | sed 's/^/      /'
    warn "they will compete for GPUs.  Press Ctrl-C now if not intended."
fi

# 6) Pretrained checkpoint cache (informational)
info "checking pretrained Swin-B cache ..."
PRE_CACHED=""
for cand in \
    "${HOME}/.cache/torch/hub/checkpoints/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth" \
    "${HOME}/.cache/mmengine/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth" \
    "${REPO_ROOT}/weights/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth"; do
    if [[ -f "${cand}" ]]; then
        PRE_CACHED="${cand}"
        break
    fi
done
if [[ -n "${PRE_CACHED}" ]]; then
    ok "pretrained ckpt cached at: ${PRE_CACHED}"
else
    warn "pretrained Swin-B not cached locally."
    warn "first training run will download it from CDN (~1.7GB)."
fi

# ----------------------------------------------------------------------
# Plan summary
# ----------------------------------------------------------------------
N_TASKS=0
for v in "${VARIANTS_ARR[@]}"; do
    for d in ${DATASETS}; do
        for s in ${SHOTS}; do
            N_TASKS=$((N_TASKS + 1))
        done
    done
done

echo
echo "================================================================"
echo " Plan"
echo "================================================================"
echo " variants        : ${VARIANTS_ARR[*]}"
echo " datasets        : ${DATASETS}"
echo " shots           : ${SHOTS}"
echo " GPUs            : ${GPUS_LIST}  (${NUM_GPUS} cards)"
echo " mode            : ${MODE}"
echo " total tasks     : ${N_TASKS}"
echo " est. duration   : sequential ~$((N_TASKS * 6))h to $((N_TASKS * 24))h"
echo "                   parallel   ~$((N_TASKS * 6 / NUM_GPUS))h to $((N_TASKS * 24 / NUM_GPUS))h"
echo " logs            : ${LOG_DIR}/runner_${TS}.log"
echo "================================================================"

if [[ ${DRY_RUN} -eq 1 ]]; then
    info "--dry-run: not launching."
    exit 0
fi

# ----------------------------------------------------------------------
# Confirmation
# ----------------------------------------------------------------------
if [[ ${YES} -eq 0 ]]; then
    echo
    read -r -p "Proceed?  [y/N] " ans
    case "${ans}" in
        y|Y|yes|YES) ;;
        *) info "aborted by user."; exit 0 ;;
    esac
fi

# ----------------------------------------------------------------------
# Launch
# ----------------------------------------------------------------------
LOG_FILE="${LOG_DIR}/runner_${TS}.log"
PID_FILE="${LOG_DIR}/runner_${TS}.pid"

if [[ ${SMOKE} -eq 1 ]]; then
    # Smoke: foreground, single task, easy to ctrl-C
    info "running smoke test in foreground (Ctrl-C to abort)"
    GPUS=${NUM_GPUS} GPU_IDS="${GPUS_LIST}" \
        DATASETS="${DATASETS}" SHOTS="${SHOTS}" \
        bash "${REPO_ROOT}/main_results_cdfsod.sh" "${VARIANTS_ARR[@]}"
    exit 0
fi

info "launching in background ..."

if [[ "${MODE}" == "sequential" ]]; then
    nohup env \
        GPUS=${NUM_GPUS} \
        GPU_IDS="${GPUS_LIST}" \
        DATASETS="${DATASETS}" \
        SHOTS="${SHOTS}" \
        bash "${REPO_ROOT}/main_results_cdfsod.sh" "${VARIANTS_ARR[@]}" \
        > "${LOG_FILE}" 2>&1 &
    RUN_PID=$!
else
    # Parallel mode: tools/run_parallel_cdfsod.py
    VARIANTS_FOR_PY="${VARIANTS_ARR[*]}"
    DATASETS_PY="${DATASETS}"
    SHOTS_PY="${SHOTS}"
    nohup python "${REPO_ROOT}/tools/run_parallel_cdfsod.py" \
        --gpus "${GPUS_LIST}" \
        --variants ${VARIANTS_FOR_PY} \
        --datasets ${DATASETS_PY} \
        --shots ${SHOTS_PY} \
        > "${LOG_FILE}" 2>&1 &
    RUN_PID=$!
fi

echo "${RUN_PID}" > "${PID_FILE}"

# Wait briefly then verify the launcher is alive (quick crashes should
# surface immediately).
sleep 3
if kill -0 "${RUN_PID}" 2>/dev/null; then
    ok "launcher started.  PID=${RUN_PID}"
else
    err "launcher died within 3s.  check ${LOG_FILE}"
    tail -50 "${LOG_FILE}" >&2
    exit 1
fi

# ----------------------------------------------------------------------
# How-to-monitor cheatsheet
# ----------------------------------------------------------------------
cat <<EOF

================================================================
 Launched OK  (PID=${RUN_PID})
================================================================

Monitor live log:
    tail -f ${LOG_FILE}

Show currently-running training process:
    ps -p \$(pgrep -af 'tools/train.py' | head -1 | awk '{print \$1}') -o cmd= 2>/dev/null
    # or simply:
    pgrep -af 'tools/train.py'

Live GPU utilisation:
    watch -n 5 nvidia-smi

Kill the whole experiment chain (graceful):
    kill ${RUN_PID}
    pkill -f 'tools/train.py'      # stop active mmdet workers too

When some best_*.pth ckpts appear, run PoE inference (idle GPU 0):
    CUDA_VISIBLE_DEVICES=0 python tools/poe_run_all.py \\
        --datasets neu-det --shots 5 --variants ets \\
        --alphas 0.1 0.3 0.5 --also-baseline

After ALL training finishes, aggregate:
    python summarize_results.py     &&  cat cdfsod_results.md
    cat work_dirs/cdfsod/poe/poe_summary.md

================================================================
EOF
