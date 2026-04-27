#!/bin/bash
# ===========================================================
# 一键运行全部 Baseline + DGFM 对比实验
# 共 5 个数据集 x 2 种方法 = 10 个实验
# ===========================================================

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset DETECTRON2_DATASETS

GPUS=4
WEIGHTS="weights/trained/few-shot/vitl_0089999.pth"
RPN_CFG="configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml"

# 每次实验之间清理显存
cleanup_gpu() {
    sleep 3
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 2
}

run_baseline() {
    local dataset=$1
    local config=$2
    local outdir=$3
    echo ""
    echo ">>> Baseline - ${dataset} $(date '+%H:%M:%S')"
    python tools/train_net.py --num-gpus $GPUS \
      --config-file ${config} \
      MODEL.WEIGHTS $WEIGHTS \
      DE.OFFLINE_RPN_CONFIG $RPN_CFG \
      OUTPUT_DIR ${outdir} || echo "WARNING: ${dataset} baseline failed!"
    cleanup_gpu
}

run_dgfm() {
    local dataset=$1
    local config=$2
    local outdir=$3
    echo ""
    echo ">>> DGFM - ${dataset} $(date '+%H:%M:%S')"
    python tools/train_net.py --num-gpus $GPUS \
      --config-file ${config} \
      MODEL.WEIGHTS $WEIGHTS \
      DE.OFFLINE_RPN_CONFIG $RPN_CFG \
      DE.USE_DGFM True DE.DGFM_REP_SCALE 4 DE.DGFM_RESIDUAL_WEIGHT 0.1 \
      OUTPUT_DIR ${outdir} || echo "WARNING: ${dataset} DGFM failed!"
    cleanup_gpu
}

echo "============================================"
echo "  开始全套实验 $(date)"
echo "============================================"

# ---------- 1. ArTaxOr ----------
run_baseline "ArTaxOr" \
  "configs/artaxor/vitl_shot5_artaxor_finetune.yaml" \
  "output/baseline/vitl/artaxor_5shot/"
run_dgfm "ArTaxOr" \
  "configs/artaxor/vitl_shot5_artaxor_finetune.yaml" \
  "output/dgfm/vitl/artaxor_5shot/"

# ---------- 2. Clipart1k ----------
run_baseline "Clipart1k" \
  "configs/clipart1k/vitl_shot5_clipart1k_finetune.yaml" \
  "output/baseline/vitl/clipart1k_5shot/"
run_dgfm "Clipart1k" \
  "configs/clipart1k/vitl_shot5_clipart1k_finetune.yaml" \
  "output/dgfm/vitl/clipart1k_5shot/"

# ---------- 3. DIOR ----------
run_baseline "DIOR" \
  "configs/dior/vitl_shot5_dior_finetune.yaml" \
  "output/baseline/vitl/dior_5shot/"
run_dgfm "DIOR" \
  "configs/dior/vitl_shot5_dior_finetune.yaml" \
  "output/dgfm/vitl/dior_5shot/"

# ---------- 4. FISH ----------
run_baseline "FISH" \
  "configs/fish/vitl_shot5_fish_finetune.yaml" \
  "output/baseline/vitl/fish_5shot/"
run_dgfm "FISH" \
  "configs/fish/vitl_shot5_fish_finetune.yaml" \
  "output/dgfm/vitl/fish_5shot/"

# ---------- 5. UODD ----------
run_baseline "UODD" \
  "configs/uodd/vitl_shot5_uodd_finetune.yaml" \
  "output/baseline/vitl/uodd_5shot/"
run_dgfm "UODD" \
  "configs/uodd/vitl_shot5_uodd_finetune.yaml" \
  "output/dgfm/vitl/uodd_5shot/"

echo ""
echo "============================================"
echo "  全部实验完成！ $(date)"
echo "============================================"
echo ""
echo ">>> 结果汇总："
echo "--- Baseline ---"
grep -rh "####" output/baseline/vitl/*/log.txt 2>/dev/null | head -20
echo ""
echo "--- DGFM ---"
grep -rh "####" output/dgfm/vitl/*/log.txt 2>/dev/null | head -20
