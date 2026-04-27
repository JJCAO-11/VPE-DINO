#!/bin/bash
# ============================================================================
# LoRA Experiments for CD-ViTO
# Run ablation studies comparing baseline vs LoRA-enhanced models
# ============================================================================

# --- Configuration ---
GPU_IDS="0,1,2,3"
NUM_GPUS=4
MODEL="l"  # l=ViT-L, b=ViT-B, s=ViT-S
RPN_CONFIG="configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml"
PRETRAINED="weights/trained/few-shot/vit${MODEL}_0089999.pth"

# ============================================================================
# Experiment 1: Baseline (no LoRA) — for comparison
# ============================================================================
echo "========================================"
echo "Experiment 1: Baseline (no LoRA)"
echo "========================================"

for dataset in neu-det uodd; do
  for shot in 1 5 10; do
    echo ">>> Baseline: ${dataset} ${shot}-shot"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} python tools/train_net.py \
      --num-gpus ${NUM_GPUS} \
      --config-file configs/${dataset}/vit${MODEL}_shot${shot}_${dataset}_finetune.yaml \
      MODEL.WEIGHTS ${PRETRAINED} \
      DE.OFFLINE_RPN_CONFIG ${RPN_CONFIG} \
      OUTPUT_DIR output/baseline/vit${MODEL}/${dataset}_${shot}shot/
  done
done

# ============================================================================
# Experiment 2: LoRA rank=4 (default, recommended for 5-shot)
# ============================================================================
echo "========================================"
echo "Experiment 2: LoRA rank=4"
echo "========================================"

for dataset in neu-det uodd; do
  for shot in 1 5 10; do
    echo ">>> LoRA r=4: ${dataset} ${shot}-shot"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} python tools/train_net.py \
      --num-gpus ${NUM_GPUS} \
      --config-file configs/${dataset}/vit${MODEL}_shot${shot}_${dataset}_finetune.yaml \
      MODEL.WEIGHTS ${PRETRAINED} \
      DE.OFFLINE_RPN_CONFIG ${RPN_CONFIG} \
      DE.USE_LORA True \
      DE.LORA_RANK 4 \
      DE.LORA_ALPHA 1.0 \
      OUTPUT_DIR output/lora_r4/vit${MODEL}/${dataset}_${shot}shot/
  done
done

# ============================================================================
# Experiment 3: LoRA rank ablation (rank=2, 8, 16)
# ============================================================================
echo "========================================"
echo "Experiment 3: LoRA rank ablation"
echo "========================================"

for rank in 2 8 16; do
  for dataset in neu-det uodd; do
    shot=5  # fix shot=5 for rank ablation
    echo ">>> LoRA r=${rank}: ${dataset} ${shot}-shot"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} python tools/train_net.py \
      --num-gpus ${NUM_GPUS} \
      --config-file configs/${dataset}/vit${MODEL}_shot${shot}_${dataset}_finetune.yaml \
      MODEL.WEIGHTS ${PRETRAINED} \
      DE.OFFLINE_RPN_CONFIG ${RPN_CONFIG} \
      DE.USE_LORA True \
      DE.LORA_RANK ${rank} \
      DE.LORA_ALPHA 1.0 \
      OUTPUT_DIR output/lora_r${rank}/vit${MODEL}/${dataset}_${shot}shot/
  done
done

# ============================================================================
# Experiment 4: MHCA Aggregator only (方案B单独)
# ============================================================================
echo "========================================"
echo "Experiment 4: MHCA Aggregator only"
echo "========================================"

for dataset in neu-det uodd; do
  for shot in 1 5 10; do
    echo ">>> MHCA only: ${dataset} ${shot}-shot"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} python tools/train_net.py \
      --num-gpus ${NUM_GPUS} \
      --config-file configs/${dataset}/vit${MODEL}_shot${shot}_${dataset}_finetune.yaml \
      MODEL.WEIGHTS ${PRETRAINED} \
      DE.OFFLINE_RPN_CONFIG ${RPN_CONFIG} \
      DE.USE_MHCA_AGGREGATOR True \
      DE.MHCA_NUM_HEADS 4 \
      OUTPUT_DIR output/mhca/vit${MODEL}/${dataset}_${shot}shot/
  done
done

# ============================================================================
# Experiment 5: LoRA + MHCA (方案A+B组合)
# ============================================================================
echo "========================================"
echo "Experiment 5: LoRA + MHCA combined"
echo "========================================"

for dataset in neu-det uodd; do
  for shot in 1 5 10; do
    echo ">>> LoRA+MHCA: ${dataset} ${shot}-shot"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} python tools/train_net.py \
      --num-gpus ${NUM_GPUS} \
      --config-file configs/${dataset}/vit${MODEL}_shot${shot}_${dataset}_finetune.yaml \
      MODEL.WEIGHTS ${PRETRAINED} \
      DE.OFFLINE_RPN_CONFIG ${RPN_CONFIG} \
      DE.USE_LORA True \
      DE.LORA_RANK 4 \
      DE.USE_MHCA_AGGREGATOR True \
      DE.MHCA_NUM_HEADS 4 \
      OUTPUT_DIR output/lora_mhca/vit${MODEL}/${dataset}_${shot}shot/
  done
done

# ============================================================================
# Experiment 6: Full benchmark — LoRA+MHCA on all 6 datasets
# ============================================================================
echo "========================================"
echo "Experiment 6: Full benchmark (LoRA+MHCA)"
echo "========================================"

for dataset in artaxor dior fish clipart1k neu-det uodd; do
  for shot in 1 5 10; do
    echo ">>> Full model: ${dataset} ${shot}-shot"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} python tools/train_net.py \
      --num-gpus ${NUM_GPUS} \
      --config-file configs/${dataset}/vit${MODEL}_shot${shot}_${dataset}_finetune.yaml \
      MODEL.WEIGHTS ${PRETRAINED} \
      DE.OFFLINE_RPN_CONFIG ${RPN_CONFIG} \
      DE.USE_LORA True \
      DE.LORA_RANK 4 \
      DE.USE_MHCA_AGGREGATOR True \
      DE.MHCA_NUM_HEADS 4 \
      OUTPUT_DIR output/full_model/vit${MODEL}/${dataset}_${shot}shot/
  done
done

echo "All experiments completed!"
