#!/usr/bin/env bash
# ============================================================
# All 54 ETS @ CD-FSOD experiments - 4-GPU distributed, sequential.
# ------------------------------------------------------------
# RUN FROM: ETS/mmdetection/   (cd into mmdetection first!)
# Each command uses ALL 4 GPUs (0,1,2,3) and trains ONE experiment.
# Run them ONE AT A TIME, in order.
#
# Three usage modes
# -----------------
# (A) Sequential, full batch (recommended):
#       bash launch_cdfsod_commands.sh
#       # or background:
#       nohup bash launch_cdfsod_commands.sh > runner.log 2>&1 &
#
# (B) Run a single experiment in foreground (see live log):
#       Copy ONE block below and paste into your shell.
#
# (C) Run a single experiment in background (with nohup):
#       Wrap the same block with `nohup bash -c '...' &` -- the
#       inline `CUDA_VISIBLE_DEVICES=` prefix will NOT work directly
#       after nohup, so always use bash -c wrapping or `env`.
#
# Output layout:
#   work_dirs/cdfsod/<variant>/<dataset>_<shot>shot/train.log
#
# Aggregate metrics afterwards:
#   cd <repo_root> && python summarize_results.py
# ============================================================
set -e

# ============================================================
# Variant: baseline
# ============================================================

# [01/54]  baseline / artaxor / 1-shot  (port 29501)
mkdir -p work_dirs/cdfsod/baseline/artaxor_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29501 tools/train.py ../configs/cdfsod/artaxor/1shot_baseline.py --work-dir work_dirs/cdfsod/baseline/artaxor_1shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/baseline/artaxor_1shot/train.log

# [02/54]  baseline / artaxor / 5-shot  (port 29502)
mkdir -p work_dirs/cdfsod/baseline/artaxor_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29502 tools/train.py ../configs/cdfsod/artaxor/5shot_baseline.py --work-dir work_dirs/cdfsod/baseline/artaxor_5shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/baseline/artaxor_5shot/train.log

# [03/54]  baseline / artaxor / 10-shot  (port 29503)
mkdir -p work_dirs/cdfsod/baseline/artaxor_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29503 tools/train.py ../configs/cdfsod/artaxor/10shot_baseline.py --work-dir work_dirs/cdfsod/baseline/artaxor_10shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/baseline/artaxor_10shot/train.log

# [04/54]  baseline / dior / 1-shot  (port 29504)
mkdir -p work_dirs/cdfsod/baseline/dior_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29504 tools/train.py ../configs/cdfsod/dior/1shot_baseline.py --work-dir work_dirs/cdfsod/baseline/dior_1shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/baseline/dior_1shot/train.log

# [05/54]  baseline / dior / 5-shot  (port 29505)
mkdir -p work_dirs/cdfsod/baseline/dior_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29505 tools/train.py ../configs/cdfsod/dior/5shot_baseline.py --work-dir work_dirs/cdfsod/baseline/dior_5shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/baseline/dior_5shot/train.log

# [06/54]  baseline / dior / 10-shot  (port 29506)
mkdir -p work_dirs/cdfsod/baseline/dior_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29506 tools/train.py ../configs/cdfsod/dior/10shot_baseline.py --work-dir work_dirs/cdfsod/baseline/dior_10shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/baseline/dior_10shot/train.log

# [07/54]  baseline / fish / 1-shot  (port 29507)
mkdir -p work_dirs/cdfsod/baseline/fish_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29507 tools/train.py ../configs/cdfsod/fish/1shot_baseline.py --work-dir work_dirs/cdfsod/baseline/fish_1shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/baseline/fish_1shot/train.log

# [08/54]  baseline / fish / 5-shot  (port 29508)
mkdir -p work_dirs/cdfsod/baseline/fish_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29508 tools/train.py ../configs/cdfsod/fish/5shot_baseline.py --work-dir work_dirs/cdfsod/baseline/fish_5shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/baseline/fish_5shot/train.log

# [09/54]  baseline / fish / 10-shot  (port 29509)
mkdir -p work_dirs/cdfsod/baseline/fish_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29509 tools/train.py ../configs/cdfsod/fish/10shot_baseline.py --work-dir work_dirs/cdfsod/baseline/fish_10shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/baseline/fish_10shot/train.log

# [10/54]  baseline / clipart1k / 1-shot  (port 29510)
mkdir -p work_dirs/cdfsod/baseline/clipart1k_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29510 tools/train.py ../configs/cdfsod/clipart1k/1shot_baseline.py --work-dir work_dirs/cdfsod/baseline/clipart1k_1shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/baseline/clipart1k_1shot/train.log

# [11/54]  baseline / clipart1k / 5-shot  (port 29511)
mkdir -p work_dirs/cdfsod/baseline/clipart1k_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29511 tools/train.py ../configs/cdfsod/clipart1k/5shot_baseline.py --work-dir work_dirs/cdfsod/baseline/clipart1k_5shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/baseline/clipart1k_5shot/train.log

# [12/54]  baseline / clipart1k / 10-shot  (port 29512)
mkdir -p work_dirs/cdfsod/baseline/clipart1k_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29512 tools/train.py ../configs/cdfsod/clipart1k/10shot_baseline.py --work-dir work_dirs/cdfsod/baseline/clipart1k_10shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/baseline/clipart1k_10shot/train.log

# [13/54]  baseline / neu-det / 1-shot  (port 29513)
mkdir -p work_dirs/cdfsod/baseline/neu-det_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29513 tools/train.py ../configs/cdfsod/neu-det/1shot_baseline.py --work-dir work_dirs/cdfsod/baseline/neu-det_1shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/baseline/neu-det_1shot/train.log

# [14/54]  baseline / neu-det / 5-shot  (port 29514)
mkdir -p work_dirs/cdfsod/baseline/neu-det_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29514 tools/train.py ../configs/cdfsod/neu-det/5shot_baseline.py --work-dir work_dirs/cdfsod/baseline/neu-det_5shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/baseline/neu-det_5shot/train.log

# [15/54]  baseline / neu-det / 10-shot  (port 29515)
mkdir -p work_dirs/cdfsod/baseline/neu-det_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29515 tools/train.py ../configs/cdfsod/neu-det/10shot_baseline.py --work-dir work_dirs/cdfsod/baseline/neu-det_10shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/baseline/neu-det_10shot/train.log

# [16/54]  baseline / uodd / 1-shot  (port 29516)
mkdir -p work_dirs/cdfsod/baseline/uodd_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29516 tools/train.py ../configs/cdfsod/uodd/1shot_baseline.py --work-dir work_dirs/cdfsod/baseline/uodd_1shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/baseline/uodd_1shot/train.log

# [17/54]  baseline / uodd / 5-shot  (port 29517)
mkdir -p work_dirs/cdfsod/baseline/uodd_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29517 tools/train.py ../configs/cdfsod/uodd/5shot_baseline.py --work-dir work_dirs/cdfsod/baseline/uodd_5shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/baseline/uodd_5shot/train.log

# [18/54]  baseline / uodd / 10-shot  (port 29518)
mkdir -p work_dirs/cdfsod/baseline/uodd_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29518 tools/train.py ../configs/cdfsod/uodd/10shot_baseline.py --work-dir work_dirs/cdfsod/baseline/uodd_10shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/baseline/uodd_10shot/train.log

# ============================================================
# Variant: ets
# ============================================================

# [19/54]  ets / artaxor / 1-shot  (port 29519)
mkdir -p work_dirs/cdfsod/ets/artaxor_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29519 tools/train.py ../configs/cdfsod/artaxor/1shot_ets.py --work-dir work_dirs/cdfsod/ets/artaxor_1shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/ets/artaxor_1shot/train.log

# [20/54]  ets / artaxor / 5-shot  (port 29520)
mkdir -p work_dirs/cdfsod/ets/artaxor_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29520 tools/train.py ../configs/cdfsod/artaxor/5shot_ets.py --work-dir work_dirs/cdfsod/ets/artaxor_5shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/ets/artaxor_5shot/train.log

# [21/54]  ets / artaxor / 10-shot  (port 29521)
mkdir -p work_dirs/cdfsod/ets/artaxor_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29521 tools/train.py ../configs/cdfsod/artaxor/10shot_ets.py --work-dir work_dirs/cdfsod/ets/artaxor_10shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/ets/artaxor_10shot/train.log

# [22/54]  ets / dior / 1-shot  (port 29522)
mkdir -p work_dirs/cdfsod/ets/dior_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29522 tools/train.py ../configs/cdfsod/dior/1shot_ets.py --work-dir work_dirs/cdfsod/ets/dior_1shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/ets/dior_1shot/train.log

# [23/54]  ets / dior / 5-shot  (port 29523)
mkdir -p work_dirs/cdfsod/ets/dior_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29523 tools/train.py ../configs/cdfsod/dior/5shot_ets.py --work-dir work_dirs/cdfsod/ets/dior_5shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/ets/dior_5shot/train.log

# [24/54]  ets / dior / 10-shot  (port 29524)
mkdir -p work_dirs/cdfsod/ets/dior_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29524 tools/train.py ../configs/cdfsod/dior/10shot_ets.py --work-dir work_dirs/cdfsod/ets/dior_10shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/ets/dior_10shot/train.log

# [25/54]  ets / fish / 1-shot  (port 29525)
mkdir -p work_dirs/cdfsod/ets/fish_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29525 tools/train.py ../configs/cdfsod/fish/1shot_ets.py --work-dir work_dirs/cdfsod/ets/fish_1shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/ets/fish_1shot/train.log

# [26/54]  ets / fish / 5-shot  (port 29526)
mkdir -p work_dirs/cdfsod/ets/fish_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29526 tools/train.py ../configs/cdfsod/fish/5shot_ets.py --work-dir work_dirs/cdfsod/ets/fish_5shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/ets/fish_5shot/train.log

# [27/54]  ets / fish / 10-shot  (port 29527)
mkdir -p work_dirs/cdfsod/ets/fish_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29527 tools/train.py ../configs/cdfsod/fish/10shot_ets.py --work-dir work_dirs/cdfsod/ets/fish_10shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/ets/fish_10shot/train.log

# [28/54]  ets / clipart1k / 1-shot  (port 29528)
mkdir -p work_dirs/cdfsod/ets/clipart1k_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29528 tools/train.py ../configs/cdfsod/clipart1k/1shot_ets.py --work-dir work_dirs/cdfsod/ets/clipart1k_1shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/ets/clipart1k_1shot/train.log

# [29/54]  ets / clipart1k / 5-shot  (port 29529)
mkdir -p work_dirs/cdfsod/ets/clipart1k_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29529 tools/train.py ../configs/cdfsod/clipart1k/5shot_ets.py --work-dir work_dirs/cdfsod/ets/clipart1k_5shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/ets/clipart1k_5shot/train.log

# [30/54]  ets / clipart1k / 10-shot  (port 29530)
mkdir -p work_dirs/cdfsod/ets/clipart1k_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29530 tools/train.py ../configs/cdfsod/clipart1k/10shot_ets.py --work-dir work_dirs/cdfsod/ets/clipart1k_10shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/ets/clipart1k_10shot/train.log

# [31/54]  ets / neu-det / 1-shot  (port 29531)
mkdir -p work_dirs/cdfsod/ets/neu-det_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29531 tools/train.py ../configs/cdfsod/neu-det/1shot_ets.py --work-dir work_dirs/cdfsod/ets/neu-det_1shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/ets/neu-det_1shot/train.log

# [32/54]  ets / neu-det / 5-shot  (port 29532)
mkdir -p work_dirs/cdfsod/ets/neu-det_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29532 tools/train.py ../configs/cdfsod/neu-det/5shot_ets.py --work-dir work_dirs/cdfsod/ets/neu-det_5shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/ets/neu-det_5shot/train.log

# [33/54]  ets / neu-det / 10-shot  (port 29533)
mkdir -p work_dirs/cdfsod/ets/neu-det_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29533 tools/train.py ../configs/cdfsod/neu-det/10shot_ets.py --work-dir work_dirs/cdfsod/ets/neu-det_10shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/ets/neu-det_10shot/train.log

# [34/54]  ets / uodd / 1-shot  (port 29534)
mkdir -p work_dirs/cdfsod/ets/uodd_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29534 tools/train.py ../configs/cdfsod/uodd/1shot_ets.py --work-dir work_dirs/cdfsod/ets/uodd_1shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/ets/uodd_1shot/train.log

# [35/54]  ets / uodd / 5-shot  (port 29535)
mkdir -p work_dirs/cdfsod/ets/uodd_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29535 tools/train.py ../configs/cdfsod/uodd/5shot_ets.py --work-dir work_dirs/cdfsod/ets/uodd_5shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/ets/uodd_5shot/train.log

# [36/54]  ets / uodd / 10-shot  (port 29536)
mkdir -p work_dirs/cdfsod/ets/uodd_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29536 tools/train.py ../configs/cdfsod/uodd/10shot_ets.py --work-dir work_dirs/cdfsod/ets/uodd_10shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/ets/uodd_10shot/train.log

# ============================================================
# Variant: vpe
# ============================================================

# [37/54]  vpe / artaxor / 1-shot  (port 29537)
mkdir -p work_dirs/cdfsod/vpe/artaxor_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29537 tools/train.py ../configs/cdfsod/artaxor/1shot_vpe.py --work-dir work_dirs/cdfsod/vpe/artaxor_1shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/vpe/artaxor_1shot/train.log

# [38/54]  vpe / artaxor / 5-shot  (port 29538)
mkdir -p work_dirs/cdfsod/vpe/artaxor_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29538 tools/train.py ../configs/cdfsod/artaxor/5shot_vpe.py --work-dir work_dirs/cdfsod/vpe/artaxor_5shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/vpe/artaxor_5shot/train.log

# [39/54]  vpe / artaxor / 10-shot  (port 29539)
mkdir -p work_dirs/cdfsod/vpe/artaxor_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29539 tools/train.py ../configs/cdfsod/artaxor/10shot_vpe.py --work-dir work_dirs/cdfsod/vpe/artaxor_10shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/vpe/artaxor_10shot/train.log

# [40/54]  vpe / dior / 1-shot  (port 29540)
mkdir -p work_dirs/cdfsod/vpe/dior_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29540 tools/train.py ../configs/cdfsod/dior/1shot_vpe.py --work-dir work_dirs/cdfsod/vpe/dior_1shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/vpe/dior_1shot/train.log

# [41/54]  vpe / dior / 5-shot  (port 29541)
mkdir -p work_dirs/cdfsod/vpe/dior_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29541 tools/train.py ../configs/cdfsod/dior/5shot_vpe.py --work-dir work_dirs/cdfsod/vpe/dior_5shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/vpe/dior_5shot/train.log

# [42/54]  vpe / dior / 10-shot  (port 29542)
mkdir -p work_dirs/cdfsod/vpe/dior_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29542 tools/train.py ../configs/cdfsod/dior/10shot_vpe.py --work-dir work_dirs/cdfsod/vpe/dior_10shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/vpe/dior_10shot/train.log

# [43/54]  vpe / fish / 1-shot  (port 29543)
mkdir -p work_dirs/cdfsod/vpe/fish_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29543 tools/train.py ../configs/cdfsod/fish/1shot_vpe.py --work-dir work_dirs/cdfsod/vpe/fish_1shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/vpe/fish_1shot/train.log

# [44/54]  vpe / fish / 5-shot  (port 29544)
mkdir -p work_dirs/cdfsod/vpe/fish_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29544 tools/train.py ../configs/cdfsod/fish/5shot_vpe.py --work-dir work_dirs/cdfsod/vpe/fish_5shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/vpe/fish_5shot/train.log

# [45/54]  vpe / fish / 10-shot  (port 29545)
mkdir -p work_dirs/cdfsod/vpe/fish_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29545 tools/train.py ../configs/cdfsod/fish/10shot_vpe.py --work-dir work_dirs/cdfsod/vpe/fish_10shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/vpe/fish_10shot/train.log

# [46/54]  vpe / clipart1k / 1-shot  (port 29546)
mkdir -p work_dirs/cdfsod/vpe/clipart1k_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29546 tools/train.py ../configs/cdfsod/clipart1k/1shot_vpe.py --work-dir work_dirs/cdfsod/vpe/clipart1k_1shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/vpe/clipart1k_1shot/train.log

# [47/54]  vpe / clipart1k / 5-shot  (port 29547)
mkdir -p work_dirs/cdfsod/vpe/clipart1k_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29547 tools/train.py ../configs/cdfsod/clipart1k/5shot_vpe.py --work-dir work_dirs/cdfsod/vpe/clipart1k_5shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/vpe/clipart1k_5shot/train.log

# [48/54]  vpe / clipart1k / 10-shot  (port 29548)
mkdir -p work_dirs/cdfsod/vpe/clipart1k_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29548 tools/train.py ../configs/cdfsod/clipart1k/10shot_vpe.py --work-dir work_dirs/cdfsod/vpe/clipart1k_10shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/vpe/clipart1k_10shot/train.log

# [49/54]  vpe / neu-det / 1-shot  (port 29549)
mkdir -p work_dirs/cdfsod/vpe/neu-det_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29549 tools/train.py ../configs/cdfsod/neu-det/1shot_vpe.py --work-dir work_dirs/cdfsod/vpe/neu-det_1shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/vpe/neu-det_1shot/train.log

# [50/54]  vpe / neu-det / 5-shot  (port 29550)
mkdir -p work_dirs/cdfsod/vpe/neu-det_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29550 tools/train.py ../configs/cdfsod/neu-det/5shot_vpe.py --work-dir work_dirs/cdfsod/vpe/neu-det_5shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/vpe/neu-det_5shot/train.log

# [51/54]  vpe / neu-det / 10-shot  (port 29551)
mkdir -p work_dirs/cdfsod/vpe/neu-det_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29551 tools/train.py ../configs/cdfsod/neu-det/10shot_vpe.py --work-dir work_dirs/cdfsod/vpe/neu-det_10shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/vpe/neu-det_10shot/train.log

# [52/54]  vpe / uodd / 1-shot  (port 29552)
mkdir -p work_dirs/cdfsod/vpe/uodd_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29552 tools/train.py ../configs/cdfsod/uodd/1shot_vpe.py --work-dir work_dirs/cdfsod/vpe/uodd_1shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/vpe/uodd_1shot/train.log

# [53/54]  vpe / uodd / 5-shot  (port 29553)
mkdir -p work_dirs/cdfsod/vpe/uodd_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29553 tools/train.py ../configs/cdfsod/uodd/5shot_vpe.py --work-dir work_dirs/cdfsod/vpe/uodd_5shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/vpe/uodd_5shot/train.log

# [54/54]  vpe / uodd / 10-shot  (port 29554)
mkdir -p work_dirs/cdfsod/vpe/uodd_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29554 tools/train.py ../configs/cdfsod/uodd/10shot_vpe.py --work-dir work_dirs/cdfsod/vpe/uodd_10shot --launcher pytorch 2>&1 | tee work_dirs/cdfsod/vpe/uodd_10shot/train.log
