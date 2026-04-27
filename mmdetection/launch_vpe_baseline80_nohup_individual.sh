#!/usr/bin/env bash
# Copy and run ONE command at a time from ETS/mmdetection. Each command uses GPUs 0,1,2,3.
echo "Copy and run ONE command at a time from this file; do not execute the whole file."
exit 1

# vpe_baseline80 / artaxor / 1-shot  (optional rerun; current best already exists in artaxor_1shot_recover80_v2_ddp)
mkdir -p work_dirs/cdfsod/vpe_baseline80/artaxor_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29601 tools/train.py ../configs/cdfsod/artaxor/1shot_vpe.py --work-dir work_dirs/cdfsod/vpe_baseline80/artaxor_1shot --launcher pytorch > work_dirs/cdfsod/vpe_baseline80/artaxor_1shot/nohup.log 2>&1 & echo "[vpe_baseline80 artaxor 1shot] pid=$! log=work_dirs/cdfsod/vpe_baseline80/artaxor_1shot/nohup.log"

# vpe_baseline80 / artaxor / 5-shot
mkdir -p work_dirs/cdfsod/vpe_baseline80/artaxor_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29602 tools/train.py ../configs/cdfsod/artaxor/5shot_vpe.py --work-dir work_dirs/cdfsod/vpe_baseline80/artaxor_5shot --launcher pytorch > work_dirs/cdfsod/vpe_baseline80/artaxor_5shot/nohup.log 2>&1 & echo "[vpe_baseline80 artaxor 5shot] pid=$! log=work_dirs/cdfsod/vpe_baseline80/artaxor_5shot/nohup.log"

# vpe_baseline80 / artaxor / 10-shot
mkdir -p work_dirs/cdfsod/vpe_baseline80/artaxor_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29603 tools/train.py ../configs/cdfsod/artaxor/10shot_vpe.py --work-dir work_dirs/cdfsod/vpe_baseline80/artaxor_10shot --launcher pytorch > work_dirs/cdfsod/vpe_baseline80/artaxor_10shot/nohup.log 2>&1 & echo "[vpe_baseline80 artaxor 10shot] pid=$! log=work_dirs/cdfsod/vpe_baseline80/artaxor_10shot/nohup.log"

# vpe_baseline80 / dior / 1-shot
mkdir -p work_dirs/cdfsod/vpe_baseline80/dior_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29604 tools/train.py ../configs/cdfsod/dior/1shot_vpe.py --work-dir work_dirs/cdfsod/vpe_baseline80/dior_1shot --launcher pytorch > work_dirs/cdfsod/vpe_baseline80/dior_1shot/nohup.log 2>&1 & echo "[vpe_baseline80 dior 1shot] pid=$! log=work_dirs/cdfsod/vpe_baseline80/dior_1shot/nohup.log"

# vpe_baseline80 / dior / 5-shot
mkdir -p work_dirs/cdfsod/vpe_baseline80/dior_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29605 tools/train.py ../configs/cdfsod/dior/5shot_vpe.py --work-dir work_dirs/cdfsod/vpe_baseline80/dior_5shot --launcher pytorch > work_dirs/cdfsod/vpe_baseline80/dior_5shot/nohup.log 2>&1 & echo "[vpe_baseline80 dior 5shot] pid=$! log=work_dirs/cdfsod/vpe_baseline80/dior_5shot/nohup.log"

# vpe_baseline80 / dior / 10-shot
mkdir -p work_dirs/cdfsod/vpe_baseline80/dior_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29606 tools/train.py ../configs/cdfsod/dior/10shot_vpe.py --work-dir work_dirs/cdfsod/vpe_baseline80/dior_10shot --launcher pytorch > work_dirs/cdfsod/vpe_baseline80/dior_10shot/nohup.log 2>&1 & echo "[vpe_baseline80 dior 10shot] pid=$! log=work_dirs/cdfsod/vpe_baseline80/dior_10shot/nohup.log"

# vpe_baseline80 / fish / 1-shot
mkdir -p work_dirs/cdfsod/vpe_baseline80/fish_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29607 tools/train.py ../configs/cdfsod/fish/1shot_vpe.py --work-dir work_dirs/cdfsod/vpe_baseline80/fish_1shot --launcher pytorch > work_dirs/cdfsod/vpe_baseline80/fish_1shot/nohup.log 2>&1 & echo "[vpe_baseline80 fish 1shot] pid=$! log=work_dirs/cdfsod/vpe_baseline80/fish_1shot/nohup.log"

# vpe_baseline80 / fish / 5-shot
mkdir -p work_dirs/cdfsod/vpe_baseline80/fish_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29608 tools/train.py ../configs/cdfsod/fish/5shot_vpe.py --work-dir work_dirs/cdfsod/vpe_baseline80/fish_5shot --launcher pytorch > work_dirs/cdfsod/vpe_baseline80/fish_5shot/nohup.log 2>&1 & echo "[vpe_baseline80 fish 5shot] pid=$! log=work_dirs/cdfsod/vpe_baseline80/fish_5shot/nohup.log"

# vpe_baseline80 / fish / 10-shot
mkdir -p work_dirs/cdfsod/vpe_baseline80/fish_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29609 tools/train.py ../configs/cdfsod/fish/10shot_vpe.py --work-dir work_dirs/cdfsod/vpe_baseline80/fish_10shot --launcher pytorch > work_dirs/cdfsod/vpe_baseline80/fish_10shot/nohup.log 2>&1 & echo "[vpe_baseline80 fish 10shot] pid=$! log=work_dirs/cdfsod/vpe_baseline80/fish_10shot/nohup.log"

# vpe_baseline80 / clipart1k / 1-shot
mkdir -p work_dirs/cdfsod/vpe_baseline80/clipart1k_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29610 tools/train.py ../configs/cdfsod/clipart1k/1shot_vpe.py --work-dir work_dirs/cdfsod/vpe_baseline80/clipart1k_1shot --launcher pytorch > work_dirs/cdfsod/vpe_baseline80/clipart1k_1shot/nohup.log 2>&1 & echo "[vpe_baseline80 clipart1k 1shot] pid=$! log=work_dirs/cdfsod/vpe_baseline80/clipart1k_1shot/nohup.log"

# vpe_baseline80 / clipart1k / 5-shot
mkdir -p work_dirs/cdfsod/vpe_baseline80/clipart1k_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29611 tools/train.py ../configs/cdfsod/clipart1k/5shot_vpe.py --work-dir work_dirs/cdfsod/vpe_baseline80/clipart1k_5shot --launcher pytorch > work_dirs/cdfsod/vpe_baseline80/clipart1k_5shot/nohup.log 2>&1 & echo "[vpe_baseline80 clipart1k 5shot] pid=$! log=work_dirs/cdfsod/vpe_baseline80/clipart1k_5shot/nohup.log"

# vpe_baseline80 / clipart1k / 10-shot
mkdir -p work_dirs/cdfsod/vpe_baseline80/clipart1k_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29612 tools/train.py ../configs/cdfsod/clipart1k/10shot_vpe.py --work-dir work_dirs/cdfsod/vpe_baseline80/clipart1k_10shot --launcher pytorch > work_dirs/cdfsod/vpe_baseline80/clipart1k_10shot/nohup.log 2>&1 & echo "[vpe_baseline80 clipart1k 10shot] pid=$! log=work_dirs/cdfsod/vpe_baseline80/clipart1k_10shot/nohup.log"

# vpe_baseline80 / neu-det / 1-shot
mkdir -p work_dirs/cdfsod/vpe_baseline80/neu-det_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29613 tools/train.py ../configs/cdfsod/neu-det/1shot_vpe.py --work-dir work_dirs/cdfsod/vpe_baseline80/neu-det_1shot --launcher pytorch > work_dirs/cdfsod/vpe_baseline80/neu-det_1shot/nohup.log 2>&1 & echo "[vpe_baseline80 neu-det 1shot] pid=$! log=work_dirs/cdfsod/vpe_baseline80/neu-det_1shot/nohup.log"

# vpe_baseline80 / neu-det / 5-shot
mkdir -p work_dirs/cdfsod/vpe_baseline80/neu-det_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29614 tools/train.py ../configs/cdfsod/neu-det/5shot_vpe.py --work-dir work_dirs/cdfsod/vpe_baseline80/neu-det_5shot --launcher pytorch > work_dirs/cdfsod/vpe_baseline80/neu-det_5shot/nohup.log 2>&1 & echo "[vpe_baseline80 neu-det 5shot] pid=$! log=work_dirs/cdfsod/vpe_baseline80/neu-det_5shot/nohup.log"

# vpe_baseline80 / neu-det / 10-shot
mkdir -p work_dirs/cdfsod/vpe_baseline80/neu-det_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29615 tools/train.py ../configs/cdfsod/neu-det/10shot_vpe.py --work-dir work_dirs/cdfsod/vpe_baseline80/neu-det_10shot --launcher pytorch > work_dirs/cdfsod/vpe_baseline80/neu-det_10shot/nohup.log 2>&1 & echo "[vpe_baseline80 neu-det 10shot] pid=$! log=work_dirs/cdfsod/vpe_baseline80/neu-det_10shot/nohup.log"

# vpe_baseline80 / uodd / 1-shot
mkdir -p work_dirs/cdfsod/vpe_baseline80/uodd_1shot && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29616 tools/train.py ../configs/cdfsod/uodd/1shot_vpe.py --work-dir work_dirs/cdfsod/vpe_baseline80/uodd_1shot --launcher pytorch > work_dirs/cdfsod/vpe_baseline80/uodd_1shot/nohup.log 2>&1 & echo "[vpe_baseline80 uodd 1shot] pid=$! log=work_dirs/cdfsod/vpe_baseline80/uodd_1shot/nohup.log"

# vpe_baseline80 / uodd / 5-shot
mkdir -p work_dirs/cdfsod/vpe_baseline80/uodd_5shot && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29617 tools/train.py ../configs/cdfsod/uodd/5shot_vpe.py --work-dir work_dirs/cdfsod/vpe_baseline80/uodd_5shot --launcher pytorch > work_dirs/cdfsod/vpe_baseline80/uodd_5shot/nohup.log 2>&1 & echo "[vpe_baseline80 uodd 5shot] pid=$! log=work_dirs/cdfsod/vpe_baseline80/uodd_5shot/nohup.log"

# vpe_baseline80 / uodd / 10-shot
mkdir -p work_dirs/cdfsod/vpe_baseline80/uodd_10shot && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=29618 tools/train.py ../configs/cdfsod/uodd/10shot_vpe.py --work-dir work_dirs/cdfsod/vpe_baseline80/uodd_10shot --launcher pytorch > work_dirs/cdfsod/vpe_baseline80/uodd_10shot/nohup.log 2>&1 & echo "[vpe_baseline80 uodd 10shot] pid=$! log=work_dirs/cdfsod/vpe_baseline80/uodd_10shot/nohup.log"
