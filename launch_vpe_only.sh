#!/usr/bin/env bash
echo "Use mmdetection/launch_vpe_baseline80_nohup_individual.sh from ETS/mmdetection."
exit 1
set -e
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
