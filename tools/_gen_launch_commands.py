"""One-off helper: dump all (3 variants x 6 ds x 3 shots = 54)
background-launchable bash commands into ``launch_cdfsod_commands.sh``.

Mode: each experiment uses ALL 4 GPUs (0,1,2,3) via
``torch.distributed.launch``, and the experiments are launched
SEQUENTIALLY (one at a time). Each command uses a unique master_port
so accidental overlap doesn't conflict.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'configs', 'cdfsod'))
import _dataset_meta as m  # noqa: E402

GPU_IDS = '0,1,2,3'
NUM_GPUS = 4
BASE_PORT = 29501
VARIANTS = (
    ('baseline', 'baseline'),
    ('ets', 'ets'),
    ('vpe_baseline80', 'vpe'),
)

HEADER = """#!/usr/bin/env bash
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
"""


def main():
    out_path = os.path.join(REPO, 'launch_cdfsod_commands.sh')
    i = 0
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(HEADER)
        for variant, cfg_variant in VARIANTS:
            f.write('\n# ============================================================\n')
            f.write(f'# Variant: {variant}\n')
            f.write('# ============================================================\n')
            for ds in m.DATASET_ORDER:
                for shot in m.SHOTS:
                    cfg = (
                        f'../configs/cdfsod/{ds}/'
                        f'{shot}shot_{cfg_variant}.py')
                    wd = f'work_dirs/cdfsod/{variant}/{ds}_{shot}shot'
                    log = f'{wd}/train.log'
                    port = BASE_PORT + i
                    f.write(
                        f'\n# [{i + 1:02d}/54]  {variant} / {ds} / '
                        f'{shot}-shot  (port {port})\n')
                    f.write(
                        f'mkdir -p {wd} && '
                        f'CUDA_VISIBLE_DEVICES={GPU_IDS} '
                        f'python -m torch.distributed.launch '
                        f'--nnodes=1 --node_rank=0 --master_addr=127.0.0.1 '
                        f'--nproc_per_node={NUM_GPUS} --master_port={port} '
                        f'tools/train.py {cfg} '
                        f'--work-dir {wd} --launcher pytorch '
                        f'2>&1 | tee {log}\n')
                    i += 1
    print(f'wrote {out_path} ({i} commands)')


if __name__ == '__main__':
    main()
