"""Helper: dump per-experiment nohup launch commands.

Each command:
  - uses ALL 4 GPUs (0,1,2,3)
  - runs in background via nohup
  - writes its own log to work_dirs/.../nohup.log
  - has a unique master_port

You run ONE at a time, wait for it to finish (check with `ps` / log tail),
then start the next.
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

HEADER = """#!/usr/bin/env bash
# ============================================================
# CD-FSOD per-experiment nohup launchers (4 GPUs each).
# ------------------------------------------------------------
# RUN FROM: ETS/mmdetection/   (cd into mmdetection first!)
#
# Usage
# -----
# 1. Pick ONE block below, copy-paste into your shell.
# 2. It will start in the background. The shell prints its PID.
# 3. Monitor progress:
#       tail -f work_dirs/cdfsod/<variant>/<ds>_<shot>shot/nohup.log
# 4. Wait for completion:
#       wait <pid>            # blocks until that pid exits
#    or check periodically:
#       ps -p <pid>            # empty output = finished
# 5. Once finished, fire the NEXT block.
#
# Quick "is anything still training?" check:
#       ps aux | grep '[t]ools/train.py'
#       nvidia-smi
#
# After all experiments finish, aggregate:
#       cd <repo_root> && python summarize_results.py
# ============================================================
"""


def main():
    out_path = os.path.join(REPO, 'launch_nohup_individual.sh')
    i = 0
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(HEADER)
        for variant in ('baseline', 'ets', 'vpe'):
            f.write('\n# ============================================================\n')
            f.write(f'# Variant: {variant}\n')
            f.write('# ============================================================\n')
            for ds in m.DATASET_ORDER:
                for shot in m.SHOTS:
                    cfg = f'../configs/cdfsod/{ds}/{shot}shot_{variant}.py'
                    wd = f'work_dirs/cdfsod/{variant}/{ds}_{shot}shot'
                    log = f'{wd}/nohup.log'
                    port = BASE_PORT + i
                    f.write(
                        f'\n# [{i + 1:02d}/54]  {variant} / {ds} / '
                        f'{shot}-shot  (port {port})\n')
                    # single-line command for easy copy-paste
                    f.write(
                        f'mkdir -p {wd} && '
                        f'CUDA_VISIBLE_DEVICES={GPU_IDS} '
                        f'nohup python -m torch.distributed.launch '
                        f'--nnodes=1 --node_rank=0 --master_addr=127.0.0.1 '
                        f'--nproc_per_node={NUM_GPUS} --master_port={port} '
                        f'tools/train.py {cfg} '
                        f'--work-dir {wd} --launcher pytorch '
                        f'> {log} 2>&1 & '
                        f'echo "[{i + 1:02d}/54] started pid=$! log={log}"\n')
                    i += 1
    print(f'wrote {out_path} ({i} commands)')


if __name__ == '__main__':
    main()
