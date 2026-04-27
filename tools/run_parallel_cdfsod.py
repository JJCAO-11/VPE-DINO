"""Parallel CD-FSOD launcher: one experiment per GPU, queue the rest.

Designed for the case where you have a small fixed pool of GPUs (e.g.
cards 1,2,3,4) and many more experiments than GPUs. A worker thread is
spawned per GPU; each worker pulls the next experiment from a shared
FIFO queue, trains it on its dedicated GPU using single-GPU mmdet
training, and then picks the next one. All training stdout/stderr is
redirected to ``<work_dir>/train.log`` so this launcher can run safely
in the background.

Examples
--------
# default: cards 1,2,3,4, both variants, all 6 datasets x 3 shots
python tools/run_parallel_cdfsod.py

# only baseline variant on cards 1,2,3,4
python tools/run_parallel_cdfsod.py --variants baseline

# subset for a quick smoke test
python tools/run_parallel_cdfsod.py --datasets neu-det uodd --shots 5

# dump the equivalent shell commands without running them
python tools/run_parallel_cdfsod.py --print-only

# detached background launch (recommended on a remote server)
nohup python tools/run_parallel_cdfsod.py --gpus 1,2,3,4 \
    > parallel_runner.log 2>&1 &
tail -f parallel_runner.log
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'configs', 'cdfsod'))
import _dataset_meta as meta  # noqa: E402


# ---------------------------------------------------------------------
# Task assembly
# ---------------------------------------------------------------------
def make_tasks(variants, datasets, shots):
    tasks = []
    for variant in variants:
        for ds in datasets:
            for shot in shots:
                cfg_rel = f'../configs/cdfsod/{ds}/{shot}shot_{variant}.py'
                cfg_abs = os.path.join(
                    REPO, 'configs', 'cdfsod', ds,
                    f'{shot}shot_{variant}.py')
                if not os.path.exists(cfg_abs):
                    print(f'[skip] missing config: {cfg_abs}')
                    continue
                tasks.append(dict(
                    variant=variant,
                    dataset=ds,
                    shot=shot,
                    cfg=cfg_rel,
                    work_dir=f'work_dirs/cdfsod/{variant}/{ds}_{shot}shot',
                ))
    return tasks


def task_tag(t):
    return f"{t['variant']:8s} {t['dataset']:9s} {t['shot']:>2d}-shot"


# ---------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------
def worker(gpu_id, task_q, results, log_lock, dry_run=False):
    while True:
        try:
            task = task_q.get_nowait()
        except queue.Empty:
            return

        tag = task_tag(task)
        wd_rel = task['work_dir']
        wd_abs = os.path.join(REPO, 'mmdetection', wd_rel)
        os.makedirs(wd_abs, exist_ok=True)
        log_file = os.path.join(wd_abs, 'train.log')

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        cmd = ['python', 'tools/train.py', task['cfg'],
               '--work-dir', wd_rel]

        if dry_run:
            with log_lock:
                print(f"# [gpu{gpu_id}] {tag}")
                print(f"CUDA_VISIBLE_DEVICES={gpu_id} \\")
                print(f"  nohup {' '.join(cmd)} \\")
                print(f"  > {os.path.relpath(log_file, REPO)} 2>&1 &")
                print()
            results.append((task, 0, 0.0))
            task_q.task_done()
            continue

        with log_lock:
            print(f"[gpu{gpu_id}] >> {datetime.now():%H:%M:%S} START "
                  f"{tag}  -> {os.path.relpath(log_file, REPO)}")

        t0 = time.time()
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"# Launched {datetime.now()}\n")
                f.write(f"# CUDA_VISIBLE_DEVICES={gpu_id}\n")
                f.write(f"# Command: {' '.join(cmd)}\n")
                f.write(f"# CWD: {os.path.join(REPO, 'mmdetection')}\n\n")
                f.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=os.path.join(REPO, 'mmdetection'),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env)
                ret = proc.wait()
        except Exception as e:
            ret = -1
            with log_lock:
                print(f"[gpu{gpu_id}] !! exception while running {tag}: {e}")
        elapsed = time.time() - t0

        status = 'OK' if ret == 0 else f'FAIL({ret})'
        with log_lock:
            print(f"[gpu{gpu_id}] << {datetime.now():%H:%M:%S} {status:<8s} "
                  f"{tag}  ({elapsed / 60:.1f} min)")

        results.append((task, ret, elapsed))
        task_q.task_done()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument('--gpus', default='1,2,3,4',
                    help='comma-separated GPU ids (default: 1,2,3,4)')
    ap.add_argument('--variants', nargs='+',
                    default=['baseline', 'ets'],
                    choices=['baseline', 'ets'])
    ap.add_argument('--datasets', nargs='+', default=meta.DATASET_ORDER,
                    help=f'(default: {meta.DATASET_ORDER})')
    ap.add_argument('--shots', nargs='+', type=int,
                    default=list(meta.SHOTS),
                    help='(default: 1 5 10)')
    ap.add_argument('--print-only', action='store_true',
                    help='print equivalent shell commands and exit '
                         '(no training)')
    args = ap.parse_args()

    gpus = [int(g.strip()) for g in args.gpus.split(',') if g.strip()]
    tasks = make_tasks(args.variants, args.datasets, args.shots)

    print('=' * 64)
    print('  Parallel CD-FSOD Runner')
    print('-' * 64)
    print(f'  GPUs        : {gpus}')
    print(f'  Variants    : {args.variants}')
    print(f'  Datasets    : {args.datasets}')
    print(f'  Shots       : {args.shots}')
    print(f'  Total tasks : {len(tasks)}')
    print(f'  Started     : {datetime.now()}')
    if args.print_only:
        print(f'  MODE        : --print-only (no training)')
    print('=' * 64)
    print()

    task_q = queue.Queue()
    for t in tasks:
        task_q.put(t)

    results = []
    log_lock = threading.Lock()

    threads = []
    for gpu in gpus:
        th = threading.Thread(
            target=worker,
            args=(gpu, task_q, results, log_lock, args.print_only),
            daemon=False,
            name=f'gpu{gpu}')
        th.start()
        threads.append(th)

    for th in threads:
        th.join()

    if args.print_only:
        return

    print()
    print('=' * 64)
    print(f'  All done at {datetime.now()}')
    print('-' * 64)
    n_ok = sum(1 for _, ret, _ in results if ret == 0)
    print(f'  OK    : {n_ok}/{len(results)}')
    print(f'  FAIL  : {len(results) - n_ok}/{len(results)}')
    if results:
        total_min = sum(e for _, _, e in results) / 60
        print(f'  Total GPU-time : {total_min:.1f} min '
              f'({total_min / 60:.2f} hours)')
    print()
    print('  Aggregate results:')
    print('    python summarize_results.py')
    print('=' * 64)


if __name__ == '__main__':
    main()
