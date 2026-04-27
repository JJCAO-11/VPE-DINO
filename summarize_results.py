"""Aggregate CD-FSOD benchmark results across all (variant, dataset, shot)
combinations and emit two artefacts:

    cdfsod_results.csv        : one row per experiment, easy to import
    cdfsod_results.md         : markdown comparison tables (per variant)

The script inspects ``mmdetection/work_dirs/cdfsod/<variant>/<dataset>_<shot>shot/``
created by ``main_results_cdfsod.sh`` and parses the latest mmengine
JSON log produced by mmdet 3.x.  It records:

    - best  bbox_mAP / mAP_50 / mAP_75 across all val/test evaluations
    - the epoch at which the best mAP was achieved

Usage (from repo root):

    python summarize_results.py
    python summarize_results.py --root mmdetection/work_dirs/cdfsod
"""
import argparse
import csv
import json
import os
import re
from glob import glob
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'configs', 'cdfsod'))
import _dataset_meta as meta  # noqa: E402

VARIANTS = ('baseline', 'ets', 'vpe_baseline80')
METRIC_KEYS = ('coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75')


def find_latest_jsonlog(exp_dir: str) -> Optional[str]:
    """Return the most recent ``vis_data/scalars.json`` under exp_dir.

    mmengine 3.x writes one ``<timestamp>/vis_data/scalars.json`` per
    training run; we pick the lexicographically largest (newest)."""
    candidates = sorted(
        glob(os.path.join(exp_dir, '*', 'vis_data', 'scalars.json')))
    return candidates[-1] if candidates else None


def parse_jsonlog(path: str) -> Dict[str, float]:
    """Scan an mmengine scalars.json and return the best metric dict."""
    best = {k: float('nan') for k in METRIC_KEYS}
    best_epoch = -1
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'coco/bbox_mAP' not in rec:
                continue
            current = rec['coco/bbox_mAP']
            best_mAP = best['coco/bbox_mAP']
            if (best_mAP != best_mAP) or (current > best_mAP):  # NaN-safe
                for k in METRIC_KEYS:
                    if k in rec:
                        best[k] = rec[k]
                best_epoch = rec.get('epoch', best_epoch)
    best['best_epoch'] = best_epoch
    return best


def collect(root: str) -> List[Dict]:
    rows: List[Dict] = []
    for variant in VARIANTS:
        for ds_key in meta.DATASET_ORDER:
            for shot in meta.SHOTS:
                exp_dir = os.path.join(
                    root, variant, f'{ds_key}_{shot}shot')
                row = dict(
                    variant=variant,
                    dataset=meta.DATASETS[ds_key]['name'],
                    dataset_key=ds_key,
                    shot=shot,
                    bbox_mAP=float('nan'),
                    bbox_mAP_50=float('nan'),
                    bbox_mAP_75=float('nan'),
                    best_epoch=-1,
                    log_path='',
                )
                if not os.path.isdir(exp_dir):
                    rows.append(row)
                    continue
                log = find_latest_jsonlog(exp_dir)
                if log is None:
                    rows.append(row)
                    continue
                metrics = parse_jsonlog(log)
                row.update(
                    bbox_mAP=metrics['coco/bbox_mAP'],
                    bbox_mAP_50=metrics['coco/bbox_mAP_50'],
                    bbox_mAP_75=metrics['coco/bbox_mAP_75'],
                    best_epoch=metrics['best_epoch'],
                    log_path=os.path.relpath(log),
                )
                rows.append(row)
    return rows


def write_csv(rows: List[Dict], path: str) -> None:
    fieldnames = [
        'variant', 'dataset', 'dataset_key', 'shot',
        'bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75',
        'best_epoch', 'log_path',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'[csv ] wrote {path}')


def fmt(v: float) -> str:
    return '-' if v != v else f'{v * 100:.1f}'  # NaN check


def write_markdown(rows: List[Dict], path: str) -> None:
    """Emit a markdown report shaped like CDFSOD-benchmark Tab.2."""
    by_variant: Dict[str, Dict[Tuple[str, int], Dict]] = {}
    for r in rows:
        by_variant.setdefault(r['variant'], {})[
            (r['dataset_key'], r['shot'])] = r

    with open(path, 'w', encoding='utf-8') as f:
        f.write('# ETS on CD-FSOD Benchmark - Results Summary\n\n')
        f.write('Numbers are COCO `bbox_mAP` (x100). '
                '"-" = experiment not yet run / no log found.\n\n')

        for variant in VARIANTS:
            if variant not in by_variant:
                continue
            f.write(f'## Variant: `{variant}`\n\n')
            f.write('| Dataset | 1-shot | 5-shot | 10-shot |\n')
            f.write('|---|---:|---:|---:|\n')
            for ds_key in meta.DATASET_ORDER:
                ds_name = meta.DATASETS[ds_key]['name']
                cells = []
                for shot in meta.SHOTS:
                    r = by_variant[variant].get(
                        (ds_key, shot),
                        dict(bbox_mAP=float('nan')))
                    cells.append(fmt(r['bbox_mAP']))
                f.write(f'| {ds_name} | {cells[0]} | {cells[1]} '
                        f'| {cells[2]} |\n')

            # Mean row
            means = []
            for shot in meta.SHOTS:
                vals = [by_variant[variant].get((k, shot), {}).get(
                    'bbox_mAP', float('nan'))
                        for k in meta.DATASET_ORDER]
                vals = [v for v in vals if v == v]
                means.append(
                    fmt(sum(vals) / len(vals)) if vals else '-')
            f.write(f'| **Mean** | **{means[0]}** | **{means[1]}** '
                    f'| **{means[2]}** |\n\n')

        f.write('\n---\n')
        f.write('Generated by `summarize_results.py`. '
                'Re-run after each new experiment finishes.\n')
    print(f'[md  ] wrote {path}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--root', default='mmdetection/work_dirs/cdfsod',
        help='Directory holding <variant>/<dataset>_<shot>shot/ runs')
    ap.add_argument(
        '--csv-out', default='cdfsod_results.csv')
    ap.add_argument(
        '--md-out', default='cdfsod_results.md')
    args = ap.parse_args()

    rows = collect(args.root)
    n_done = sum(1 for r in rows if r['bbox_mAP'] == r['bbox_mAP'])
    print(f'[scan] found metrics for {n_done}/{len(rows)} experiments '
          f'under {args.root}')
    write_csv(rows, args.csv_out)
    write_markdown(rows, args.md_out)


if __name__ == '__main__':
    main()
