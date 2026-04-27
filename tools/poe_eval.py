#!/usr/bin/env python
"""COCO mAP evaluation for PoE prediction JSONs.

Wraps pycocotools to evaluate a results-format JSON (the kind written by
``tools/poe_inference.py``) against a COCO-style ground-truth annotation.
Reports the standard 12 COCO metrics plus per-category AP for a quick
diagnosis of which classes benefit / regress under PoE fusion.

Usage:
    python tools/poe_eval.py \\
        --gt   datasets/NEU-DET/annotations/test.json \\
        --pred work_dirs/cdfsod/poe/neu-det_5shot/predictions.json

If you also have the fine-tuned-only predictions (``--also-baseline`` of
``poe_inference.py``), pass it via ``--baseline-pred`` to print a side-by-
side diff:

    python tools/poe_eval.py \\
        --gt          datasets/NEU-DET/annotations/test.json \\
        --pred        work_dirs/.../predictions.json \\
        --baseline-pred work_dirs/.../predictions_ft_only.json
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gt', required=True,
                   help='COCO-style ground-truth annotation JSON.')
    p.add_argument('--pred', required=True,
                   help='Predictions JSON (results format).')
    p.add_argument('--baseline-pred', default=None,
                   help='Optional: fine-tuned-only predictions for diff.')
    p.add_argument('--per-class', action='store_true', default=True,
                   help='Print per-category AP table.')
    p.add_argument('--quiet-cocoeval', action='store_true', default=True,
                   help='Suppress pycocotools per-step log spam.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------
def evaluate(gt_path: str, pred_path: str, quiet: bool = True):
    """Run pycocotools COCOeval; return dict of metrics + per-class AP."""
    gt = COCO(gt_path)

    with open(pred_path) as f:
        preds = json.load(f)
    if len(preds) == 0:
        print(f'[eval] WARN: no predictions in {pred_path}')
        return None

    # pycocotools is verbose; mute its block-level chatter if requested.
    sink = io.StringIO() if quiet else sys.stdout
    with contextlib.redirect_stdout(sink):
        dt = gt.loadRes(pred_path)
        e  = COCOeval(gt, dt, 'bbox')
        e.evaluate()
        e.accumulate()
        e.summarize()

    # Standard 12 metrics, in the same order as e.summarize() prints.
    stats = {
        'AP':      e.stats[0],
        'AP50':    e.stats[1],
        'AP75':    e.stats[2],
        'AP_S':    e.stats[3],
        'AP_M':    e.stats[4],
        'AP_L':    e.stats[5],
        'AR_1':    e.stats[6],
        'AR_10':   e.stats[7],
        'AR_100':  e.stats[8],
        'AR_S':    e.stats[9],
        'AR_M':    e.stats[10],
        'AR_L':    e.stats[11],
    }

    # Per-class AP (averaged over IoU [.5:.95], all areas, max_det=100).
    # precision shape: [T, R, K, A, M] = [10, 101, K_classes, 4, 3]
    prec = e.eval['precision']     # may contain -1 for missing classes
    cat_ids = gt.getCatIds()
    cats    = gt.loadCats(cat_ids)
    per_class = {}
    for k, cat in enumerate(cats):
        cls_prec = prec[:, :, k, 0, 2]   # all IoUs, all rec, A=all, M=100
        ap = float(cls_prec[cls_prec > -1].mean()) if (cls_prec > -1).any() else 0.0
        per_class[cat['name']] = ap
    stats['per_class'] = per_class
    return stats


def fmt_pct(x: float) -> str:
    return f'{x * 100:6.2f}'


def print_metrics(name: str, m: dict):
    if m is None:
        return
    print(f'\n=== {name} ===')
    for k in ['AP', 'AP50', 'AP75', 'AP_S', 'AP_M', 'AP_L',
              'AR_1', 'AR_10', 'AR_100']:
        print(f'  {k:<7s} = {fmt_pct(m[k])}')
    if 'per_class' in m:
        print('  per-class AP:')
        for cls, ap in m['per_class'].items():
            print(f'    {cls:<24s} = {fmt_pct(ap)}')


def print_diff(baseline: dict, fused: dict):
    print('\n=== diff (PoE - baseline, %) ===')
    for k in ['AP', 'AP50', 'AP75']:
        delta = (fused[k] - baseline[k]) * 100
        sign = '+' if delta >= 0 else ''
        print(f'  {k:<7s}  {sign}{delta:6.2f}')
    print('  per-class AP delta:')
    for cls, ap_fused in fused.get('per_class', {}).items():
        ap_base = baseline['per_class'].get(cls, 0.0)
        delta = (ap_fused - ap_base) * 100
        sign = '+' if delta >= 0 else ''
        print(f'    {cls:<24s} '
              f'base={fmt_pct(ap_base)}  '
              f'PoE={fmt_pct(ap_fused)}  '
              f'delta={sign}{delta:6.2f}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if not Path(args.gt).exists():
        sys.exit(f'[eval] GT not found: {args.gt}')
    if not Path(args.pred).exists():
        sys.exit(f'[eval] pred not found: {args.pred}')

    fused = evaluate(args.gt, args.pred, quiet=args.quiet_cocoeval)
    print_metrics(f'PoE fused ({Path(args.pred).name})', fused)

    if args.baseline_pred:
        baseline = evaluate(args.gt, args.baseline_pred,
                            quiet=args.quiet_cocoeval)
        print_metrics(f'Fine-tuned only ({Path(args.baseline_pred).name})',
                      baseline)
        if baseline is not None and fused is not None:
            print_diff(baseline, fused)


if __name__ == '__main__':
    main()
