#!/usr/bin/env python
"""Batch runner: PoE inference + evaluation across all CD-FSOD experiments.

Walks every (dataset, shot, variant) combo for which a fine-tuned ckpt
exists under ``mmdetection/work_dirs/cdfsod/<variant>/<dataset>_<shot>shot/``
and:
   1. invokes ``tools/poe_inference.py`` to produce fused predictions
   2. invokes ``tools/poe_eval.py`` to compute mAP
   3. records the AP / AP50 / per-class AP into a CSV + Markdown table

Output goes to ``work_dirs/cdfsod/poe/<variant>/<dataset>_<shot>shot/`` with
a top-level ``poe_summary.{csv,md}`` aggregator.

Usage (run from ETS repo root):

    # everything (all 18 / 36 fine-tuned ckpts found):
    python tools/poe_run_all.py

    # one variant only:
    python tools/poe_run_all.py --variants ets

    # alpha sweep on a small subset:
    python tools/poe_run_all.py --datasets neu-det --shots 5 \\
        --alphas 0.1 0.3 0.5 0.7 --also-baseline

    # dry-run (print commands, do nothing):
    python tools/poe_run_all.py --print-only
"""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Inject repo root for the sibling import -----------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'configs' / 'cdfsod'))

# Single source of truth for dataset metadata ------------------------------
from _dataset_meta import (  # noqa: E402
    DATASETS, DATASET_ORDER, SHOTS, get_pre_aliases,
)

PRETRAINED_URL = (
    'https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/'
    'groundingdino_swinb_cogcoor_mmdet-55949c9c.pth'
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--variants', nargs='+',
                   default=['baseline', 'ets'],
                   choices=['baseline', 'ets'])
    p.add_argument('--datasets', nargs='+', default=DATASET_ORDER,
                   help='Subset of datasets to run.')
    p.add_argument('--shots', nargs='+', type=int, default=list(SHOTS))
    p.add_argument('--alphas', nargs='+', type=float, default=[0.3],
                   help='PoE weights to sweep (one experiment per alpha).')
    p.add_argument('--iou-thr', type=float, default=0.3)
    p.add_argument('--mode', default='log_linear',
                   choices=['log_linear', 'multiplicative',
                            'additive', 'gated'])
    p.add_argument('--pre-ckpt', default=PRETRAINED_URL,
                   help='Pretrained ckpt URL or local path.')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--also-baseline', action='store_true',
                   help='Also dump fine-tuned-only predictions (for diff).')
    p.add_argument('--limit', type=int, default=-1,
                   help='Limit images per experiment (for quick smoke test).')
    p.add_argument('--out-root', default='work_dirs/cdfsod/poe',
                   help='Where to put per-experiment predictions and summary.')
    p.add_argument('--print-only', action='store_true',
                   help='Print commands without executing them.')
    p.add_argument('--skip-existing', action='store_true', default=True,
                   help='Skip if predictions JSON already exists.')
    p.add_argument('--rerun', action='store_false', dest='skip_existing',
                   help='Force re-run even if predictions exist.')
    p.add_argument('--resummarize-only', action='store_true',
                   help='Skip inference + eval; just re-parse cached eval.log '
                        'files under --out-root and rewrite poe_summary.{csv,md}.')
    p.add_argument('--no-pre-aliases', action='store_true',
                   help='Disable the pretrained-expert alias prompt. The '
                        'pretrained model is queried with the original '
                        'class names (legacy behaviour, matches the '
                        'pre-bugfix runs).')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def find_ft_ckpt(variant: str, dataset: str, shot: int) -> Optional[str]:
    """Locate the BEST fine-tuned checkpoint, if available.

    Search order in ``mmdetection/work_dirs/cdfsod/<variant>/<dataset>_<shot>shot/``:
      1. best_coco_bbox_mAP_*.pth (mmdet's `save_best='coco/bbox_mAP'`)
      2. best_*.pth (any other best metric)
      3. last (highest) epoch_*.pth
    """
    wd = REPO_ROOT / 'mmdetection' / 'work_dirs' / 'cdfsod' / variant / \
         f'{dataset}_{shot}shot'
    for pat in ('best_coco_bbox_mAP_*.pth', 'best_*.pth', 'epoch_*.pth'):
        cands = sorted(wd.glob(pat))
        if cands:
            return str(cands[-1])
    return None


def find_leaf_config(variant: str, dataset: str, shot: int) -> Optional[str]:
    """Return path to the auto-generated leaf config."""
    cfg = REPO_ROOT / 'configs' / 'cdfsod' / dataset / f'{shot}shot_{variant}.py'
    return str(cfg) if cfg.exists() else None


def find_test_data(dataset: str) -> Optional[Tuple[str, str]]:
    """Return (test_ann_path, img_prefix) or None if missing."""
    meta = DATASETS[dataset]
    # data_root in meta is relative to mmdetection/  ->  resolve to repo root.
    rel = meta['data_root'].lstrip('./').replace('../', '')
    root = REPO_ROOT / rel
    ann = root / 'annotations' / 'test.json'
    img = root / 'test'
    if ann.exists() and img.exists():
        return str(ann), str(img)
    return None


# ---------------------------------------------------------------------------
# Exec helpers
# ---------------------------------------------------------------------------
def run_cmd(cmd: List[str], print_only: bool) -> int:
    """Run a subprocess; return exit code."""
    pretty = ' '.join(shlex.quote(c) for c in cmd)
    print(f'\n$ {pretty}\n')
    if print_only:
        return 0
    return subprocess.call(cmd)


_METRIC_KEYS = ('AP', 'AP50', 'AP75', 'AP_S', 'AP_M', 'AP_L',
                'AR_1', 'AR_10', 'AR_100')
# Anchor at start-of-line + word boundary so 'AP' does NOT match 'AP_L' etc.
_METRIC_RE = re.compile(
    r'^(?P<k>' + '|'.join(_METRIC_KEYS) + r')\s*=\s*(?P<v>-?\d+(?:\.\d+)?)\s*$')


def parse_eval_log(log_text: str) -> dict:
    """Pull the metrics dict out of `tools/poe_eval.py` stdout.

    The eval script prints two blocks when --baseline-pred is given:
        === PoE fused (predictions.json) ===
            AP = 50.47 ...
        === Fine-tuned only (predictions_ft_only.json) ===
            AP = 51.25 ...

    We only want the FIRST (fused) block. Bug history: the previous version
    used ``line.startswith('AP')`` which (1) also matched ``AP_L`` lines, and
    (2) walked into the baseline block, so the stored 'AP' was actually the
    baseline 'AP_L'. Both fixed below.
    """
    out: dict = {}
    in_fused = False
    fused_done = False
    for raw in log_text.splitlines():
        stripped = raw.strip()
        if stripped.startswith('==='):
            # Block boundary. Only the very first '=== PoE fused ===' block
            # is treated as the source of truth.
            if not fused_done and 'PoE fused' in stripped:
                in_fused = True
            else:
                if in_fused:
                    fused_done = True
                in_fused = False
            continue
        if not in_fused:
            continue
        m = _METRIC_RE.match(stripped)
        if m:
            try:
                out[m.group('k')] = float(m.group('v')) / 100.0
            except ValueError:
                pass
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def _write_summary(out_root: Path, summary_rows: List[dict]) -> None:
    """Persist summary_rows to poe_summary.{csv,md} under ``out_root``."""
    csv_path = out_root / 'poe_summary.csv'
    md_path = out_root / 'poe_summary.md'

    fields = ['variant', 'dataset', 'shot', 'alpha', 'mode', 'aliased',
              'AP', 'AP50', 'AP75', 'AR_100', 'pred_json']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k, '') for k in fields})
    print(f'[csv] {csv_path}')

    md_lines = ['# PoE Fusion Results', '',
                '| variant | dataset | shot | alpha | mode | aliased '
                '| AP | AP50 | AP75 | AR100 |',
                '|---|---|---|---|---|---|---|---|---|---|']

    def fmt(x):
        return f'{x * 100:.2f}' if isinstance(x, (int, float)) else '-'

    for row in summary_rows:
        md_lines.append(
            f"| {row['variant']} | {row['dataset']} | {row['shot']}-shot | "
            f"{row['alpha']:g} | {row['mode']} | "
            f"{'yes' if row.get('aliased') else 'no'} | "
            f"{fmt(row['AP'])} | "
            f"{fmt(row['AP50'])} | {fmt(row['AP75'])} | {fmt(row['AR_100'])} |"
        )
    md_path.write_text('\n'.join(md_lines))
    print(f'[md]  {md_path}')


def collect_cached_rows(out_root: Path) -> List[dict]:
    """Walk ``out_root`` and rebuild summary_rows from each ``eval.log``.

    Directory layout produced by the runner:
        out_root/<variant>/<dataset>_<shot>shot/alpha<alpha>_<mode>[_paliased]/
            eval.log

    The optional ``_paliased`` suffix (added when the pretrained expert
    uses dataset-specific English aliases) is split into a separate
    ``aliased`` boolean column so the summary stays human-readable.
    """
    rows: List[dict] = []
    for log_path in sorted(out_root.glob('*/*/alpha*_*/eval.log')):
        # parts relative to out_root
        rel = log_path.relative_to(out_root).parts
        if len(rel) < 4:
            continue
        variant, ds_shot, alpha_mode, _ = rel[0], rel[1], rel[2], rel[3]
        if '_' not in ds_shot or 'shot' not in ds_shot:
            continue
        dataset, _, shot_tok = ds_shot.rpartition('_')
        try:
            shot = int(shot_tok.replace('shot', ''))
        except ValueError:
            continue
        if not alpha_mode.startswith('alpha'):
            continue
        rest = alpha_mode[len('alpha'):]
        # Strip optional alias-prompt suffix BEFORE splitting alpha/mode.
        aliased = rest.endswith('_paliased')
        if aliased:
            rest = rest[:-len('_paliased')]
        try:
            alpha_str, mode = rest.split('_', 1)
            alpha = float(alpha_str)
        except ValueError:
            continue

        try:
            metrics = parse_eval_log(log_path.read_text())
        except OSError:
            continue
        pred_json = log_path.parent / 'predictions.json'
        rows.append(dict(
            variant=variant, dataset=dataset, shot=shot,
            alpha=alpha, mode=mode, aliased=aliased,
            AP=metrics.get('AP'),
            AP50=metrics.get('AP50'),
            AP75=metrics.get('AP75'),
            AR_100=metrics.get('AR_100'),
            pred_json=str(pred_json.relative_to(REPO_ROOT)),
        ))
    return rows


def main():
    args = parse_args()

    out_root = REPO_ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    # Fast path: rebuild summary from cached eval.log (no inference, no eval) ----
    if args.resummarize_only:
        summary_rows = collect_cached_rows(out_root)
        if not summary_rows:
            print(f'[warn] no eval.log found under {out_root}')
            return
        print(f'[resummarize] rebuilt {len(summary_rows)} rows from cached logs')
        _write_summary(out_root, summary_rows)
        return

    for variant in args.variants:
        for dataset in args.datasets:
            if dataset not in DATASETS:
                print(f'[skip] unknown dataset: {dataset}')
                continue
            for shot in args.shots:
                ft_ckpt = find_ft_ckpt(variant, dataset, shot)
                cfg = find_leaf_config(variant, dataset, shot)
                test_data = find_test_data(dataset)

                tag = f'{variant}/{dataset}_{shot}shot'
                if ft_ckpt is None:
                    print(f'[skip] no ft ckpt for {tag}')
                    continue
                if cfg is None:
                    print(f'[skip] no leaf config for {tag}')
                    continue
                if test_data is None:
                    print(f'[skip] no test annotation/img dir for {tag}')
                    continue

                test_ann, img_prefix = test_data

                # Decide whether the pretrained expert uses an alias
                # prompt for this dataset.  ``--no-pre-aliases`` overrides
                # the meta lookup.  When aliases are active we suffix the
                # experiment dir with ``_paliased`` so the legacy outputs
                # (without aliases) are NOT overwritten.
                aliases = None if args.no_pre_aliases else get_pre_aliases(dataset)
                alias_suffix = '_paliased' if aliases else ''

                for alpha in args.alphas:
                    # Output dir per (variant, dataset, shot, alpha) ---------
                    sub = (f'{dataset}_{shot}shot/'
                           f'alpha{alpha:g}_{args.mode}{alias_suffix}')
                    exp_dir = out_root / variant / sub
                    exp_dir.mkdir(parents=True, exist_ok=True)

                    pred_json = exp_dir / 'predictions.json'
                    eval_log  = exp_dir / 'eval.log'

                    # Skip if cached --------------------------------------
                    if args.skip_existing and pred_json.exists():
                        print(f'[cached] {tag} alpha={alpha} '
                              f'({pred_json.relative_to(REPO_ROOT)})')
                    else:
                        # ----- 1. PoE inference -----------------------------
                        infer_cmd = [
                            sys.executable, str(REPO_ROOT / 'tools' / 'poe_inference.py'),
                            '--ft-config',  cfg,
                            '--ft-ckpt',    ft_ckpt,
                            '--pre-config', cfg,
                            '--pre-ckpt',   args.pre_ckpt,
                            '--test-ann',   test_ann,
                            '--img-prefix', img_prefix,
                            '--alpha',      str(alpha),
                            '--iou-thr',    str(args.iou_thr),
                            '--mode',       args.mode,
                            '--device',     args.device,
                            '--out',        str(pred_json),
                            '--dataset',    dataset,
                        ]
                        if args.no_pre_aliases:
                            infer_cmd.append('--no-pre-aliases')
                        if args.also_baseline:
                            infer_cmd.append('--also-baseline')
                        if args.limit > 0:
                            infer_cmd.extend(['--limit', str(args.limit)])

                        rc = run_cmd(infer_cmd, args.print_only)
                        if rc != 0:
                            print(f'[err] inference failed ({rc}) for {tag}')
                            continue

                    # ----- 2. eval --------------------------------------
                    eval_cmd = [
                        sys.executable, str(REPO_ROOT / 'tools' / 'poe_eval.py'),
                        '--gt',   test_ann,
                        '--pred', str(pred_json),
                    ]
                    if args.also_baseline:
                        bl_json = pred_json.with_name(pred_json.stem + '_ft_only.json')
                        if bl_json.exists():
                            eval_cmd.extend(['--baseline-pred', str(bl_json)])

                    if args.print_only:
                        run_cmd(eval_cmd, print_only=True)
                        continue

                    # Capture stdout so we can parse + persist.
                    pretty = ' '.join(shlex.quote(c) for c in eval_cmd)
                    print(f'\n$ {pretty}\n')
                    proc = subprocess.run(eval_cmd, capture_output=True, text=True)
                    eval_log.write_text(proc.stdout + '\n' + proc.stderr)
                    print(proc.stdout)

                    metrics = parse_eval_log(proc.stdout)
                    summary_rows.append(dict(
                        variant=variant, dataset=dataset, shot=shot,
                        alpha=alpha, mode=args.mode,
                        aliased=bool(aliases),
                        AP=metrics.get('AP'),
                        AP50=metrics.get('AP50'),
                        AP75=metrics.get('AP75'),
                        AR_100=metrics.get('AR_100'),
                        pred_json=str(pred_json.relative_to(REPO_ROOT)),
                    ))

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    if not summary_rows or args.print_only:
        return
    _write_summary(out_root, summary_rows)


if __name__ == '__main__':
    main()
