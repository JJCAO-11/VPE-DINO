#!/usr/bin/env python
"""PoE (Product-of-Experts) Inference for CD-FSOD.

Adapted from Ong et al., "Fixing Background Misclassification in Few-Shot
Object Detection via Product of Experts" (TPAMI 2026), this script
combines two GroundingDINO checkpoints at INFERENCE time:

  - Pretrained GroundingDINO Swin-B (Objects365 + GoldG) as the "base
    expert" -- never seen the target domain.
  - Fine-tuned ETS GroundingDINO (after K-shot fine-tuning on the
    target dataset) as the "novel expert".

For every detection produced by the fine-tuned model, we look up the
strongest pretrained detection at the SAME spatial location (by IoU,
regardless of label).  This pretrained score acts as an "objectness /
known-pattern" signal.  The two scores are fused via a Product-of-Experts
rule (log-linear pooling):

    log s_fused = (1 - alpha) * log s_ft  +  alpha * log s_pre

which is mathematically equivalent to multiplying probabilities raised to
weights -- the standard PoE form.  We support 4 fusion modes:

    log_linear      : True PoE.  log s = (1-a) log s_ft + a log s_pre.
    multiplicative  : Boost mode.  s_ft * (1 + a * s_pre)  (clipped to 1).
    additive        : Convex combo.  (1-a) s_ft + a s_pre.
    gated           : s_ft + a * 1[s_pre>tau] * s_pre.

The script outputs predictions in COCO results-format JSON, ready to be
fed into pycocotools / `tools/poe_eval.py`.

Usage (run from the ETS repo root):

    python tools/poe_inference.py \\
        --ft-config  configs/cdfsod/neu-det/5shot_ets.py \\
        --ft-ckpt   "mmdetection/work_dirs/cdfsod/ets/neu-det_5shot/best_*.pth" \\
        --pre-ckpt  "https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth" \\
        --test-ann   datasets/NEU-DET/annotations/test.json \\
        --img-prefix datasets/NEU-DET/test \\
        --alpha      0.3 \\
        --iou-thr    0.3 \\
        --mode       log_linear \\
        --out        work_dirs/cdfsod/poe/neu-det_5shot/predictions.json

The fine-tuned ckpt path may be a glob (e.g. ``best_*.pth``); the latest
match (alphabetically) is used.  The pretrained ckpt may be a URL OR a
local path; mmdet's loader handles both.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torchvision.ops import box_iou
from tqdm import tqdm

from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config


# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

# Make `_dataset_meta` (under configs/cdfsod/) importable so we can look up
# pretrained-expert alias prompts by dataset name.
sys.path.insert(0, str(REPO_ROOT / 'configs' / 'cdfsod'))
try:
    from _dataset_meta import get_pre_aliases  # type: ignore
except ImportError:  # pragma: no cover - meta is part of the repo
    def get_pre_aliases(_dataset):  # type: ignore[no-redef]
        return None

DEFAULT_PRETRAINED_URL = (
    'https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/'
    'groundingdino_swinb_cogcoor_mmdet-55949c9c.pth'
)


def _patch_lang_model_path(cfg: Config) -> Config:
    """Resolve relative BERT paths to absolute ones.

    Training configs (e.g. ``mmdetection/configs/grounding_dino/CDFSOD/
    GroudingDINO-few-shot.py``) use ``lang_model_name='../weights/bert-
    base-uncased'``.  This is relative to ``mmdetection/`` (where training
    runs).  PoE inference runs from the repo root, so the relative path
    breaks unless we rewrite it.
    """
    try:
        bert_name = cfg.model.language_model.name
    except (AttributeError, KeyError):
        return cfg
    if not isinstance(bert_name, str):
        return cfg
    if (os.path.isabs(bert_name)
            or bert_name.startswith(('http://', 'https://'))
            or '/' not in bert_name):
        # Already absolute, URL, or bare HF repo id (e.g. 'bert-base-uncased')
        return cfg
    candidate = (REPO_ROOT / 'mmdetection' / bert_name).resolve()
    if candidate.exists():
        cfg.model.language_model.name = str(candidate)
        print(f'[poe] patched BERT path: {bert_name!r} -> {candidate}')
    else:
        print(f'[poe][warn] BERT path {bert_name!r} not absolute and not '
              f'found at {candidate}; leaving as-is.')
    return cfg


def _build_detector(config_path: str, ckpt: str, device: str):
    """Load config, patch BERT path, then init detector."""
    cfg = Config.fromfile(config_path)
    cfg = _patch_lang_model_path(cfg)
    return init_detector(cfg, ckpt, device=device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Models ----------------------------------------------------------------
    p.add_argument('--ft-config', required=True,
                   help='Fine-tuned model config (e.g. configs/cdfsod/'
                        'neu-det/5shot_ets.py).')
    p.add_argument('--ft-ckpt', required=True,
                   help='Fine-tuned checkpoint path (glob OK).')
    p.add_argument('--pre-config', default=None,
                   help='Pretrained model config; defaults to --ft-config '
                        '(architecture is the same).')
    p.add_argument('--pre-ckpt', default=DEFAULT_PRETRAINED_URL,
                   help='Pretrained checkpoint URL or local path.  '
                        f'Default: {DEFAULT_PRETRAINED_URL}')
    # Data ------------------------------------------------------------------
    p.add_argument('--test-ann', required=True,
                   help='Test annotation file (COCO format).')
    p.add_argument('--img-prefix', required=True,
                   help='Directory containing test images.')
    p.add_argument('--out', required=True,
                   help='Output predictions JSON.')
    # Fusion params --------------------------------------------------------
    p.add_argument('--alpha', type=float, default=0.3,
                   help='PoE weight on pretrained score (0..1).  Higher = '
                        'rely more on pretrained "objectness".  Default 0.3.')
    p.add_argument('--iou-thr', type=float, default=0.3,
                   help='IoU threshold for matching ft<->pre boxes.')
    p.add_argument('--mode', default='log_linear',
                   choices=['log_linear', 'multiplicative',
                            'additive', 'gated'],
                   help='Fusion mode.  log_linear is the true PoE form.')
    p.add_argument('--gate-thr', type=float, default=0.05,
                   help='In gated mode: only fuse when pre score > this.')
    p.add_argument('--eps', type=float, default=1e-3,
                   help='Floor for log-stability.')
    # Misc -----------------------------------------------------------------
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--score-thr', type=float, default=0.001,
                   help='Drop predictions below this score (post-fusion).')
    p.add_argument('--max-per-img', type=int, default=300)
    p.add_argument('--limit', type=int, default=-1,
                   help='Run on first N images only (debug).  -1 = all.')
    p.add_argument('--also-baseline', action='store_true',
                   help='Also dump fine-tuned-only predictions side-by-side '
                        '(for fair comparison without fusion).')
    # Pretrained-expert prompt aliasing -----------------------------------
    p.add_argument('--dataset', default=None,
                   help='Dataset short name (artaxor/dior/...). When set '
                        'and the dataset has a `pre_aliases` entry in '
                        'configs/cdfsod/_dataset_meta.py, the PRETRAINED '
                        'expert is queried with the alias prompt while '
                        'the fine-tuned expert keeps the original class '
                        'names. Class IDs are matched by position so no '
                        'remapping is needed downstream.')
    p.add_argument('--pre-aliases', nargs='+', default=None,
                   help='Manual override: comma- or space-separated alias '
                        'list, one per class in COCO category order. Takes '
                        'precedence over --dataset lookup.')
    p.add_argument('--no-pre-aliases', action='store_true',
                   help='Disable alias prompt: query both experts with the '
                        'original class names (legacy behaviour).')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve_ckpt(spec: str) -> str:
    """Resolve a glob pattern (for fine-tuned ckpt) to a single file."""
    if spec.startswith('http://') or spec.startswith('https://'):
        return spec
    matches = sorted(glob.glob(spec))
    if not matches:
        if os.path.exists(spec):
            return spec
        sys.exit(f'[poe] No checkpoint matched: {spec}')
    if len(matches) > 1:
        print(f'[poe] {len(matches)} ckpts matched, using newest: {matches[-1]}')
    return matches[-1]


def build_text_prompt(classes: List[str]) -> str:
    """GroundingDINO expects '. '-delimited prompt ending in '.'."""
    return ' . '.join(classes) + ' .'


def resolve_pre_aliases(args, classes: List[str]) -> Optional[List[str]]:
    """Decide which alias list (if any) the pretrained expert should use.

    Resolution order:
        1. ``--no-pre-aliases``           -> None (legacy: same prompt)
        2. ``--pre-aliases a b c ...``    -> use as-is (must match #classes)
        3. ``--dataset <ds>`` + meta hit  -> use ``get_pre_aliases(ds)``
        4. Otherwise                      -> None

    Returns ``None`` if the resolved aliases are equal to ``classes``
    (no useful divergence) so the caller can short-circuit and reuse one
    prompt for both experts.
    """
    if args.no_pre_aliases:
        return None
    aliases: Optional[Tuple[str, ...]] = None
    if args.pre_aliases is not None:
        aliases = tuple(args.pre_aliases)
    elif args.dataset is not None:
        aliases = get_pre_aliases(args.dataset)
    if aliases is None:
        return None
    if len(aliases) != len(classes):
        sys.exit(
            f'[poe] pre_aliases length mismatch: got {len(aliases)} '
            f'aliases for {len(classes)} classes (must be 1:1 by position).')
    if tuple(aliases) == tuple(classes):
        return None
    return list(aliases)


def extract_predictions(result) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pull (bboxes, scores, labels) out of a DetDataSample.

    All tensors moved to CPU for cheap bookkeeping.
    """
    inst = result.pred_instances
    return inst.bboxes.detach().cpu(), \
           inst.scores.detach().cpu(), \
           inst.labels.detach().cpu()


# ---------------------------------------------------------------------------
# Core fusion
# ---------------------------------------------------------------------------
def fuse_scores(boxes_ft: torch.Tensor,
                scores_ft: torch.Tensor,
                boxes_pre: torch.Tensor,
                scores_pre: torch.Tensor,
                alpha: float = 0.3,
                iou_thr: float = 0.3,
                mode: str = 'log_linear',
                eps: float = 1e-3,
                gate_thr: float = 0.05) -> torch.Tensor:
    """Compute PoE-fused scores for fine-tuned detections.

    For every box in `boxes_ft`, find the maximum-score box in `boxes_pre`
    that overlaps it with IoU > iou_thr (regardless of class label).
    Combine the two scores under one of four fusion modes.

    Returns: fused_scores tensor of shape ``[N_ft]``.
    """
    if len(boxes_ft) == 0:
        return scores_ft

    # Look up matched pre-score for every ft-box (label-agnostic) -----------
    if len(boxes_pre) == 0:
        # No pretrained detection in image -> pretrained never "agrees".
        # Use eps as floor (log_linear) or 0 (others).
        scores_pre_matched = torch.full_like(scores_ft, eps)
    else:
        ious = box_iou(boxes_ft, boxes_pre)              # [N_ft, N_pre]
        # Mask non-overlapping pairs to -inf, then max along pre.
        score_grid = scores_pre.unsqueeze(0).expand_as(ious).clone()
        score_grid[ious <= iou_thr] = 0.0
        scores_pre_matched, _ = score_grid.max(dim=1)    # [N_ft]
        scores_pre_matched = scores_pre_matched.clamp(min=0.0)

    # Apply chosen fusion ---------------------------------------------------
    if mode == 'log_linear':
        # True PoE.  log s = (1-a) log s_ft + a log s_pre.
        log_ft  = scores_ft.clamp(min=eps).log()
        log_pre = scores_pre_matched.clamp(min=eps).log()
        fused   = ((1.0 - alpha) * log_ft + alpha * log_pre).exp()

    elif mode == 'multiplicative':
        # Boost s_ft proportional to pretrained agreement.
        fused = scores_ft * (1.0 + alpha * scores_pre_matched)

    elif mode == 'additive':
        # Convex combination of probabilities (Mixture of Experts, NOT PoE).
        fused = (1.0 - alpha) * scores_ft + alpha * scores_pre_matched

    elif mode == 'gated':
        # Only boost when pretrained agrees clearly.
        gate  = (scores_pre_matched > gate_thr).float()
        fused = scores_ft + alpha * gate * scores_pre_matched

    else:
        raise ValueError(f'Unknown fusion mode: {mode!r}')

    return fused.clamp(min=0.0, max=1.0)


# ---------------------------------------------------------------------------
# COCO output
# ---------------------------------------------------------------------------
def to_coco_results(boxes: torch.Tensor,
                    scores: torch.Tensor,
                    labels: torch.Tensor,
                    image_id: int,
                    cat_ids: List[int],
                    score_thr: float = 0.001,
                    max_det: int = 300) -> List[dict]:
    """Convert tensors to COCO results JSON entries.

    Args:
        cat_ids: Mapping from label-index (model-output) -> COCO category_id.
                 Built from sorted COCO categories of the test annotation.
    """
    if len(boxes) == 0:
        return []

    keep = scores >= score_thr
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if len(boxes) > max_det:
        topk = torch.topk(scores, max_det)
        boxes = boxes[topk.indices]
        scores = scores[topk.indices]
        labels = labels[topk.indices]

    # Preserve original image_id type (some COCO datasets use string UUIDs,
    # e.g. ArTaxOr).  pycocotools accepts whatever type the GT uses.
    if isinstance(image_id, (int,)) or (
            isinstance(image_id, str) and image_id.lstrip('-').isdigit()):
        img_id_out = int(image_id)
    else:
        img_id_out = image_id  # keep as string

    out = []
    for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        x1, y1, x2, y2 = box
        out.append({
            'image_id': img_id_out,
            'category_id': int(cat_ids[int(label)]),
            'bbox': [round(x1, 2), round(y1, 2),
                     round(x2 - x1, 2), round(y2 - y1, 2)],
            'score': round(float(score), 5),
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    pre_config = args.pre_config or args.ft_config
    ft_ckpt = resolve_ckpt(args.ft_ckpt)
    pre_ckpt = resolve_ckpt(args.pre_ckpt)

    print('=' * 70)
    print(f'[poe] ft_config  = {args.ft_config}')
    print(f'[poe] ft_ckpt    = {ft_ckpt}')
    print(f'[poe] pre_config = {pre_config}')
    print(f'[poe] pre_ckpt   = {pre_ckpt}')
    print(f'[poe] alpha={args.alpha}, iou={args.iou_thr}, mode={args.mode}')
    print('=' * 70)

    # ------------------------------------------------------------------
    # Build models
    # ------------------------------------------------------------------
    print('[poe] loading fine-tuned model ...')
    ft_model = _build_detector(args.ft_config, ft_ckpt, device=args.device)
    print('[poe] loading pretrained model ...')
    pre_model = _build_detector(pre_config, pre_ckpt, device=args.device)
    ft_model.eval()
    pre_model.eval()

    # ------------------------------------------------------------------
    # Load COCO test annotations -- need (id, file_name) pairs and the
    # category mapping (label-index in {0..C-1}  ->  COCO category_id).
    # ------------------------------------------------------------------
    with open(args.test_ann) as f:
        coco_ann = json.load(f)
    images = coco_ann['images']
    cats = sorted(coco_ann['categories'], key=lambda c: c['id'])
    cat_ids = [c['id'] for c in cats]
    classes = [c['name'] for c in cats]

    ft_text_prompt = build_text_prompt(classes)
    pre_aliases = resolve_pre_aliases(args, classes)
    if pre_aliases is None:
        pre_text_prompt = ft_text_prompt
        print(f'[poe] {len(classes)} classes -> text_prompt="{ft_text_prompt}"  '
              f'(both experts share this prompt)')
    else:
        pre_text_prompt = build_text_prompt(pre_aliases)
        print(f'[poe] {len(classes)} classes (alias prompt active)')
        print(f'[poe]   ft  prompt = "{ft_text_prompt}"')
        print(f'[poe]   pre prompt = "{pre_text_prompt}"')
        print('[poe]   class IDs match by POSITION (ft labels are reused '
              'verbatim in the COCO output)')

    if args.limit > 0:
        images = images[:args.limit]
        print(f'[poe] LIMITING to first {args.limit} images')

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    fused_results = []
    baseline_results = []  # only filled if --also-baseline

    pbar = tqdm(images, desc='PoE infer', unit='img')
    n_skipped = 0
    for img_info in pbar:
        img_path = os.path.join(args.img_prefix, img_info['file_name'])
        if not os.path.exists(img_path):
            n_skipped += 1
            continue

        try:
            r_ft = inference_detector(
                ft_model, img_path,
                text_prompt=ft_text_prompt, custom_entities=True)
            r_pre = inference_detector(
                pre_model, img_path,
                text_prompt=pre_text_prompt, custom_entities=True)
        except Exception as e:
            tqdm.write(f'[err] {img_path}: {e}')
            n_skipped += 1
            continue

        boxes_ft, scores_ft, labels_ft = extract_predictions(r_ft)
        boxes_pre, scores_pre, _       = extract_predictions(r_pre)

        # Fuse ----------------------------------------------------------
        fused_scores = fuse_scores(
            boxes_ft, scores_ft, boxes_pre, scores_pre,
            alpha=args.alpha, iou_thr=args.iou_thr, mode=args.mode,
            eps=args.eps, gate_thr=args.gate_thr,
        )

        fused_results.extend(to_coco_results(
            boxes_ft, fused_scores, labels_ft,
            image_id=img_info['id'], cat_ids=cat_ids,
            score_thr=args.score_thr, max_det=args.max_per_img,
        ))

        if args.also_baseline:
            baseline_results.extend(to_coco_results(
                boxes_ft, scores_ft, labels_ft,
                image_id=img_info['id'], cat_ids=cat_ids,
                score_thr=args.score_thr, max_det=args.max_per_img,
            ))

        pbar.set_postfix(ft=len(boxes_ft), pre=len(boxes_pre))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(fused_results, f)
    print(f'[poe] saved {len(fused_results)} fused predictions -> {out_path}')

    if args.also_baseline:
        bl_path = out_path.with_name(out_path.stem + '_ft_only.json')
        with open(bl_path, 'w') as f:
            json.dump(baseline_results, f)
        print(f'[poe] saved {len(baseline_results)} baseline predictions -> {bl_path}')

    if n_skipped:
        print(f'[poe] WARNING: {n_skipped} images skipped (missing or err)')


if __name__ == '__main__':
    main()
