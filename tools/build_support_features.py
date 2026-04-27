#!/usr/bin/env python
"""Pre-compute K-shot support visual features for VPE-DINO.

For every (dataset, shot) combo we:
  1. read ``datasets/<NAME>/annotations/<shot>_shot.json`` (COCO format)
  2. crop each annotation's bbox out of its train image
  3. push each crop through GroundingDINO's Swin-B backbone (ImageNet-norm,
     224x224) and global-avg-pool the deepest stage to a 1024-d vector
  4. group vectors by COCO category and save the result as a single
     ``support_features/<dataset>_<shot>shot.pt`` checkpoint.

The saved tensor has shape ``[num_classes, K_max, 1024]`` (padded with the
class-mean for short rows) plus a ``K_per_class`` list so the consumer can
mask padding.

Output is consumed by:
  * ``mmdetection/mmdet/models/detectors/vpe_grounding_dino.py``
  * ``configs/cdfsod/_base_vpe.py``

Run from the ETS repo root (NOT from mmdetection/) so the relative paths
in `_dataset_meta` resolve correctly:

    python tools/build_support_features.py
    python tools/build_support_features.py --datasets neu-det --shots 1 5
    python tools/build_support_features.py --pretrained /path/to/local.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'configs' / 'cdfsod'))

from _dataset_meta import DATASETS, DATASET_ORDER, SHOTS  # noqa: E402

DEFAULT_PRETRAINED_URL = (
    'https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/'
    'groundingdino_swinb_cogcoor_mmdet-55949c9c.pth'
)

# ImageNet normalisation used by GD's Swin-B (matches
# `mmdetection/configs/_base_/.../GroudingDINO-few-shot.py: data_preprocessor`).
IMAGENET_MEAN = (123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0)
IMAGENET_STD = (58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datasets', nargs='+', default=DATASET_ORDER,
                   help='Subset of datasets to extract for. Default: all 6.')
    p.add_argument('--shots', nargs='+', type=int, default=list(SHOTS))
    p.add_argument('--pretrained', default=DEFAULT_PRETRAINED_URL,
                   help='URL or path to GD-Swin-B checkpoint. The Swin '
                        'sub-state-dict is extracted; full model not loaded.')
    p.add_argument('--crop-size', type=int, default=224,
                   help='Square size each bbox is resized to before the '
                        'backbone. Larger = more detail but slower; '
                        '224 matches ImageNet pretraining.')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out-dir', default='support_features',
                   help='Output directory (relative to ETS repo root). '
                        'Each (ds, shot) combo writes one .pt file.')
    p.add_argument('--overwrite', action='store_true',
                   help='Re-extract even if .pt already exists.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------
def build_swin_backbone(pretrained: str, device: str) -> nn.Module:
    """Construct GroundingDINO's Swin-B backbone and load the matching
    sub-state-dict from the GD checkpoint.

    The Swin config is copied verbatim from
    ``configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB.py``::

        embed_dims=128, depths=[2,2,18,2], num_heads=[4,8,16,32],
        window_size=12, drop_path_rate=0.3, patch_norm=True,
        pretrain_img_size=384, out_indices=(0,1,2,3)

    We must build it via mmdet's registry because it inherits a custom
    ``mmdet.models.backbones.SwinTransformer`` (which differs slightly
    from torchvision's). All other parts of the GD model are skipped.
    """
    # Lazy import: mmdet is only available on the server.
    from mmdet.registry import MODELS
    from mmengine.runner import CheckpointLoader

    cfg = dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
    )
    backbone = MODELS.build(cfg)

    # Load only the backbone tensors out of the full GD checkpoint -----------
    print(f'[vpe] loading pretrained weights: {pretrained}')
    ckpt = CheckpointLoader.load_checkpoint(pretrained, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    sub = {k[len('backbone.'):]: v
           for k, v in state.items() if k.startswith('backbone.')}
    if not sub:
        raise RuntimeError(
            f'no backbone.* keys found in {pretrained!r}; got {list(state)[:5]}')
    missing, unexpected = backbone.load_state_dict(sub, strict=False)
    if missing:
        print(f'[vpe][warn] {len(missing)} missing keys (e.g. {missing[:3]})')
    if unexpected:
        print(f'[vpe][warn] {len(unexpected)} unexpected keys '
              f'(e.g. {unexpected[:3]})')

    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


# ---------------------------------------------------------------------------
# Crop + forward
# ---------------------------------------------------------------------------
class CropTransform:
    """Pillow crop + resize + ImageNet normalise -> torch.Tensor [3,H,W]."""

    def __init__(self, size: int):
        self.size = size
        self.norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image: Image.Image,
                 bbox_xywh: Tuple[float, float, float, float]) -> torch.Tensor:
        x, y, w, h = bbox_xywh
        # Defensive: clip to image bounds, ensure positive area.
        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        x1 = min(image.width, int(round(x + max(1.0, w))))
        y1 = min(image.height, int(round(y + max(1.0, h))))
        if x1 <= x0 or y1 <= y0:
            return None  # caller skips
        crop = image.crop((x0, y0, x1, y1)).convert('RGB')
        crop = crop.resize((self.size, self.size), Image.BILINEAR)
        return self.norm(self.to_tensor(crop))


@torch.no_grad()
def extract_pooled(backbone: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Run a [B,3,H,W] crop tensor through Swin and return [B,1024] pooled
    features from the deepest stage.
    """
    feats = backbone(batch)  # tuple of 4 stages
    last = feats[-1]          # [B, 1024, H/32, W/32]
    return last.mean(dim=(2, 3))


# ---------------------------------------------------------------------------
# Main per-(dataset, shot) extractor
# ---------------------------------------------------------------------------
def extract_for(dataset: str, shot: int, args, backbone: nn.Module,
                cropper: CropTransform, out_path: Path) -> Optional[dict]:
    meta = DATASETS[dataset]
    classes = list(meta['classes'])
    # data_root is relative to mmdetection/.  Resolve to repo root.
    rel = meta['data_root'].lstrip('./').replace('../', '')
    root = REPO_ROOT / rel
    ann_path = root / 'annotations' / f'{shot}_shot.json'
    if not ann_path.exists():
        print(f'[skip] {dataset} {shot}-shot: annotation not found '
              f'({ann_path.relative_to(REPO_ROOT)})')
        return None

    with open(ann_path) as f:
        coco = json.load(f)

    # Build helpful indices ------------------------------------------------
    cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}
    name_to_idx = {n: i for i, n in enumerate(classes)}
    img_by_id = {im['id']: im for im in coco['images']}

    # Try a few train-image directory conventions (CDFSOD-benchmark uses
    # `train` for K-shot images).
    img_dirs = [root / d for d in ('train', 'images', 'images/train')]
    img_dirs = [d for d in img_dirs if d.exists()]
    if not img_dirs:
        print(f'[skip] {dataset} {shot}-shot: no images dir under {root}')
        return None

    def find_image(file_name: str) -> Optional[Path]:
        for d in img_dirs:
            cand = d / file_name
            if cand.exists():
                return cand
        return None

    # Group annotations by class index ------------------------------------
    per_class_feats: List[List[torch.Tensor]] = [[] for _ in classes]
    skipped = 0
    bad_class = 0

    pbar = tqdm(coco['annotations'],
                desc=f'{dataset} {shot}-shot crops', unit='ann')
    for ann in pbar:
        cat_name = cat_id_to_name.get(ann['category_id'])
        if cat_name is None or cat_name not in name_to_idx:
            bad_class += 1
            continue
        cls_idx = name_to_idx[cat_name]
        img_info = img_by_id.get(ann['image_id'])
        if img_info is None:
            skipped += 1
            continue
        img_path = find_image(img_info['file_name'])
        if img_path is None:
            skipped += 1
            continue
        try:
            with Image.open(img_path) as im:
                tensor = cropper(im.convert('RGB'),
                                 tuple(ann['bbox']))
        except Exception as e:
            tqdm.write(f'[err] {img_path}: {e}')
            skipped += 1
            continue
        if tensor is None:
            skipped += 1
            continue
        # Forward immediately to save host RAM.
        feat = extract_pooled(backbone,
                              tensor.unsqueeze(0).to(args.device)).cpu()[0]
        per_class_feats[cls_idx].append(feat)

    # Stack to [C, K_max, D] with class-mean padding ----------------------
    K_per_class = [len(v) for v in per_class_feats]
    if min(K_per_class) == 0:
        empty = [classes[i] for i, k in enumerate(K_per_class) if k == 0]
        print(f'[warn] {dataset} {shot}-shot: empty support for classes '
              f'{empty}; padding with global mean as fallback')

    K_max = max(K_per_class)
    D = 1024
    out = torch.zeros(len(classes), K_max, D)
    # Compute global mean to use as fallback for empty classes.
    nonempty = [v for v in per_class_feats if v]
    global_mean = (torch.stack([f for fs in nonempty for f in fs]).mean(0)
                   if nonempty else torch.zeros(D))
    for i, fs in enumerate(per_class_feats):
        if not fs:
            out[i, 0] = global_mean
            K_per_class[i] = 1
            continue
        s = torch.stack(fs)             # [k, D]
        out[i, :s.shape[0]] = s
        # Pad shorter rows with that class's mean for downstream simplicity.
        if s.shape[0] < K_max:
            out[i, s.shape[0]:] = s.mean(0, keepdim=True)

    bundle = dict(
        features=out,                   # [C, K_max, D]
        K_per_class=K_per_class,        # actual K per class
        classes=classes,                # in dataset_meta order
        dataset=dataset,
        shot=shot,
        crop_size=args.crop_size,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    print(f'[ok] {dataset} {shot}-shot -> {out_path.relative_to(REPO_ROOT)}  '
          f'shape={tuple(out.shape)}  K_per_class={K_per_class}  '
          f'(skipped={skipped}, bad_class={bad_class})')
    return bundle


def main():
    args = parse_args()
    out_root = REPO_ROOT / args.out_dir

    # Load backbone once and reuse across all datasets/shots.
    backbone = build_swin_backbone(args.pretrained, args.device)
    cropper = CropTransform(args.crop_size)

    for ds in args.datasets:
        if ds not in DATASETS:
            print(f'[skip] unknown dataset: {ds}')
            continue
        for shot in args.shots:
            out_path = out_root / f'{ds}_{shot}shot.pt'
            if out_path.exists() and not args.overwrite:
                print(f'[cached] {out_path.relative_to(REPO_ROOT)}')
                continue
            extract_for(ds, shot, args, backbone, cropper, out_path)


if __name__ == '__main__':
    main()
