# =====================================================================
# CD-FSOD Benchmark - VPE-DINO Base Config
# ---------------------------------------------------------------------
# Inherits the baseline no-mix augmentation pipeline from
# ``_base_baseline.py`` and stacks the VPE visual-prompt module on top.
# This is the validated "VPE + baseline pipeline" recipe:
# no CachedMosaic / CachedMixUp / YOLOXHSV, no EMA, and 80 epochs.
#
# Differences vs. the inherited baseline config:
#
#   1. model.type = 'VPEGroundingDINO'  (registered in
#      `mmdetection/mmdet/models/detectors/vpe_grounding_dino.py`)
#   2. model.support_features_path  -> path to a per-(dataset,shot)
#      ``.pt`` bundle produced by ``tools/build_support_features.py``.
#      Leaf configs must override this with their actual path.
#   3. The new ``vpe_proj`` Linear is the only added trainable module
#      (~262K params at D_swin=1024 -> D_text=256).
#
# Leaf configs (e.g. ``configs/cdfsod/artaxor/5shot_vpe.py``) override:
#     data_root, metainfo, train_ann_file, model.bbox_head.num_classes,
#     model.support_features_path
# =====================================================================
_base_ = ['./_base_baseline.py']

# ---------------------------------------------------------------------
# Model: swap detector type, leave everything else inherited
# ---------------------------------------------------------------------
model = dict(
    type='VPEGroundingDINO',
    # Filled in by the leaf config (one path per (dataset, shot)).
    support_features_path=None,
    # vpe_dim defaults to text_feat_map.out_features (256 for GD-SwinB).
    vpe_dim=None,
    vpe_pool='mean',     # K-shot pooling: 'mean' | 'max'
    vpe_disable=False,   # set True for ablation: disables injection
)

# ---------------------------------------------------------------------
# Optimizer override: give the new vpe_proj a slightly higher LR than
# the slowed-down backbone (which uses lr_mult=0.1).  vpe_proj is
# initialised to zero, so it needs full lr to learn effectively.
# ---------------------------------------------------------------------
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.),
        'backbone': dict(lr_mult=0.1),
        'vpe_proj': dict(lr_mult=1.0, decay_mult=0.0),
    }))

max_epochs = 80
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=10)

param_scheduler = [
    dict(
        type='MultiStepLR',
        by_epoch=True,
        begin=0,
        end=max_epochs,
        milestones=[33],
        gamma=0.1),
]

custom_hooks = []
