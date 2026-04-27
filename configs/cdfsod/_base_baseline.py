# =====================================================================
# CD-FSOD Benchmark - Baseline Base Config (for fair comparison)
# ---------------------------------------------------------------------
# Inherits the GroundingDINO Swin-B model from mmdetection and disables
# the heavy mix augmentation (CachedMosaic / CachedMixUp / YOLOXHSV) so
# that ETS performance is reported under the SAME augmentation regime
# typically used in CDFSOD-benchmark (CD-ViTO, DE-ViT-FT, etc.).
#
# Leaf configs (e.g. configs/cdfsod/artaxor/5shot_baseline.py) only need
# to override:
#     data_root, metainfo, train_ann_file, model.bbox_head.num_classes
# =====================================================================
_base_ = [
    '../../mmdetection/configs/grounding_dino/CDFSOD/'
    'GroudingDINO-few-shot-SwinB.py',
]

# ---------------------------------------------------------------------
# Default placeholders -- ALL OVERRIDDEN by leaf configs
# ---------------------------------------------------------------------
data_root = '../datasets/PLACEHOLDER/'
metainfo = dict(classes=('placeholder',))
train_ann_file = 'annotations/5_shot.json'
backend_args = None

# ---------------------------------------------------------------------
# Pipelines (no mix-augmentation -- baseline regime)
# ---------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                    keep_ratio=True),
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                    keep_ratio=True),
            ],
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities')),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities')),
]

# ---------------------------------------------------------------------
# Dataloaders (overridden parameters cascade from leaf configs)
# ---------------------------------------------------------------------
dataset_type = 'CocoDataset'

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        metainfo=metainfo,
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False),
        return_classes=True))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        metainfo=metainfo,
        pipeline=test_pipeline,
        return_classes=True))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric='bbox',
    classwise=True,
    format_only=False,
    backend_args=backend_args)

test_evaluator = val_evaluator

# ---------------------------------------------------------------------
# Optimizer / Schedule (single fine-tuning run, no grid-search)
# ---------------------------------------------------------------------
max_epochs = 100
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Schedule: cosine decay only, without LinearLR warmup.
#
# See ``_base_ets.py`` for the full diagnostic.  In short, the previous
# warmup+cosine form still behaved badly on 1-shot runs with only
# 1 iter/epoch: after conversion, cosine captured the warmup-reduced LR
# as its base, so the LR climbed toward 1e-5 instead of starting near
# the optimizer base LR and decaying.
#
# `eta_min_ratio=0.1` decays every parameter group to 10% of its own base
# LR, preserving the intended backbone lr_mult=0.1 and main LR
# relationship throughout training.
param_scheduler = [
    dict(type='CosineAnnealingLR', by_epoch=True,
         begin=0, end=max_epochs, T_max=max_epochs, eta_min_ratio=0.1,
         convert_to_iter_based=True),
]

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.),
        'backbone': dict(lr_mult=0.1),
    }))

default_hooks = dict(
    checkpoint=dict(by_epoch=True, interval=10, max_keep_ckpts=2,
                    save_best='coco/bbox_mAP', rule='greater'))

# EMA stabilisation -- crucial for 1-shot training where val mAP
# oscillates wildly across epochs. Maintains an exponentially-averaged
# shadow copy of the weights for evaluation.
custom_hooks = [
    # NOTE: ``EMAHookSafe`` is a thin mmdet-side subclass of mmengine's
    # ``EMAHook`` that handles flat/un-wrapped checkpoints (e.g. the
    # ``groundingdino_swinb_cogcoor_mmdet-*.pth`` we use as load_from).
    # The vanilla ``EMAHook`` crashes on those with KeyError: 'state_dict'.
    dict(
        type='EMAHookSafe',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority='ABOVE_NORMAL'),
]

auto_scale_lr = dict(base_batch_size=8, enable=False)
