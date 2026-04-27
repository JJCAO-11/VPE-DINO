_base_ = ['./_base_vpe.py']

max_epochs = 50
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=10)

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

# ---------------------------------------------------------------------
# Switch to baseline pipeline (no Mosaic/MixUp/HSV) -- the old
# high-scoring run used this simpler augmentation, which works better
# for 1-shot with only 7 training images.
# ---------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
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

train_dataloader = dict(
    num_workers=4,
    dataset=dict(pipeline=train_pipeline))
