resolution = (256, 320) # CORRECT sequnce, checked in loading.py 
channels = 4

import common_settings

train_pipeline = [
    dict(type='LoadRgbdImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize',
         img_scale=[(240, 320), (480, 640)], # CORRECT sequnce, checked in loading.py H, W
         multiscale_mode='range',
         keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    dict(type='DecimateDepth', probability=0.5),
    dict(type='CorruptRgbd', corruption="motion_blur", max_severity=3, channels="color"),
    dict(type='CorruptRgbd', corruption="brightness", max_severity=5, channels="random"),
    dict(type='CorruptRgbd', corruption="saturate", max_severity=5, channels="color"),
    dict(type='ConvertRgbdToBgrd'), 
    dict(type='Normalize', **common_settings.get_norm_params(channels, "train")),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadRgbdImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[resolution],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **common_settings.get_norm_params(channels, "test")),
            dict(type='Pad', size_divisor=32), # Pad is required ! since our Real-cam images have shape 240x320 
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
val_pipeline = [ 
    dict(type='LoadRgbdImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize',
         img_scale=[resolution],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='Normalize', **common_settings.get_norm_params(channels, "train")),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'], 
        meta_keys=('filename','ori_shape', 'img_shape', 'pad_shape', 'img_norm_cfg', 'scale_factor')),
]
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=val_pipeline),
    test=dict(pipeline=test_pipeline))
    