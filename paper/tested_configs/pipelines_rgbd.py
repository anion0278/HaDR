resolution = (256, 320) # CORRECT sequnce, checked in loading.py 
channels = 4

import common_settings

train_pipeline = [
    dict(type='LoadBgrdImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize',
         img_scale=[resolution],
         multiscale_mode='value',
         keep_ratio=True),
    # dict(type='Resize',
    #      img_scale=[resolution], # if single value is provided and keep_ration = True, then the sequence should be H,W
    #      # if single value is provided and keep_ration = False, then the sequence should be W,H
    #      multiscale_mode='value', 
    #      keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    dict(type='RandomCropResizeShiftRgbd', crop_size_ratio = 0.03),
    # dict(type='DecimateDepth', probability=0.5),
    # dict(type='DropImgComponent', probability=0.15),
    # dict(type='CorruptRgbd', corruption="motion_blur", max_severity=3, channels="color"),
    # dict(type='CorruptRgbd', corruption="gaussian_blur", max_severity=2, channels="depth"),
    # dict(type='CorruptRgbd', corruption="brightness", max_severity=5, channels="random"),
    # dict(type='CorruptRgbd', corruption="saturate", max_severity=5, channels="color"),
    dict(type='ConvertBgrdToRgbd'), 
    dict(type='Normalize', **common_settings.get_norm_params(channels, "train")),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadBgrdImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[resolution],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ConvertBgrdToRgbd'),
            dict(type='Normalize', **common_settings.get_norm_params(channels, "test")),
            dict(type='Pad', size_divisor=32), # Pad is required ! since our Real-cam images have shape 240x320 
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
val_pipeline = [ 
    dict(type='LoadBgrdImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize',
         img_scale=[resolution], # if single value is provided, then the sequence should be W,H
         multiscale_mode='value', 
         keep_ratio=True),
    dict(type='ConvertBgrdToRgbd'), 
    dict(type='Normalize', **common_settings.get_norm_params(channels, "train")),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'], 
        meta_keys=('filename','ori_shape', 'img_shape', 'pad_shape', 'img_norm_cfg', 'scale_factor')),
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=val_pipeline),
    test=dict(pipeline=test_pipeline))
    