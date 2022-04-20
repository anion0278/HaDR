resolution = (256, 320) # SEQUENCE !!

# model settings
model = dict(
    type='SOLOv2',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        in_channels = 4,
        num_stages=4, # there are actually 5 stages, conv1 is freezed whenever num_stages > 1s
        frozen_stages=4,
        out_indices=(0, 1, 2, 3), # C2, C3, C4, C5
        style='pytorch'), 
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    bbox_head=dict(
        type='SOLOv2Head',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        seg_feat_channels=512,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        ins_out_channels=256,
        loss_ins=dict(
            type='DiceLoss',
            use_sigmoid=True,
            loss_weight=3.0),
        loss_cate=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)),
    mask_feat_head=dict(
            type='MaskFeatHead',
            in_channels=256,
            out_channels=128,
            start_level=0,
            end_level=3,
            num_classes=256,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    )
# training and testing settings
train_cfg = dict()
test_cfg = dict(
    nms_pre=500,
    score_thr=0.1,
    mask_thr=0.5,
    update_thr=0.05,
    kernel='gaussian',  # gaussian/linear
    sigma=2.0,
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53, 128.0], std=[58.395, 57.12, 57.375, 57.0], to_rgb=False) # !!!
train_pipeline = [
    dict(type='LoadRgbdImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize',
         img_scale=[(224, 288), (288, 352)],
         multiscale_mode='range',
         keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    dict(type='DecimateDepth', probability=0.5),
    dict(type='CorruptRgbd', corruption="motion_blur", max_severity=4),
    dict(type='CorruptRgbd', corruption="elastic_transform", max_severity=2),
    dict(type='CorruptRgbd', corruption="brightness", max_severity=4),
    dict(type='CorruptRgbd', corruption="contrast", max_severity=3),
    dict(type='CorruptRgbd', corruption="saturate", max_severity=3),
    dict(type='CorruptRgbd', corruption="fog", max_severity=4),
    dict(type='CorruptRgbd', corruption="defocus_blur", max_severity=2),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ConvertRgbdToBgrd'), # TODO CHECK whether it works correctly !!!!
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
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ConvertRgbdToBgrd'), # TODO CHECK whether it works correctly !!!!
            dict(type='Pad', size_divisor=32), 
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
val_pipeline = [ 
    dict(type='LoadRgbdImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ConvertRgbdToBgrd'),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'], 
        meta_keys=('filename','ori_shape', 'img_shape', 'pad_shape', 'img_norm_cfg')),
]
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=val_pipeline),
    test=dict(pipeline=test_pipeline))
    
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.001)
# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-8, by_epoch=False) # if by_epoch = False, then changes according to iteration

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.01,
    step=[27, 33])

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 36
device_ids = range(1)
gpu_ids = range(1)
gpus = 1
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/solov2_release_r101_fpn_8gpu_3x'
load_from = None
resume_from = None
workflow = [('train', 1)]
