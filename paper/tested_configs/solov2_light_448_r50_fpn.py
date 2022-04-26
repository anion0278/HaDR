num_classes = 2

# model settings
model = dict(
    type='SOLOv2',
    #pretrained='torchvision://resnet50', # The backbone weights will be overwritten when using load_from or resume_from. 
    # https://github.com/open-mmlab/mmdetection/issues/7817#issuecomment-1108503826
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels = 3,
        num_stages=4,
        out_indices=(0, 1, 2, 3), # C2, C3, C4, C5
        frozen_stages=-1, # -1 is unfrozen, 0 -> C1 is frozen, 1 - C1, C2 are frozen and so on
        style='pytorch'),
        # norm_eval = True # true by default, "you're fine-tuning to minimize training, it's typically best to keep batch normalization frozen"
        # https://stackoverflow.com/questions/63016740/why-its-necessary-to-frozen-all-inner-state-of-a-batch-normalization-layer-when
        # required for traning from scratch
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    bbox_head=dict(
        type='SOLOv2Head',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=2,
        seg_feat_channels=256,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 56), (28, 112), (56, 224), (112, 448), (224, 896)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        ins_out_channels=128,
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
            num_classes=128,
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

