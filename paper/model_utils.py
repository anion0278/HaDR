import ws_specific_settings as wss

import common_settings as s
s.add_packages_paths()
from mmcv import Config
from mmdet.apis import set_random_seed

def get_pipelines(in_channels):
    from mmcv import Config
    options = {  
        1 : "pipelines_d",
        3 : "pipelines_rgb",
        4 : "pipelines_rgbd" }
    return Config.fromfile(s.path_to_configs % options[in_channels],'temp_pipe_config').data

def get_config(arch_name, channels):
    cfg = Config.fromfile(s.path_to_configs % arch_name, 'temp')
    cfg.model.backbone.in_channels = channels
    cfg.data = get_pipelines(cfg.model.backbone.in_channels)
    set_config_params(cfg)
    return cfg

def set_config_params(cfg):
    cfg.data.imgs_per_gpu = 4
    cfg.data.workers_per_gpu = wss.workers
    cfg.data.train.type = "CocoDataset"
    cfg.data.val.type = "CocoDataset"
    cfg.seed = 0
    set_random_seed(0, deterministic=True)
    cfg.workflow = [("train", 1), ("val", 1)] 

    cfg.device_ids = range(1)
    cfg.gpu_ids = range(1)
    cfg.gpus = 1
    cfg.dist_params = dict(backend='nccl')
    cfg.resume_from = None
    cfg.checkpoint_config = dict(create_symlink=False, interval = 4) # TRY TO USE state of optimizer save_optimizer = True

    cfg.log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
    cfg.log_level = 'INFO'
    cfg.log_config.interval = 1

    if "pretrained" in cfg.model: cfg.model.pop("pretrained") # get rid of pretrained backbone since we will init weights from checkpoint
    cfg.lr_config = dict(policy="poly", power=0.9, min_lr=1e-7, by_epoch=False) # if by_epoch = False, then changes according to iteration
    cfg.optimizer = dict(type='SGD', lr=0.0002, momentum=0.9, weight_decay=0.001)
    #dict(type='Adam', lr=0.0003, weight_decay=0.0001)
    cfg.optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))