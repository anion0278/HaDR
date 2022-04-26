import os, sys
path = os.path.abspath("./modified_packges")
sys.path.insert(0,path)
import mmcv
# print(os.path.abspath(mmcv.__file__))

from mmcv import Config
from mmcv.runner import save_checkpoint

from datetime import datetime as dt
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import ws_specific_settings as wss
import common_settings as s

import warnings
warnings.filterwarnings("ignore")  # disables annoying deprecation warnings


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
    cfg.checkpoint_config = dict(create_symlink=False, interval = 1) # TRY TO USE state of optimizer save_optimizer = True

    cfg.log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
    cfg.log_level = 'INFO'
    cfg.log_config.interval = 1

    if "pretrained" in cfg.model: cfg.model.pop("pretrained") # get rid of pretrained backbone since we will init weights from checkpoint
    cfg.lr_config = dict(policy="poly", power=0.9, min_lr=1e-7, by_epoch=False) # if by_epoch = False, then changes according to iteration
    cfg.optimizer = dict(type='SGD', lr=0.0002, momentum=0.9, weight_decay=0.001)
    cfg.optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))


def get_datasets(cfg):
    datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.val)]
    datasets[0].CLASSES = ["hand"]
    datasets[1].CLASSES = datasets[0].CLASSES
    return datasets

if __name__ == "__main__":
 
    TEST = True  # if True runs only 100 same images from validation dataset for BOTH TRAIN and VAL

    storage = wss.storage

    experiment_tag = "2G"
    # arch_name = "fast_mask_rcnn_r101_fpn"
    arch_name = "solov2_light_448_r50_fpn"
    # arch_name = "mask_rcnn_r101_fpn"
    # arch_name = "solov2_r101_fpn"
    channels = 4

    training_dataset = "sim_train_320x256" 
    validation_dataset = "sim_val_320x256"
    dataset_size = "full" 
    if TEST:
        dataset_size = "100" 
        wss.workers = 2
    train_dataset_path = storage + ":/datasets/" + training_dataset
    val_dataset_path =  storage + ":/datasets/" + validation_dataset
    main_channel = "depth" if channels == 1 else "color" 
    timestamp = dt.now().strftime("%a_D%d_M%m_%Hh_%Mm") 

    cfg = get_config(arch_name, channels)

    config_id = experiment_tag + "-" + arch_name + "_%sch" % channels + "-" + training_dataset + "_"+ dataset_size + "-" + timestamp
    print("CURRENT CONFIGURATION ID: " + config_id)
    cfg.work_dir = storage + ":/models/" + config_id
    os.makedirs(cfg.work_dir, exist_ok=True)

    cfg.data.train.ann_file = train_dataset_path + "/instances_hands_%s.json" % dataset_size 
    cfg.data.train.img_prefix = train_dataset_path + "/" + main_channel + "/"
    
    cfg.data.val.ann_file = val_dataset_path + "/instances_hands_%s.json" % dataset_size 
    cfg.data.val.img_prefix = val_dataset_path + "/" + main_channel + "/"

    datasets = get_datasets(cfg)

# FULLY FROZEN BACKBONE: https://img1.21food.com/img/cj/2014/10/9/1412794284347212.jpg

    cfg.load_from = storage + ":/models/" + arch_name + ".pth"
    cfg.optimizer.lr = 1e-4
    cfg.model.backbone.frozen_stages = 4
    cfg.total_epochs = 10
    model = build_detector(cfg.model, train_cfg = cfg.train_cfg, test_cfg = cfg.test_cfg)
    model.CLASSES = datasets[0].CLASSES # needed for the first time
    train_detector(model, datasets, cfg, distributed=False, validate=False, timestamp = timestamp)
    latest_checkpoint = cfg.work_dir + "/intermediate_" + arch_name +  ".pth"
    save_checkpoint(model, latest_checkpoint)
    print("Intermediate training finished")

# NON-FROZEN
# TODO DRY function

    cfg.load_from = latest_checkpoint
    cfg.optimizer.lr = 1e-5
    cfg.model.backbone.frozen_stages = -1
    cfg.total_epochs = 20
    model = build_detector(cfg.model, train_cfg = cfg.train_cfg, test_cfg = cfg.test_cfg)
    train_detector(model, datasets, cfg, distributed=False, validate=False, timestamp = timestamp)
    save_checkpoint(model, cfg.work_dir + "/final_" + arch_name + ".pth")
    print("Final (full network) training finished!")
