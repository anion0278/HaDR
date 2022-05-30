import common_settings as s
s.add_packages_paths()
import mmcv
# print(os.path.abspath(mmcv.__file__))

from mmcv.runner import save_checkpoint

from datetime import datetime as dt
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import ws_specific_settings as wss
import model_utils as utils
import os, argparse

import warnings
warnings.filterwarnings("ignore")  # disables annoying deprecation warnings

default_experiment_tag = "1C"
# default_arch_name = "fast_mask_rcnn_r101_fpn"
default_arch_name = "solov2_light_448_r50_fpn"
# default_arch_name = "mask_rcnn_r101_fpn"
# default_arch_name = "solov2_r101_fpn"
is_aug_enabled = True
default_channels = 4
TEST = False  # if True runs only 100 same images from validation dataset for BOTH TRAIN and VAL

def get_datasets(cfg):
    datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.val)]
    datasets[0].CLASSES = ["hand"]
    datasets[1].CLASSES = datasets[0].CLASSES
    return datasets

def manage_aug(cfg, is_aug_enabled):
    if not is_aug_enabled: 
        import copy
        # train_norm = cfg.data.train["pipeline"][3]
        cfg.data.train = copy.deepcopy(cfg.data.val)
        # cfg.data.train["pipeline"][3] = train_norm
    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument(
        '--tag',
        type=str,
        required = False,
        default=default_experiment_tag,
        help='tag for the experiment')
    parser.add_argument(
        '--arch',
        type=str,
        required = False,
        default=default_arch_name,
        help='architecture config name')
    parser.add_argument(
        '--channels', 
        type=int, 
        required = False,
        default=default_channels, 
        help='number of channels')
    parser.add_argument(
        '--aug',
        type=s.str2bool,
        required = False,
        default=is_aug_enabled,
        help='enable/disable augmenatations')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    storage = wss.storage
    
    frozen_epochs = 10
    unfrozen_epochs = 20
    training_dataset = "sim_train_320x256" 
    validation_dataset = "sim_val_320x256"

    dataset_size = "full" 
    if TEST:
        dataset_size = "100" 
        wss.workers = 2
    train_dataset_path = s.path_to_datasets + training_dataset
    val_dataset_path =  s.path_to_datasets + validation_dataset
    main_channel =  utils.get_main_channel_name(args.channels)
    timestamp = dt.now().strftime("%a_D%d_M%m_%Hh_%Mm") 

    cfg = utils.get_config(args.arch, args.channels)

    config_id = f"{args.tag}-{args.arch}_{args.channels}ch-{training_dataset}_{dataset_size}-Aug{args.aug}-BS{cfg.data.imgs_per_gpu}-{frozen_epochs}+{unfrozen_epochs}ep-{timestamp}"
    print("CURRENT CONFIGURATION ID: " + config_id)
    cfg.work_dir = storage + ":/models/" + config_id
    os.makedirs(cfg.work_dir, exist_ok=True)

    cgf = manage_aug(cfg, args.aug)
    cfg.data.train.ann_file =  f"{train_dataset_path}/instances_hands_{dataset_size}.json"
    cfg.data.train.img_prefix =  f"{train_dataset_path}/{main_channel}/"
    cfg.data.val.ann_file = f"{val_dataset_path}/instances_hands_{dataset_size}.json"
    cfg.data.val.img_prefix = f"{val_dataset_path}/{main_channel}/"
    datasets = get_datasets(cfg)

# FULLY FROZEN BACKBONE: https://img1.21food.com/img/cj/2014/10/9/1412794284347212.jpg

    cfg.load_from = f"{s.path_to_models}{args.arch}.pth"
    cfg.optimizer.lr = 1e-4
    cfg.model.backbone.frozen_stages = 4
    cfg.total_epochs = frozen_epochs
    model = build_detector(cfg.model, train_cfg = cfg.train_cfg, test_cfg = cfg.test_cfg)
    model.CLASSES = datasets[0].CLASSES # needed for the first time
    train_detector(model, datasets, cfg, distributed=False, validate=False, timestamp = timestamp)
    latest_checkpoint = cfg.work_dir + "/intermediate.pth" 
    save_checkpoint(model, latest_checkpoint)
    print("Intermediate training finished")

# NON-FROZEN
# TODO DRY function

    cfg.load_from = latest_checkpoint
    cfg.optimizer.lr = 1e-5
    cfg.model.backbone.frozen_stages = -1
    cfg.total_epochs = unfrozen_epochs
    model = build_detector(cfg.model, train_cfg = cfg.train_cfg, test_cfg = cfg.test_cfg)
    train_detector(model, datasets, cfg, distributed=False, validate=False, timestamp = timestamp)
    save_checkpoint(model, cfg.work_dir + "/final.pth")  
    print("Final (full network) training finished!")
