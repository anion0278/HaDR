import common_settings as s
from paper.email_notification import send_email
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
import os, argparse, traceback
import email_notification as outlook

import warnings
warnings.filterwarnings("ignore")  # disables annoying deprecation warnings

default_experiment_tag = "1C"
# default_arch_name = "fast_mask_rcnn_r101_fpn"
default_arch_name = "solov2_light_448_r50_fpn"
# default_arch_name = "mask_rcnn_r101_fpn"
#default_arch_name = "solov2_r101_fpn"
is_batchnorm_fixed = False
is_model_coco_pretrained = True
is_aug_enabled = True
default_channels = 4

frozen_epochs = 0
frozen_lr = 1e-4 
unfrozen_epochs = 20
unfrozen_lr = 1e-4

training_dataset = "sim_train_320x256" 
validation_dataset = "real_merged_l515_640x480"
dataset_size = "full" 

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

def train_stage(stage_name, cfg, frozen_backbone_stages, load_checkpoint, lr, epochs):
    cfg.load_from = load_checkpoint
    cfg.optimizer.lr = lr
    cfg.total_epochs = epochs
    cfg.model.backbone.frozen_stages = frozen_backbone_stages
    model = build_detector(cfg.model, train_cfg = cfg.train_cfg, test_cfg = cfg.test_cfg)
    model.CLASSES = datasets[0].CLASSES
    utils.store_json_config(cfg, f"{cfg.work_dir}/config_{stage_name}.json") 
    train_detector(model, datasets, cfg, distributed=False, validate=False, timestamp = timestamp)
    latest_checkpoint = f"{cfg.work_dir}/{stage_name}.pth" 
    save_checkpoint(model, latest_checkpoint)
    print(f"{stage_name} training finished")
    return latest_checkpoint

if __name__ == "__main__":
    try:
        args = parse_args()
        print(args)
        storage = wss.storage
        
        if TEST:
            dataset_size = "100" 
            wss.workers = 2
        train_dataset_path = s.path_to_datasets + training_dataset
        val_dataset_path =  s.path_to_datasets + validation_dataset
        main_channel =  utils.get_main_channel_name(args.channels)
        timestamp = dt.now().strftime("%a_%d_%b_%Hh_%Mm") 

        cfg = utils.get_config(args.arch, args.channels)

        policy = cfg.lr_config.policy
        if policy == "step": policy += str(cfg.lr_config.step)
        policy = policy.replace(" ", "").replace(",", "-")

        # should not contain commas ,
        config_id = f"{args.tag}-{args.arch}_{args.channels}ch-CocoPretrained={is_model_coco_pretrained}-DS={training_dataset}_{dataset_size}-Aug={args.aug}-BS={cfg.data.imgs_per_gpu}-BNfixed={is_batchnorm_fixed}"\
                + f"-FrozenEP={frozen_epochs}+LR={frozen_lr}-UnfrozenEP={unfrozen_epochs}_+LR={unfrozen_lr}-LRConfig={policy}-{timestamp}"

        print("CURRENT CONFIGURATION ID: " + config_id)
        cfg.work_dir = storage + ":/models/" + config_id
        os.makedirs(cfg.work_dir, exist_ok=True)

        cgf = manage_aug(cfg, args.aug)
        cfg.data.train.ann_file =  f"{train_dataset_path}/instances_hands_{dataset_size}.json"
        cfg.data.train.img_prefix =  f"{train_dataset_path}/{main_channel}/"
        cfg.data.val.ann_file = f"{val_dataset_path}/instances_hands_{dataset_size}.json"
        cfg.data.val.img_prefix = f"{val_dataset_path}/{main_channel}/"
        datasets = get_datasets(cfg)

        cfg.model.backbone.norm_eval = is_batchnorm_fixed

        # FULLY FROZEN BACKBONE: https://img1.21food.com/img/cj/2014/10/9/1412794284347212.jpg
        intermediate_chckp = train_stage(
            "intermediate",
            cfg, 4,
            f"{s.path_to_models}{args.arch}.pth" if is_model_coco_pretrained else None,
            frozen_lr,
            frozen_epochs)

        # NON-FROZEN
        train_stage(
            "final",
            cfg, -1,
            intermediate_chckp,
            unfrozen_lr,
            unfrozen_epochs)

        outlook.send_email("HGR: Training finished!", f"Finished training: {config_id}", wss.email_recipients)

    except Exception as ex:
        error_desc = str(ex) + "\n"+ "".join(traceback.TracebackException.from_exception(ex).format())
        print(f"Exception occured: {error_desc}")
        outlook.send_email("HGR: Exception during training!", error_desc, wss.email_recipients)
