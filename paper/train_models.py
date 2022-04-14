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

import warnings
warnings.filterwarnings("ignore")  # disables annoying deprecation warnings


if __name__ == '__main__':
 
    TEST = False  # if True runs only 100 same images from validation dataset for BOTH TRAIN and VAL

    storage = wss.storage

    experiment_tag = "1B"
    arch_name = "solov2_r101_fpn"
    timestamp = dt.now().strftime("%a_D%d_M%m_%Hh_%Mm") 

    training_dataset = "dataset9x_matte+2hands" 
    validation_dataset = "sim_validation_dataset"
    dataset_size = "full" 
    if TEST:
        dataset_size = "100" 
        training_dataset = validation_dataset
    train_dataset_path = storage + ':/datasets/' + training_dataset
    val_dataset_path =  storage + ":/datasets/" + validation_dataset

    output_dir = experiment_tag + "-" + arch_name + "-" + training_dataset + dataset_size + "-" + timestamp

    print("CURRENT CONFIGURATION ID: " + output_dir)

    cfg = Config.fromfile('./paper/tested_configs/' + arch_name + '_custom.py')
    cfg.load_from = storage + ":/models/" + arch_name + '.pth'
    cfg.work_dir = storage + ":/models/" + output_dir
    os.makedirs(cfg.work_dir, exist_ok=True)

    cfg.data.train.ann_file = train_dataset_path + '/instances_hands_%s.json' % dataset_size 
    cfg.data.train.img_prefix = train_dataset_path + "/color/"
    cfg.data.train.type = 'CocoDataset'

    cfg.data.val.ann_file = val_dataset_path + '/instances_hands_%s.json' % dataset_size 
    cfg.data.val.img_prefix = val_dataset_path + "/color/"
    cfg.data.val.pipeline = cfg.val_pipeline
    cfg.data.val.type = 'CocoDataset'

    datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.val)]
    datasets[0].CLASSES = ["hand"]
    datasets[1].CLASSES = datasets[0].CLASSES

    cfg.seed = 0
    set_random_seed(0, deterministic=True)
    cfg.workflow = [('train', 1), ('val', 1)] 

    cfg.data.imgs_per_gpu = 4
    cfg.data.workers_per_gpu = wss.workers
    cfg.checkpoint_config = dict(create_symlink=False, interval = 1) # TRY TO USE state of optimizer save_optimizer = True
    cfg.log_config.interval = 1

    cfg.optimizer.lr = 1e-4
    cfg.model.backbone.frozen_stages = 0
    cfg.total_epochs = 2
    cfg.lr_config = dict(policy='poly', power=0.9, min_lr=1e-7, by_epoch=False) # if by_epoch = False, then changes according to iteration

# FULLY FROZEN BACKBONE: https://img1.21food.com/img/cj/2014/10/9/1412794284347212.jpg

    cfg.model.pop("pretrained") # get rid of pretrained backbone since we will init weights from checkpoint
    model = build_detector(cfg.model, train_cfg = cfg.train_cfg, test_cfg = cfg.test_cfg)
    model.CLASSES = datasets[0].CLASSES # needed for the first time
    print("this line is needed to make line-cleaning command working")
    train_detector(model, datasets, cfg, distributed=False, validate=False, timestamp = timestamp)
    latest_checkpoint = cfg.work_dir + "/intermediate_" + arch_name +  ".pth"
    save_checkpoint(model, latest_checkpoint)
    print("Intermediate training finished")

# NON-FROZEN

    cfg.load_from = latest_checkpoint
    cfg.optimizer.lr = 1e-5
    cfg.lr_config.policy = "step"
    cfg.model.backbone.frozen_stages = 0
    cfg.total_epochs = 2
    model = build_detector(cfg.model, train_cfg = cfg.train_cfg, test_cfg = cfg.test_cfg)
    train_detector(model, datasets, cfg, distributed=False, validate=False)
    save_checkpoint(model, cfg.work_dir + "/final_" + arch_name + ".pth")
    print("Final (full network) training finished!")
