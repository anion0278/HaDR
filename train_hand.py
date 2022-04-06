# SOLOv2 train  
import os, sys
path = os.path.abspath("./modified_packges")
sys.path.insert(0,path)
import mmcv, imagecorruptions
from mmcv import Config
from mmcv.runner import save_checkpoint
print(os.path.abspath(mmcv.__file__))
#print(os.path.abspath(imagecorruptions.__file__))

from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import time
from datetime import datetime as dt

import warnings
warnings.filterwarnings("ignore")  # disables annoying deprecation warnings


if __name__ == '__main__':

    storage = "G"
    arch_name = "solov2_r101_fpn"

    cfg = Config.fromfile('./paper/tested_configs/' + arch_name + '_custom.py')
    cfg.work_dir = storage + ":/models/"
    cfg.load_from = cfg.work_dir + arch_name + '.pth'
    # cfg.load_from = './checkpoints/r101_e6.pth' 

    PREFIX = os.path.abspath(storage + ':/datasets/dataset9x_matte+2hands')
    cfg.data.train.ann_file = PREFIX + '/instances_hands_full.json' 

    cfg.data.train.img_prefix = PREFIX + "/color/"
    cfg.data.train.type = 'CocoDataset'

    VAL_PREFIX =  storage + ":/datasets/sim_validation_dataset"
    cfg.data.val.ann_file = VAL_PREFIX + '/instances_hands_200.json'  
    cfg.data.val.img_prefix = VAL_PREFIX + "/color/"
    cfg.data.val.pipeline = cfg.val_pipeline
    cfg.data.val.type = 'CocoDataset'

    datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.val)]
    datasets[0].CLASSES = ["hand"]
    datasets[1].CLASSES = datasets[0].CLASSES

    cfg.seed = 0
    set_random_seed(0, deterministic=True)
    cfg.workflow = [('train', 1), ('val', 1)]

# POZOR ROZLISENI VSTUPNICH DAT SE ZMENILO
    cfg.data.imgs_per_gpu = 8
    cfg.lr_config.warmup_iters = 250 # should be ~ equal to single epoch
    cfg.checkpoint_config = dict(create_symlink=False, interval = 4) # TRY TO USE state of optimizer save_optimizer = True
    cfg.log_config.interval = 1

    cfg.optimizer.lr = 1e-4
    cfg.model.backbone.frozen_stages = 0
    cfg.total_epochs = 36


# FULLY FROZEN BACKBONE: https://img1.21food.com/img/cj/2014/10/9/1412794284347212.jpg

    #cfg.model.pop("pretrained") # get rid of pretrained backbone since we will init weights from checkpoint
    model = build_detector(cfg.model, train_cfg = cfg.train_cfg, test_cfg = cfg.test_cfg)
    model.CLASSES = datasets[0].CLASSES # needed for the first time
    print("this line is needed to make line-cleaning command working")
    train_detector(model, datasets, cfg, distributed=False, validate=False, timestamp = dt.now().strftime("%a_%d_%m_%y"))
    latest_checkpoint = cfg.work_dir + "lastest_" + arch_name +  ".pth"
    save_checkpoint(model, latest_checkpoint)
    print("Pre-training finished")

# PARTIALLY FROZEN

#     cfg.load_from = latest_checkpoint
#     cfg.optimizer.lr = 1e-5
#     cfg.lr_config.policy = "step" # this is needed because dict.pop() method is used in mmcv\runner\runner.py", line 366, in register_lr_hook
#     cfg.model.backbone.frozen_stages = 1
#     cfg.total_epochs = 5
#     model = build_detector(cfg.model, train_cfg = cfg.train_cfg, test_cfg = cfg.test_cfg)
#     train_detector(model, datasets, cfg, distributed=False, validate=False)
#     save_checkpoint(model, latest_checkpoint)
#     print("Intermediate training finished")

# # NON-FROZEN

#     cfg.load_from = latest_checkpoint
#     cfg.optimizer.lr = 1e-6
#     cfg.lr_config.policy = "step"
#     cfg.model.backbone.frozen_stages = 0
#     cfg.total_epochs = 8
#     model = build_detector(cfg.model, train_cfg = cfg.train_cfg, test_cfg = cfg.test_cfg)
#     train_detector(model, datasets, cfg, distributed=False, validate=False)
#     save_checkpoint(model, cfg.work_dir + "final_" + arch_name + ".pth")
#     print("Final (full network) training finished!")
