# SOLOv2 train  
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, init_detector, inference_detector
import os


if __name__ == '__main__':

    cfg = Config.fromfile('./configs/solov2/solov2_light_512_dcn_r50_fpn_custom.py')
    cfg.load_from = './checkpoints/SOLOv2_LIGHT_512_DCN_R50_3x.pth'

    cfg.dataset_type = 'CocoDataset'
    PREFIX = os.path.abspath('../HGR_CNN/datasets/rgbd_joined_dataset/ruka_2')
    cfg.data.train.img_prefix = PREFIX + "/color/"
    cfg.data.train.ann_file = PREFIX + '/instances_hands_train2022.json'
    cfg.data.train.type = 'CocoDataset'

    cfg.optimizer.lr = 0.00001
    cfg.lr_config.warmup = None

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
    cfg.device_ids = range(1)
    cfg.gpus = 1

    cfg.work_dir = "./checkpoints"
    cfg.total_epochs = 2
    cfg.checkpoint_config = dict(create_symlink=False, interval = 1)
    cfg.data.imgs_per_gpu = 16
    cfg.data.workers_per_gpu = 1

    cfg.log_config = dict(
        interval=1,
        hooks=[
            dict(type='TextLoggerHook'),
            dict(type='TensorboardLoggerHook')
        ])

    model = build_detector(cfg.model)
    datasets = [build_dataset(cfg.data.train)]
    datasets[0].CLASSES = ["hand", "dummy"]
    model.CLASSES = datasets[0].CLASSES

    # CHECK OUT frozen_stages=0 for fine-tunning !!
    # try resnext backbone

    train_detector(model, datasets[0], cfg, distributed=False, validate=False)
    print("Training finished")
