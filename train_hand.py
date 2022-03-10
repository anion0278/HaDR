# SOLOv2 train  
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, init_detector, inference_detector
import os


if __name__ == '__main__':

    cfg = Config.fromfile('./configs/solov2/solov2_light_448_r50_fpn_custom.py') #works, batch size 32
    cfg.load_from = './checkpoints/epoch_2.pth' #SOLOv2_LIGHT_448_R50_3x

    # cfg = Config.fromfile('./configs/solov2/solov2_light_512_dcn_r50_fpn_custom.py')
    # cfg.load_from = './checkpoints/SOLOv2_LIGHT_512_DCN_R50_3x.pth'

    # cfg = Config.fromfile('./configs/solov2/solov2_light_448_r50_fpn_8gpu_3x.py') #works, batch size 12
    # cfg.load_from = './checkpoints/SOLOv2_Light_448_R50_3x.pth'

    cfg.dataset_type = 'CocoDataset'
    PREFIX = os.path.abspath('../datasets/rgbd_joined_dataset/ruka_2')
    cfg.data.train.img_prefix = PREFIX + "/color/"
    cfg.data.train.ann_file = PREFIX + '/instances_hands_train2022.json'
    cfg.data.train.type = 'CocoDataset'

    cfg.optimizer.lr = 0.0000000001
    cfg.lr_config.warmup = None

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
    cfg.device_ids = range(1)
    cfg.gpus = 1

    cfg.work_dir = "./checkpoints"
    cfg.total_epochs = 4
    cfg.checkpoint_config = dict(create_symlink=False, interval = 1)
    cfg.data.imgs_per_gpu = 32
    cfg.data.workers_per_gpu = 4

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

# 2022-03-10 13:03:54,250 - mmdet - INFO - Epoch [2][38/1279]     lr: 0.00001, eta: 0:17:22, time: 0.769, data_time: 0.295, memory: 2937, loss_ins: 0.4544, loss_cate: 0.3656, loss: 0.8200