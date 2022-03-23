# SOLOv2 train  
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import os


if __name__ == '__main__':

    cfg = Config.fromfile('./configs/solov2/solov2_r101_fpn_custom.py') #works, batch size 32
    cfg.load_from = './checkpoints/epoch_2.pth' #SOLOv2_LIGHT_448_R50_3x 

    # cfg = Config.fromfile('./configs/mask_rcnn_r101_fpn_custom.py') 
    # cfg.load_from = './checkpoints/epoch_1.pth'  

    # cfg = Config.fromfile('./configs/solov2/solov2_light_448_r50_fpn_custom.py') #works, batch size 32
    # cfg.load_from = './checkpoints/s2ch4_epoch_9.pth' #SOLOv2_LIGHT_448_R50_3x 
    # is is possible to dynamically download Pretrained model - the string should start with "modelzoo://"

    # cfg = Config.fromfile('./configs/solov2/solov2_light_448_r50_fpn_8gpu_3x.py') #works, batch size 12
    # cfg.load_from = './checkpoints/SOLOv2_Light_448_R50_3x.pth'

    # cfg = Config.fromfile('./configs/solov2/solov2_light_512_dcn_r50_fpn_custom.py') # does not work with 4 channels
    # cfg.load_from = './checkpoints/SOLOv2_LIGHT_512_DCN_R50_3x.pth'

    cfg.dataset_type = 'CocoDataset'
    # PREFIX = os.path.abspath('H:\image_cam')
    # cfg.data.train.ann_file = PREFIX + '/instances_hands_train2022.json'  # 3800 30600 101200
    PREFIX = os.path.abspath('H:/dataset_9k_448_256_noblur')
    cfg.data.train.ann_file = PREFIX + '/instances_hands_6000.json'  # 3800 30600 101200
    # PREFIX = os.path.abspath('G:/datasety/new')
    # cfg.data.train.ann_file = PREFIX + '/instances_hands_3800.json'  # 3800 30600 101200

    # PREFIX = os.path.abspath('../datasets/rgbd_joined_dataset/ruka_2')
    # cfg.data.train.ann_file = PREFIX + '/instances_hands_train2022.json'

    cfg.data.train.img_prefix = PREFIX + "/color/"
    cfg.data.train.type = 'CocoDataset'

    cfg.optimizer.lr = 0.00003
    # cfg.lr_config.warmup = None

    # Set seed thus the results are more reproducible
    cfg.seed = 22
    set_random_seed(22, deterministic=False)
    cfg.gpu_ids = range(1)
    cfg.device_ids = range(1)
    cfg.gpus = 1

    cfg.work_dir = "./checkpoints"
    cfg.total_epochs = 6
    cfg.checkpoint_config = dict(create_symlink=False, interval = 2)

    cfg.log_config = dict(
        interval=1,
        hooks=[
            dict(type='TextLoggerHook'),
            # dict(type='TensorboardLoggerHook')
        ])

    model = build_detector(cfg.model, train_cfg = cfg.train_cfg, test_cfg = cfg.test_cfg)
    datasets = [build_dataset(cfg.data.train)]
    datasets[0].CLASSES = ["hand"]
    model.CLASSES = datasets[0].CLASSES

    # CHECK OUT frozen_stages=0 for fine-tunning !!
    # try resnext backbone

    train_detector(model, datasets[0], cfg, distributed=False, validate=False)
    print("Training finished")

# 2022-03-10 13:03:54,250 - mmdet - INFO - Epoch [2][38/1279]     lr: 0.00001, eta: 0:17:22, time: 0.769, data_time: 0.295, memory: 2937, loss_ins: 0.4544, loss_cate: 0.3656, loss: 0.8200