import ws_specific_settings as wss
import os
import common_settings as s
s.add_packages_paths()
from mmcv import Config
from mmdet.apis import set_random_seed
import tkinter as tk
from tkinter import filedialog

def ask_user_for_dataset():
    options = {
        "Real cam" : "real_merged_l515_640x480",
        "Sim val" : "sim_val_320x256",
        "Sim train" : "sim_train_320x256"}
    val = s.DropDownMenuChoice().show_options(options, "Choose dataset:", default_value=wss.tested_dataset)
    print(f"Selected: {val}")
    return val

def ask_user_for_checkpoint(default_checkpoint_path):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    while True:
        default_checkpoint_path = filedialog.askdirectory(initialdir=default_checkpoint_path, title='Select a directory of trained model')
        checkpoint_path_full = os.path.join(default_checkpoint_path, s.tested_checkpoint_file_name)
        if os.path.exists(checkpoint_path_full):
            print(f"Selected {checkpoint_path_full}")
            break
        tk.messagebox.showerror(title="Choose different folder", message=f"Folder does not contain {s.tested_checkpoint_file_name}")
    root.destroy()
    return checkpoint_path_full

def get_pipelines(in_channels):
    from mmcv import Config
    options = {  
        1 : "pipelines_d",
        3 : "pipelines_rgb",
        4 : "pipelines_rgbd" }
    return Config.fromfile(s.path_to_configs_formatted % options[in_channels],'temp_pipe_config').data

def get_config(arch_name, channels):
    cfg = Config.fromfile(s.path_to_configs_formatted % arch_name, 'temp')
    cfg.model.backbone.in_channels = channels
    cfg.data = get_pipelines(cfg.model.backbone.in_channels)
    __set_config_params(cfg)
    return cfg

def parse_config_and_channels_from_checkpoint_path(checkpoint_path):
    import re, os
    matches = re.search(r"^\d[A-Za-z]+-(?P<arch>\w+)_(?P<channels>\d)ch", os.path.basename(checkpoint_path))
    return matches.group('arch'), int(matches.group('channels'))

def get_main_channel_name(channels):
    return "depth" if  channels == 1 else "color"

def __set_config_params(cfg):
    cfg.data.imgs_per_gpu = 32
    cfg.data.workers_per_gpu = wss.workers
    cfg.data.train.type = "CocoDataset"
    cfg.data.val.type = "CocoDataset"
    cfg.seed = 0
    set_random_seed(0, deterministic=True)
    cfg.workflow = [("train", 1), ("val", 1)] 

    cfg.device_ids = range(wss.gpus)
    cfg.gpu_ids = range(wss.gpus)
    cfg.gpus = wss.gpus
    cfg.dist_params = dict(backend='nccl')
    cfg.resume_from = None
    cfg.checkpoint_config = dict(create_symlink=False, interval = 4) # TRY TO USE state of optimizer save_optimizer = True

    cfg.log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
    cfg.log_level = 'INFO'

    if "pretrained" in cfg.model: cfg.model.pop("pretrained") # get rid of pretrained backbone since we will init weights from checkpoint
    cfg.lr_config = dict(policy="poly", power=0.9, min_lr=1e-7, by_epoch=False) # if by_epoch = False, then changes according to iteration
    cfg.optimizer = dict(type='SGD', lr=0.0002, momentum=0.9, weight_decay=0.001)
    #dict(type='Adam', lr=0.0003, weight_decay=0.0001)
    cfg.optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))