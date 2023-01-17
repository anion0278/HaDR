import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # for skimage.measure
import argparse
import os
import os.path as osp
import shutil
import tempfile
import sys, os
path = os.path.abspath("./modified_packges")
sys.path.insert(0,path)
import mmcv
import torch
import torch.nn.functional as F
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import init_dist, get_dist_info, load_checkpoint
from mmdet.core import coco_eval, results2json, results2json_segm, wrap_fp16_model, tensor2imgs, get_classes
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
import time
import numpy as np
import pycocotools.mask as mask_util
import sys
import ws_specific_settings as wss
import model_utils as utils
import common_settings as s
from skimage.measure import label, regionprops, find_contours
import cv2

min_score = 0.5

checkpoint_path_full = 

eval_dataset = ""
eval_dataset_annotations = "/instances_hands_full.json"

arch, channels = utils.parse_config_and_channels_from_checkpoint_path(os.path.dirname(checkpoint_path_full))
cfg = utils.get_config(arch, channels)
cfg.data.test.test_mode = True

PREFIX = os.path.abspath(eval_dataset)
annotations = eval_dataset_annotations
from os.path import exists

# evaluating pretrained COCO model on ARMS datasets, because it has only "Person" class
arms_annotations = eval_dataset_annotations.replace("hands", "arms")
if cfg.model.bbox_head.num_classes == 81 and exists(PREFIX + arms_annotations): 
    annotations = arms_annotations
    print("Eval ARMS")

cfg.data.test.ann_file = dataset + ""
cfg.data.test.img_prefix = f"{dataset}/{utils.get_main_channel_name(cfg.model.backbone.in_channels)}/"
cfg.data.test.type =  "CocoDataset"


dataset = build_dataset(cfg.data.test)

eval_dest = os.path.join(checkpoint_path_full, os.path.pardir, "confidence_score_thrs_evals.txt")
f = open(eval_dest,"a+")

for min_score in [0.5, 0.65]:
    f.write(f"Threshold: {min_score}\n")
    from eval_params import CustomizedEvalParams
    eval_params = CustomizedEvalParams(dataset.coco)
    coco_eval(["out.pkl.segm.json"], ["segm"], dataset.coco, file = f, override_eval_params = eval_params, classwise=True, min_score=min_score)
f.close()