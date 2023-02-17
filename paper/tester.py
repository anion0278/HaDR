import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # for skimage.measure
import argparse
import os
import os.path as osp
import sys, os
path = os.path.abspath("./modified_packges")
sys.path.insert(0,path)
import mmcv
import torch
from mmcv.parallel import MMDataParallel
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
import cv2
from os.path import exists
from eval_params import CustomizedEvalParams
path = os.path.abspath("./paper/third-party_solutions")
sys.path.insert(0,path)
import mediapipe_adapter

import warnings
warnings.filterwarnings("ignore")  # disables annoying deprecation warnings

TEST = True
eval_score_threshold = True
eval_mediapipe = True

default_min_score = 0.0

eval_dataset_annotations = "/instances_hands_full.json"
if TEST:
    eval_dataset_annotations = "/instances_hands_100.json" 
    wss.workers = 1

def mask_to_bbox(mask):
    x, y, w, h = cv2.boundingRect(mask)
    # xy, xy2 = (x,y), (x+w,y+h)
    # m = mask
    # m = cv2.rectangle(m, xy, xy2, (1,1,1), 1)
    # import matplotlib.pyplot as plt
    # fig = plt.figure(1)
    # ax1 = fig.add_subplot(221)
    # ax1.imshow(m)
    # plt.show()
    return x, y, x+w, y+h

def get_boxes_with_masks(result, num_classes):
    assert len(result) == 1
    masks = [[] for _ in range(num_classes)]
    bboxes = [[] for _ in range(num_classes)]
    for cur_result in result:
        if cur_result is None: break
        seg_pred = cur_result[0].cpu().numpy().astype(np.uint8)
        cate_label = cur_result[1].cpu().numpy().astype(np.int)
        cate_score = cur_result[2].cpu().numpy().astype(np.float)
        num_ins = seg_pred.shape[0]
        for idx in range(num_ins):
            class_label_id = cate_label[idx] 
            if class_label_id >= num_classes: continue
            cur_mask = seg_pred[idx, ...]
            rle = mask_util.encode(np.array(cur_mask[:, :, np.newaxis], order="F"))[0]
            bbox_with_score = [*mask_to_bbox(cur_mask[:, :]), cate_score[idx]] # same way as mask rcnn
            bboxes[class_label_id].append(bbox_with_score) 
            masks[class_label_id].append(rle)
    return np.array(bboxes), masks


def single_gpu_test(model, data_loader, show=False, verbose=True):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            seg_result = model(return_loss=False, rescale=not show, **data)
        if len(seg_result) == 1: # for SOLO
            segmentation_data = seg_result
            bboxes,masks = get_boxes_with_masks(segmentation_data, num_classes=len(dataset.CLASSES))
            seg_result = (bboxes, masks)

        results.append(seg_result)
        batch_size = data["img"][0].size(0)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Custom test detector")
    parser.add_argument("--checkpoint_path", help="checkpoint path", default=(None, s.path_to_models+wss.tested_model))
    return parser.parse_args()

def main():
    print(sys.argv)
    args = parse_args()
    eval_types = ["bbox", "segm"]

    if eval_mediapipe:
        model = mediapipe_adapter.MediaPipePredictor(default_min_score)
        cfg = utils.get_config("solov2_light_448_r50_fpn", 3) # name of arch is not important here
        checkpoint_path_full = s.path_to_models + "/mediapipe/out"
        eval_types = ["bbox"]
        eval_dataset = s.path_to_datasets + utils.ask_user_for_dataset() 
    else:
        if len(args.checkpoint_path) == 2:
            checkpoint_path_full = utils.ask_user_for_checkpoint(args.checkpoint_path[1])
            eval_dataset = s.path_to_datasets + utils.ask_user_for_dataset() 
        else:
            checkpoint_path_full = os.path.join(args.checkpoint_path, s.tested_checkpoint_file_name)
            eval_dataset = s.path_to_datasets + wss.tested_dataset
        arch, channels = utils.parse_config_and_channels_from_checkpoint_path(os.path.dirname(checkpoint_path_full))
        cfg = utils.get_config(arch, channels)
        # build the model and load checkpoint
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        while not osp.isfile(checkpoint_path_full):
            print("Waiting for {} to exist...".format(checkpoint_path_full))
            time.sleep(10) 
        # loading is required
        checkpoint = load_checkpoint(model, checkpoint_path_full, map_location='cuda:0')
        model = MMDataParallel(model, device_ids=[0])

    predictions_file = os.path.dirname(checkpoint_path_full) + "/out.pkl"

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    PREFIX = os.path.abspath(eval_dataset)
    annotations = eval_dataset_annotations
    
    # evaluating pretrained COCO model on ARMS datasets, because it has only "Person" class
    arms_annotations = eval_dataset_annotations.replace("hands", "arms")
    if cfg.model.bbox_head.num_classes == 81 and exists(PREFIX + arms_annotations): 
        annotations = arms_annotations
        print("Eval ARMS")
    
    cfg.data.test.ann_file = PREFIX + annotations
    cfg.data.test.img_prefix = f"{PREFIX}/{utils.get_main_channel_name(cfg.model.backbone.in_channels)}/"
    cfg.data.test.type = "CocoDataset"

    dataset = build_dataset(cfg.data.test)
    dataset.CLASSES = ["hand"] # for hands tests

    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=wss.workers,
        dist=False,
        shuffle=False)
    
    outputs = single_gpu_test(model, data_loader)
    result_files = store_results(predictions_file, dataset, outputs)

    print("Starting evaluation...")
    total_out_file = open(s.path_to_models + "evals.txt","a+")
    total_out_file.write(checkpoint_path_full + f" Dataset: {eval_dataset}\n")
    if eval_score_threshold:
        score_thrs_out_file = open(os.path.join(checkpoint_path_full, os.pardir, s.score_thrs_file_name),"w+")
        score_thrs_out_file.write(checkpoint_path_full + f" Dataset: {eval_dataset}\n")
        eval_predictions_in_score_range(dataset, eval_types, result_files, score_thrs_out_file, data_loader, predictions_file)
        score_thrs_out_file.close()
    eval_predicitons(dataset, eval_types, result_files, total_out_file, default_min_score)
    total_out_file.close()

def store_results(predictions_file, dataset, outputs):
    mmcv.dump(outputs, predictions_file)
    result_files = results2json(dataset, outputs, predictions_file)
    return result_files

def eval_predictions_in_score_range(dataset, eval_types, result_files, eval_out_file, data_loader, predictions_file):
    step = 0.025
    for min_score in np.arange(0.0, 1.0 + step, step):
        if eval_mediapipe: # mediapipe requires evaluation to be perfomed each time (because hand confidence is not the same value as min_detection_confidence)
            model = mediapipe_adapter.MediaPipePredictor(min_score)
            outputs = single_gpu_test(model, data_loader)
            result_files = store_results(predictions_file, dataset, outputs)
        eval_predicitons(dataset, eval_types, result_files, eval_out_file, min_score)

def eval_predicitons(dataset, eval_types, result_files, eval_out_file, min_score):
    eval_out_file.write(f"Min score: {min_score:.3f}\n")
    eval_params = CustomizedEvalParams(dataset.coco)
    coco_eval(result_files, eval_types, dataset.coco, file = eval_out_file, 
                override_eval_params = eval_params, classwise=True, min_score=min_score)


if __name__ == "__main__":
    main()
