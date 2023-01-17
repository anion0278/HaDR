import sys, os
path = os.path.abspath("./modified_packges")
sys.path.insert(0,path)
import numpy as np
import sys
import ws_specific_settings as wss
import common_settings as s
import cv2, regex as re
import model_utils as utils


path_to_models = "E:\models\FINAL_TRAIN_ours" # s.path_to_models

ap_range = "0.50:0.95" # . does not have to be replaced with escaped version
regex_pattern = f"Min score: (\d.\d+)\n[\S\s]+?IoU=({ap_range}).+ = (\d.\d+)\n"

arch_translation = {
    "mask_rcnn_r50_fpn" : "Mask R-CNN ResNet50",
    "mask_rcnn_r101_fpn" : "Mask R-CNN ResNet101",
    "solov2_light_448_r50_fpn" : "SOLOv2 ResNet50",
    "solov2_r101_fpn" : "SOLOv2 ResNet101"
}

channels_translation = {
    1 : "Depth",
    3 : "RGB",
    4 : "RGB-D",
}

data = []

for dir in s.list_all_dirs_only(path_to_models):
    f = open(os.path.join(path_to_models, dir, s.score_thrs_file_name), "r")
    file_content = f.read()
    arch, channels = utils.parse_config_and_channels_from_checkpoint_path(dir)
    arch_name = arch_translation[arch]
    channels_name = channels_translation[channels]
    model_data = []
    for match in re.finditer(regex_pattern, file_content):
        min_score = match.group(1)
        #ap_range = match.group(2)
        ap = match.group(3)
        single_score_thrs = [min_score, ap]
        model_data.append(single_score_thrs)
    data.append([f"{arch_name}: {channels_name}", model_data])

print(data)