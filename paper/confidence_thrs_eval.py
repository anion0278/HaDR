import sys, os
path = os.path.abspath("./modified_packges")
sys.path.insert(0,path)
import numpy as np
import sys
import ws_specific_settings as wss
import common_settings as s
import cv2, regex as re
import model_utils as utils
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})
path_to_models = "E:\models\FINAL_TRAIN_ours" # s.path_to_models

ap_range = "0.50:0.95" # . does not have to be replaced with escaped version
regex_pattern = f"Min score: (\d.\d+)\n[\S\s]+?IoU=({ap_range}).+ = (\d.\d+)\n"

def make_graph(data):
    fig, ax = plt.subplots()
    markers = ["o","v","s","x"]
    lines = ["-",":","--"]
    colors = ["blue","green","gold","darkorange"]
    for i,[name,line] in enumerate(data):
        x,y = np.array(line).T
        ax.plot(x,y,marker = markers[i//3],markersize = 3,label = name,linestyle=lines[i%3],color=colors[i//3])
    ax.set_ylim(0,0.6)   
    ax.set_xlim(0,1)   
    ax.grid(visible=True)
    ax.set_xticks(np.arange(0,1.05,0.2))
    ax.set_yticks(np.arange(0,0.65,0.1))
    ax.set_xlabel("Confidence score threshold")
    ax.set_ylabel("AP@["+ap_range+"]")
    plt.subplots_adjust(right=0.7)
    plt.legend(bbox_to_anchor=(1.04,1),loc="upper left")
    plt.show()

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
        min_score = float(match.group(1))
        #ap_range = match.group(2)
        ap = float(match.group(3))
        single_score_thrs = [min_score, ap]
        model_data.append(single_score_thrs)
    data.append([f"{arch_name} ({channels_name})", model_data])

make_graph(data)

