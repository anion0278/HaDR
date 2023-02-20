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
path_to_models = "D:/models/mediapipe_eval" # s.path_to_models

ap_range = "0.50:0.95" # . does not have to be replaced with escaped version
regex_pattern = f"Min score: (\d.\d+)\n[\S\s]+?IoU=({ap_range}).+ = (\d.\d+)\n"

def read_ap_data(path_to_ap_scores_file, config):
    f = open(path_to_ap_scores_file, "r")
    file_content = f.read()
    try:
        arch, channels = utils.parse_config_and_channels_from_checkpoint_path(config)
        arch_name = utils.get_arch_translation(arch)
        channels_name = utils.get_channels_translation(channels)
    except:
        arch_name = "MediaPipe"
        channels_name = "RGB"
    model_data = []
    for match in re.finditer(regex_pattern, file_content):
        min_score = float(match.group(1))
        #ap_range = match.group(2)
        ap = float(match.group(3))
        single_score_thrs = [min_score, ap]
        model_data.append(single_score_thrs)
    return [f"{arch_name} ({channels_name})", model_data]

def eval_ap():
    for model_dir in s.list_all_dirs_only(path_to_models):
        if (re.search("^\d[A-Za-z0-9]+-",model_dir)):
            dirname = os.path.join(path_to_models, model_dir)
            if (os.path.exists(os.path.join(dirname,"final.pth"))):
                command = f"python paper/tester.py --checkpoint_path {dirname}"
                print(command)
                os.system(command)


if __name__ == "__main__":

    all_models_data = []
    for dir in s.list_all_dirs_only(path_to_models):
        model_data = read_ap_data(os.path.join(path_to_models, dir, s.score_thrs_file_name), dir)
        all_models_data.append(model_data)

    utils.draw_score_thrs_graph(
        all_models_data,
        "Confidence score threshold",
        "AP@["+ap_range+"]",
        0.6,
        np.arange(0,0.65,0.1)
    )

