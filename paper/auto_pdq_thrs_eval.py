import subprocess, os, shutil, re, sys
import numpy as np
import common_settings as s
import model_utils as utils
sys.path.insert(0,os.path.abspath("./paper/pdq"))
import read_files

path_to_models = s.path_to_models + "/mediapipe_eval/"
dataset_anns_path = s.path_to_datasets + "real_merged_l515_640x480/instances_hands_full.json"
output_file_name = "pdq_score_threshold_evals.txt"
coco_dets_file_name = "out.pkl.bbox.json"
rvc_dets_file_name = "rvc1_det.json"

reevaluate_map = False
eval_mediapipe = True
if eval_mediapipe: reevaluate_map = False

regex_pattern = f"threshold: (\d.\d+) - PDQ: (\d.\d+) - avg_pPDQ: (\d.\d+) - spatial_PDQ: (\d.\d+) - label_PDQ: (\d.\d+) - mAP: (\d.\d+) - TP: (\d+) - FP: (\d+) - FN: (\d+)"

def plot_data():
    data = []
    for dir in s.list_all_dirs_only(path_to_models):
        data.append(read_pdq_data(os.path.join(path_to_models, dir, output_file_name), dir))
    
    utils.draw_score_thrs_graph(
        data,
        "Confidence score threshold",
        "PDQ [-]",
        0.2,
        np.arange(0,0.201,0.025))

def read_pdq_data(path_to_pdq_file, config):
    f = open(path_to_pdq_file, "r")
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
        pdq = float(match.group(2))
            # avg_pPDQ = float(match.group(3))
            # spatial_PDQ = float(match.group(4))
            # label_PDQ = float(match.group(5))
            # map = float(match.group(6))
            # tp = float(match.group(7))
            # fp = float(match.group(8))
            # fn = float(match.group(9))
        single_score_thrs = [min_score, pdq]
        model_data.append(single_score_thrs)
    return [f"{arch_name} ({channels_name})", model_data]

def eval_pdq(model_dir):
    if reevaluate_map:
        # generate predictions and evaluate mAP 
        os.system(f"python paper/tester.py --checkpoint_path {model_dir}")

    out_file_path = os.path.join(model_dir, output_file_name)

    total_out_file = open(out_file_path,"w+")
    print("Evaluating PDQ for: " + model_dir)
    total_out_file.write("Evaluating PDQ for: " + model_dir)
    total_out_file.close()

    step = 0.025
    for min_score in np.arange(0.0, 1.0 + step, step):
        total_out_file = open(out_file_path,"a+")
        print(f"\nScore threshold: {min_score:.3f}")
        total_out_file.write(f"\nScore threshold: {min_score:.3f}")
        total_out_file.close()
        eval_pdq_for_score_threshold(model_dir, min_score, out_file_path)
    total_out_file.close()


def prepare_pdq_dir(model_dir):
    pdq_eval_dir = os.path.join(model_dir,"pdq")
    if os.path.exists(pdq_eval_dir):
        shutil.rmtree(pdq_eval_dir) # remove all previous files
    os.makedirs(pdq_eval_dir)
    return pdq_eval_dir


def eval_pdq_for_score_threshold(model_dir, min_score_thrs, out_file_path):
    pdq_eval_dir = prepare_pdq_dir(model_dir)

    if eval_mediapipe:
        os.system(f"python paper/tester.py --mediapipe --score_thrs {min_score_thrs}")

    read_files.convert_coco_det_to_rvc_det(
        f"{model_dir}/{coco_dets_file_name}", 
        dataset_anns_path, 
        f"{pdq_eval_dir}/{rvc_dets_file_name}", 
        min_score_thrs)

    subprocess.call(["python", "paper/pdq/evaluate.py", 
            "--test_set", "coco", 
            "--gt_loc", dataset_anns_path, 
            "--det_loc", f"{pdq_eval_dir}/{rvc_dets_file_name}", 
            "--save_folder", pdq_eval_dir,
            "--out_loc", out_file_path])


if __name__ == "__main__":

    # for dir in s.list_all_dirs_only(path_to_models):
    #     eval_pdq(os.path.join(path_to_models, dir))

    plot_data()