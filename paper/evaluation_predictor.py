import time
import numpy as np
import cv2
import ws_specific_settings as wss
import model_utils as utils
import image_loader

import common_settings as s
s.add_packages_paths()
from mmdet.apis import init_detector, show_result_ins, predict_image, show_result_pyplot

image_dir = r"C:\datasets\real_merged_l515_640x480"
# checkpoint_dir = "2G-solov2_light_448_r50_fpn_4ch-sim_train_320x256_full-AugTrue-10+20ep"
checkpoint_dir = "2R-solov2_r101_fpn_3ch-real_merged_l515_640x480_full-AugTrue-10+20ep-Wed_D18_M05_10h_14m"


def get_tested_image(input_channels, img_bgrd):
    options = { 1: img_bgrd[:,:,3:4], 
                3: img_bgrd[:,:,0:3],
                4: img_bgrd }
    return options[input_channels]

def detect(img_bgrd, arch):
    start = time.time()
    tested_image = get_tested_image(cfg.model.backbone.in_channels, img_bgrd)
    result = predict_image(model, tested_image)
    elapsed_time = time.time() - start
    title = f"Inference time: {elapsed_time:2.2f}s, FPS: {(1/elapsed_time):2.0f}" 
    color, depth = image_loader.separate_color_from_depth(img_bgrd)
    depth = np.stack((np.squeeze(depth),)*3, axis=-1)
    ins_visualization = show_result_pyplot if "mask" in arch else show_result_ins # mask rcnn requires different visualization
    res_img_bgrb = ins_visualization(color, result, model.CLASSES, score_thr=0.10)
    res_img_d = ins_visualization(depth, result, model.CLASSES, score_thr=0.10)
    window_id = "win id"
    cv2.imshow(window_id, np.hstack([res_img_bgrb, res_img_d]))
    cv2.setWindowTitle(window_id, title)
    cv2.waitKey(0)
    

if __name__ == "__main__":
    checkpoint_path = wss.storage + ":/models/" + checkpoint_dir 
    arch, channels = utils.parse_config_and_channels_from_checkpoint_path(checkpoint_path)
    cfg = utils.get_config(arch, channels)
    model = init_detector(cfg, checkpoint_path + "/" + s.tested_checkpoint_file_name, device='cuda:0')
    if len(model.CLASSES) > 1: 
        print(f"Overrinding the model classes! Current classes: {model.CLASSES}")
        model.CLASSES = ["person"] if cfg.model.bbox_head.num_classes == 81 else ["hand"]
            
    loader = image_loader.ImageLoader(image_dir)
    while(True):
        detect(loader.get_rgbd_image(), arch)

    camera.close() # TODO 
