import time
import numpy as np
import cv2, os
import ws_specific_settings as wss
import model_utils as utils
import image_loader

import common_settings as s
s.add_packages_paths()
from mmdet.apis import init_detector, show_result_ins, predict_image, show_result_pyplot

default_checkpoint_dir = wss.tested_model
dataset_dir = s.path_to_datasets + wss.tested_dataset
threshold = s.visualization_threshold

def get_tested_image(input_channels, img_bgrd):
    options = { 1: img_bgrd[:,:,3:4], 
                3: img_bgrd[:,:,0:3],
                4: img_bgrd }
    return options[input_channels]

def detect(img_bgrd, arch):
    start = time.time()
    tested_image = get_tested_image(cfg.model.backbone.in_channels, img_bgrd)
    #tested_image = cv2.resize(tested_image,[320,256])
    result = predict_image(model, tested_image)
    elapsed_time = time.time() - start
    title = f"Inference time: {elapsed_time:2.2f}s, FPS: {(1/elapsed_time):2.0f}" 
    color, depth = image_loader.separate_color_from_depth(img_bgrd)
    depth = np.stack((np.squeeze(depth),)*3, axis=-1)
    ins_visualization = show_result_pyplot if "mask" in arch else show_result_ins # mask rcnn requires different visualization
    res_img_bgrb = ins_visualization(color, result, model.CLASSES, score_thr=threshold)
    res_img_d = ins_visualization(depth, result, model.CLASSES, score_thr=threshold)
    window_id = "win id"
    cv2.imshow(window_id, np.hstack([res_img_bgrb, res_img_d]))
    cv2.setWindowTitle(window_id, title)
    cv2.waitKey(0)
    

if __name__ == "__main__":
    checkpoint_path_full = utils.ask_user_for_checkpoint(s.path_to_models + default_checkpoint_dir)
    arch, channels = utils.parse_config_and_channels_from_checkpoint_path(os.path.dirname(checkpoint_path_full))
    cfg = utils.get_config(arch, channels)
    model = init_detector(cfg, checkpoint_path_full, device='cuda:0')
    if len(model.CLASSES) > 1: 
         print(f"Overrinding the model classes! Current classes: {len(model.CLASSES)}")
         model.CLASSES = ["person"] if cfg.model.bbox_head.num_classes == 81 else ["hand"]
            
    loader = image_loader.ImageLoader(dataset_dir)
    while(True):
        detect(loader.get_rgbd_image(), arch)

    camera.close() # TODO 
