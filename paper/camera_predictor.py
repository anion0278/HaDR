import time
import numpy as np
import cv2
import ws_specific_settings as wss
import model_utils as utils
import camera

import common_settings as s
s.add_packages_paths()
from mmdet.apis import init_detector, show_result_ins, predict_image, show_result_pyplot

# arch_name = "fast_mask_rcnn_r101_fpn"
# arch_name = "solov2_light_448_r50_fpn"
arch_name = "mask_rcnn_r101_fpn"
# arch_name = "solov2_r101_fpn"
tested_model_dir = "maskrcnn-test"
channels = 4

def get_tested_image(input_channels, img_rbgd):
    options = { 1: img_rbgd[:,:,3:4], 
                3: img_rbgd[:,:,0:3],
                4: img_rbgd }
    return options[input_channels]

def detect(img_rbgd):
    start = time.time()
    tested_image = get_tested_image(cfg.model.backbone.in_channels, img_rbgd)
    result = predict_image(model, tested_image)
    elapsed_time = time.time() - start
    title = f"Inference time: {elapsed_time:2.2f}s, FPS: {(1/elapsed_time):2.0f}" 
    depth = img_rbgd[:,:,3:4]
    depth = np.stack((np.squeeze(depth),)*3, axis=-1)
    ins_visualization = show_result_pyplot if "mask" in arch_name else show_result_ins # mask rcnn requires different visualization
    res_img_rgb = ins_visualization(img_rbgd[:,:,0:3], result, model.CLASSES, score_thr=0.5)
    res_img_d = ins_visualization(depth, result, model.CLASSES, score_thr=0.5)
    window_id = "win id"
    cv2.imshow(window_id, np.hstack([res_img_rgb, res_img_d]))
    cv2.setWindowTitle(window_id, title)
    cv2.waitKey(1)
    

if __name__ == "__main__":
    config_file = f'../SOLO/paper/tested_configs/{arch_name}.py' 
    cfg = utils.get_config(arch_name, channels)
    checkpoint_file = wss.storage + ":/models/" + tested_model_dir +"/final.pth" 
    model = init_detector(cfg, checkpoint_file, device='cuda:0')
    cam = camera.RgbdCamera((640,480), 30)
    while(True):
        detect(cam.get_rgbd_image())

    pipeline.stop() # TODO 
