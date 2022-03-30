from mmdet.apis import init_detector, show_result_ins, predict_image
import time

import numpy as np
import cv2
import sys
sys.path.insert(0,r"C:\Users\C201_ALES\source\repos\hand_tracker\HGR_CNN")
import pyrealsense2 as rs

config_file = '../SOLO/configs/solov2/solov2_r101_fpn_custom.py'
checkpoint_file = '../SOLO/checkpoints/r101_e7_two_hands.pth'

# config_file = '../SOLO/configs/solov2/solov2_light_448_r50_fpn_custom.py'
# checkpoint_file = '../SOLO/checkpoints/s2ch4_epoch_10.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

model.CLASSES = ["hand"]

def detect(img_rbgd):
    start = time.time()
    result = predict_image(model, img_rbgd)
    title = "Inference time: %.2f s" % (time.time() - start)
    depth = img_rbgd[:,:,3:4]
    depth = np.stack((np.squeeze(depth),)*3, axis=-1)
    imgd_res = show_result_ins(depth, result, model.CLASSES, score_thr=0.5)
    img_res = show_result_ins(img_rbgd[:,:,0:3], result, model.CLASSES, score_thr=0.5)
    window_id = "win id"
    cv2.imshow(window_id, np.hstack([img_res, imgd_res]))
    cv2.setWindowTitle(window_id, title)
    cv2.waitKey(1)
    

img_camera_size = (640,480)
camera_rate = 30

pipeline = rs.pipeline()
pipeline_config = rs.config()

pipeline_config.enable_stream(rs.stream.depth, img_camera_size[0], img_camera_size[1], rs.format.z16, camera_rate)
pipeline_config.enable_stream(rs.stream.color, img_camera_size[0], img_camera_size[1], rs.format.bgr8, camera_rate)

profile = pipeline.start(pipeline_config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

#D515
# depth_sensor.set_option(rs.option.visual_preset,5)

#D435
filter_HF = rs.hole_filling_filter()
filter_HF.set_option(rs.option.holes_fill, 3)
colorizer = rs.colorizer()
colorizer.set_option(rs.option.visual_preset,1)
colorizer.set_option(rs.option.min_distance,0.2)        
colorizer.set_option(rs.option.max_distance,1.05)
colorizer.set_option(rs.option.color_scheme,2)
colorizer.set_option(rs.option.histogram_equalization_enabled,0)

align_to = rs.stream.color
align = rs.align(align_to)

while(True):
    frames = pipeline.wait_for_frames() # original frames
    aligned_frames = align.process(frames) # aligned frames
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    #D435
    filtered = filter_HF.process(depth_frame)
    filtered = colorizer.colorize(filtered)
    depth_colorized = np.asanyarray(filtered.get_data())
    depth_colorized = depth_colorized[:,:,0:1]
    #depth_colorized = depth_colorized[..., np.newaxis]
    
    #D515
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    rawdepth = np.asanyarray(depth_frame.get_data())
    min = 0.2
    max = 1.05
    mapped_depth = np.clip(rawdepth*depth_scale,min,max)
    norm = lambda n: n/max
    mapped_depth = 255-norm(mapped_depth)*255
    mapped_depth = mapped_depth[..., np.newaxis]

    #D515
    #rgbd = np.concatenate([color_image,mapped_depth.astype("uint8")],axis=2)

    #D435
    rgbd = np.concatenate([color_image,depth_colorized.astype("uint8")],axis=2)
    detect(rgbd)
    
pipeline.stop()
