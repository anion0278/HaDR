from mmdet.apis import init_detector, show_result_ins, predict_image
import time
import pyrealsense2 as rs
import numpy as np
import cv2


config_file = '../SOLO/configs/solov2/solov2_r101_fpn_custom.py'
checkpoint_file = '../SOLO/checkpoints/epoch_6.pth'

# config_file = '../SOLO/configs/solov2/solov2_light_448_r50_fpn_custom.py'
# checkpoint_file = '../SOLO/checkpoints/s2ch4_epoch_10.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

model.CLASSES = ["hand"]

def detect(img_rbgd):
    start = time.time()
    result = predict_image(model, img_rbgd)
    title = "Inference time: %.2f s" % (time.time() - start)
    img_res = show_result_ins(img_rbgd[:,:,0:3], result, model.CLASSES, score_thr=0.5)
    window_id = "win id"
    cv2.imshow(window_id, img_res)
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
depth_sensor.set_option(rs.option.visual_preset,5)

align_to = rs.stream.color
align = rs.align(align_to)

while(True):
    frames = pipeline.wait_for_frames() # original frames
    aligned_frames = align.process(frames) # aligned frames
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    rawdepth = np.asanyarray(depth_frame.get_data())
    min = 0.2
    max = 1.05
    mapped_depth = np.clip(rawdepth*depth_scale,min,max)
    norm = lambda n: n/max
    mapped_depth = 255-norm(mapped_depth)*255
    mapped_depth = mapped_depth[..., np.newaxis]
    rgbd = np.concatenate([color_image,mapped_depth.astype("uint8")],axis=2)
    detect(rgbd)
    
pipeline.stop()
