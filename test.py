from mmdet.apis import init_detector, inference_detector, show_result_pyplot
config_file = '../SOLO/configs/solov2/solov2_r101_fpn_custom.py'
#config_file = '../SOLO/configs/solov2/solov2_light_448_r50_fpn_custom.py'
checkpoint_file = '../SOLO/checkpoints/epoch_6.pth'
#checkpoint_file = '../SOLO/checkpoints/s2ch4_epoch_10.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
print("test")



from mmdet.apis import inference_detector, show_result_ins, predict_image
import pyrealsense2 as rs
import numpy as np
import os
import cv2

model.CLASSES = ["hand"]

def detect(img_rbgd):
    result = predict_image(model, img_rbgd)
    img_res = show_result_ins(img_rbgd[:,:,0:3], result, model.CLASSES, score_thr=0.5)
    cv2.imshow("inference", img_res)
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
i=0
while(True):
    frames = pipeline.wait_for_frames() # original frames
    aligned_frames = align.process(frames) # aligned frames
    depth_frame = aligned_frames.get_depth_frame()
    #depth_frame = frames.get_depth_frame()
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
    # img_path = os.path.abspath(r"C:/camera_image")
    #cv2.imwrite(os.path.join(img_path,"color",str(i)+".png"), color_image.astype("uint8"))
    #cv2.imwrite(os.path.join(img_path,"depth",str(i)+".png"), color_image.astype("uint8"))
    detect(rgbd)
    i = i+1
pipeline.stop()

#for f in ["425", "399", "292", "315", "497", "458", "302", "282", "372", ]:
# for f in ["12_131", "7_240", "8_196", "9_152", "12_131"]:
    #import os
    #img = os.path.abspath('../../datasets/real_cam/2_ruce_rukavice_250/color') + "/" + f+".png"
    #detect(img)
