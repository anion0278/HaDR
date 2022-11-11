import os, sys
path = os.path.abspath("..\HGR_CNN")
sys.path.insert(0,path)
import pyrealsense2 as rs
import numpy as np

def separate_color_from_depth(img_bgrd):
    return img_bgrd[:,:,0:3], img_bgrd[:,:,3:4] # color, depth

class RgbdCamera():
    def __init__(self, resolution, rate):
        self.pipeline = rs.pipeline()
        pipeline_config = rs.config()

        pipeline_config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, rate)
        pipeline_config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, rate)

        profile = self.pipeline.start(pipeline_config)
        device = profile.get_device()
        self.device_type = str(device.get_info(rs.camera_info.product_line))

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        self.filter_HF = rs.hole_filling_filter()
        self.filter_HF.set_option(rs.option.holes_fill, 3)

        if(self.device_type == "L500"):
            depth_sensor.set_option(rs.option.visual_preset,5)
        else:
            self.filter_HF = rs.hole_filling_filter()
            self.filter_HF.set_option(rs.option.holes_fill, 3)
            self.colorizer = rs.colorizer()
            self.colorizer.set_option(rs.option.visual_preset,1)
            self.colorizer.set_option(rs.option.min_distance,0.2)        
            self.colorizer.set_option(rs.option.max_distance,1.0)
            self.colorizer.set_option(rs.option.color_scheme,2)
            self.colorizer.set_option(rs.option.histogram_equalization_enabled,0)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_rgbd_image(self):
        frames = self.pipeline.wait_for_frames() # original frames
        aligned_frames = self.align.process(frames) # aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())
        filtered = self.filter_HF.process(depth_frame)
        
        if(self.device_type == "L500"):
            rawdepth = np.asanyarray(filtered.get_data())
            min = 0.2
            max = 1.05
            mapped_depth = np.clip(rawdepth*self.depth_scale,min,max)
            norm = lambda n: n/max
            mapped_depth = 255-norm(mapped_depth)*255
            mapped_depth = mapped_depth[..., np.newaxis]
            rgbd_img = np.concatenate([color_image,mapped_depth.astype("uint8")],axis=2)
        else:
            filtered = self.colorizer.colorize(filtered)
            depth_colorized = np.asanyarray(filtered.get_data())
            depth_colorized = depth_colorized[:,:,0:1]
            rgbd_img = np.concatenate([color_image,depth_colorized.astype("uint8")],axis=2)
        return rgbd_img

    def close(self):
        self.pipeline.stop()
    