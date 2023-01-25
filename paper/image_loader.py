import os, sys
path = os.path.abspath("..\HGR_CNN")
sys.path.insert(0,path)
import cv2
import numpy as np

def separate_color_from_depth(img_bgrd):
    return img_bgrd[:,:,0:3], img_bgrd[:,:,3:4] # color, depth

class ImageLoader():
    def __init__(self, image_dir):
        self.color_dir = os.path.join(image_dir,"color")
        self.depth_dir = os.path.join(image_dir,"depth")
        self.images = os.listdir(self.color_dir)
        self.iterator = iter(self.images)
        self.img_name = ""

    def get_rgbd_image(self):
        try:
            filename = next(self.iterator)
        except StopIteration:
            print("No more images")
            return None
        self.img_name = filename
        color_image = cv2.imread(os.path.join(self.color_dir,filename))
        depth = cv2.imread(os.path.join(self.depth_dir,filename),cv2.IMREAD_GRAYSCALE)
        depth = depth[..., np.newaxis]
        rgbd_img = np.concatenate([color_image,depth.astype("uint8")],axis=2)
        return rgbd_img
    
    def get_current_image_name(self):
        return self.img_name

    def close(self):
        self.pipeline.stop()
    