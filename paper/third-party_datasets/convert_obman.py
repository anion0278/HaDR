import cv2, struct
import numpy as np
import itertools
from PIL import Image
import numpy as np
import pickle
import os
from joblib import Parallel,delayed


# two hands 00149537.png
dataset_dir = "D:\datasets\_ObMan_orig"

class_name = "hand"

#see https://github.com/hassony2/obman_train/blob/master/handobjectdatasets/obman.py

def process_img(sample_id, id, files_len, dataset_dir):
    print(f"Processing {id} from {files_len}")
    try:

        depth = cv2.imread(os.path.join(dataset_dir, "depth", sample_id), 1)
        with (open(os.path.join(dataset_dir, "meta", sample_id.replace("png", "pkl")), "rb")) as openfile:
            meta = pickle.load(openfile)
        depth_m = (depth[:, :, 0] - 1) / 254 * (meta["depth_min"] - meta["depth_max"]) + meta["depth_max"]
        depth_over_limit = abs(depth_m - meta["depth_min"])
        depth_m[depth_over_limit < 0.005] = 1
        depth_m[depth_m>1] = 1
        depth_byte = 255-depth_m * 255
        cv2.imwrite(os.path.join(dataset_dir, "depth_formatted", sample_id), depth_byte)

        #class 20 - forehand
        #class 1..20 - arm parts
        #class 21..99 - hand parts 
        #24 right fingers, 22 - right palm
        # 23 left fingers, 21 - left palm
        # 100 - grasped object
        #class 0 - background
        mask = cv2.imread(os.path.join(dataset_dir, "segm", sample_id), cv2.IMREAD_UNCHANGED)
        mask = mask[:, :, 0] 
        mask[mask<20]=0
        mask[mask==20]=0
        mask[mask==100]=0
        mask_l = mask
        mask_r = mask.copy()
        mask_r[mask_r % 2 == 0]=0
        mask_l[mask_l % 2 != 0]=0
        mask_r[mask_r != 0]=255
        mask_l[mask_l != 0]=255
        cv2.imwrite(os.path.join(dataset_dir, "mask_formatted", class_name+"_"+sample_id.replace(".png", "_i1.png")), mask_l)
        cv2.imwrite(os.path.join(dataset_dir, "mask_formatted", class_name+"_"+sample_id.replace(".png", "_i2.png")), mask_r)

        color = cv2.imread(os.path.join(dataset_dir, "rgb", sample_id.replace(".png",".jpg")), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(os.path.join(dataset_dir, "color_formatted", class_name+"_"+sample_id.replace(".png", "_i2.png")), color)

        # import matplotlib.pyplot as plt
        # fig = plt.figure(1)
        # ax1 = fig.add_subplot(221)
        # ax2 = fig.add_subplot(222)
        # ax3 = fig.add_subplot(223)
        # ax1.imshow(depth_m)
        # ax2.imshow(mask_l)
        # ax3.imshow(mask_r)
        # plt.show()

    except Exception as ex:
        print("Some images dont have corresponding depth")
        return
        
samples = os.listdir(os.path.join(dataset_dir, "depth"))
samples_len = int(len(samples))
Parallel(n_jobs=5)(delayed(process_img)(samples[sample_id], sample_id, samples_len, dataset_dir) for sample_id in range(samples_len))
