""" Basic example showing samples from the dataset.

    Chose the set you want to see and run the script:
    > python view_samples.py

    Close the figure to see the next sample.
"""
from __future__ import print_function, unicode_literals

import pickle
import os
from pickletools import uint8
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio, cv2
from joblib import Parallel,delayed
import time


subset = "evaluation"
set_path = 'D:/datasets/RHD_published_v2/' + subset

class_name = "hand"

# auxiliary function
def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 2**8 + bottom_bits).astype('float32')
    depth_map /= float(2**16 - 1)
    depth_map *= 5.0
    return depth_map

def process_img(sample_id, anno, items_len):
    print(f"processing {sample_id} from total: {items_len}")

    color = cv2.imread(os.path.join(set_path, 'color', '%.5d.png' % sample_id))
    mask = imageio.v2.imread(os.path.join(set_path, 'mask', '%.5d.png' % sample_id))
    depth = imageio.v2.imread(os.path.join(set_path, 'depth', '%.5d.png' % sample_id))

    # process rgb coded depth into float: top bits are stored in red, bottom in green channel
    depth = depth_two_uint8_to_float(depth[:, :, 0], depth[:, :, 1])  # depth in meters from the camera
    depth[depth > 1.0] = 1.0
    depth = ((1.0-depth) * 255.0).astype("uint8")

    # 0-bg, 1 - body, 2-17 left hand, 18-33 right hand
    mask[mask == 1] = 0
    mask_l = mask
    mask_r = mask.copy()

    mask_l[mask_l > 17] = 0
    mask_l[mask_l - 1 < 17] = 255

    mask_r[mask_r >= 18] = 255
    mask_r[mask_r < 18] = 0

    cv2.imwrite(f"{set_path}/color_formatted/{class_name}_{subset}_{sample_id}.png", color)
    cv2.imwrite(f"{set_path}/depth_formatted/{class_name}_{subset}_{sample_id}.png", depth)
    cv2.imwrite(f"{set_path}/mask_formatted/{class_name}_{subset}_{sample_id}_i1.png", mask_l)
    cv2.imwrite(f"{set_path}/mask_formatted/{class_name}_{subset}_{sample_id}_i2.png", mask_r)

    # fig = plt.figure(1)
    # ax1 = fig.add_subplot(221)
    # ax2 = fig.add_subplot(222)
    # ax3 = fig.add_subplot(223)
    # ax4 = fig.add_subplot(224)
    # ax1.imshow(color)
    # ax2.imshow(depth)
    # ax3.imshow(mask_l)
    # ax4.imshow(mask_r)
    # plt.show() 

    pass

# load annotations of this set
with open(os.path.join(set_path, 'anno_%s.pickle' % subset), 'rb') as fi:
    anno_all = pickle.load(fi)
    
items = anno_all.items()
items_len = len(items)
Parallel(n_jobs=5)(delayed(process_img)(sample_id, anno, items_len) for sample_id, anno in items)
 
