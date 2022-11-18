import cv2, struct
import numpy as np
import itertools
from PIL import Image
import numpy as np
import os
from joblib import Parallel,delayed

# SEE https://stackoverflow.com/questions/67474451/why-does-cv2-imread-output-a-matrix-of-zeros-for-a-32-bit-image-even-when-using
# AND https://handtracker.mpi-inf.mpg.de/data/SynthHands/README.txt

min_range = 0.2
max_range = 1

dataset_dir = "D:\datasets\DenseHands"

class_name = "hand"

def process_img(sample_id, files_len, subset_path, subset_names_full):
    print(f"Processing {sample_id} from {files_len}")
    try:
        rgb_mask_l = cv2.imread(os.path.join(subset_path, "dense_corrs", str(sample_id)+"-colorAsign-L.png"), cv2.IMREAD_UNCHANGED)
        rgb_mask_r = cv2.imread(os.path.join(subset_path, "dense_corrs", str(sample_id)+"-colorAsign-R.png"), cv2.IMREAD_UNCHANGED)

        rgb_mask_l[rgb_mask_l < 255] = 0
        rgb_mask_l = cv2.cvtColor(rgb_mask_l, cv2.COLOR_BGR2GRAY)
        rgb_mask_l = 255 - rgb_mask_l
        rgb_mask_l[rgb_mask_l>0] = 255

        rgb_mask_r[rgb_mask_r < 255] = 0
        rgb_mask_r = cv2.cvtColor(rgb_mask_r, cv2.COLOR_BGR2GRAY)
        rgb_mask_r = 255 - rgb_mask_r
        rgb_mask_r[rgb_mask_r>0] = 255
        
        depth_pil = Image.open(os.path.join(subset_path, "depth", str(sample_id)+"-depth.png"))
        depth_pil_cv = np.array(depth_pil)
        depth_cv = depth_pil_cv.view(np.int32)
        depth_mm = depth_cv.astype(np.uint16)
        depth_meters = depth_mm / 1000.0 # mm to m
        depth_meters[depth_meters > max_range] = max_range
        depth_meters[depth_meters < min_range] = min_range
        depth_meters = (depth_meters - min_range) / (max_range - min_range)
        depth_byte = (255 - depth_meters * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(dataset_dir, "mask_formatted", f"{class_name}_{subset_names_full}_i1.png"), rgb_mask_l)
        cv2.imwrite(os.path.join(dataset_dir, "mask_formatted", f"{class_name}_{subset_names_full}_i2.png"), rgb_mask_r)
        cv2.imwrite(os.path.join(dataset_dir, "depth_formatted", f"{class_name}_{subset_names_full}.png"), depth_byte)

        # import matplotlib.pyplot as plt
        # fig = plt.figure(1)
        # ax1 = fig.add_subplot(221)
        # ax2 = fig.add_subplot(222)
        # ax1.imshow(depth_byte)
        # ax2.imshow(combined_mask)
        # plt.show()

    except:
        print("Some images dont have corresponding depth")
        return
        

for viewpoint_dir in ["Ego", "Bottom", "Front", "Top"]: 
    for subset in os.listdir(os.path.join(dataset_dir, viewpoint_dir)):
        subset_path = os.path.join(dataset_dir, viewpoint_dir, subset)
        files = os.listdir(os.path.join(subset_path, "dense_corrs"))
        samples_len = int(len(files) / 2)
        from joblib import Parallel,delayed
        Parallel(n_jobs=5)(delayed(process_img)(sample_id, samples_len, subset_path, f"{viewpoint_dir}_{subset}_{sample_id}") for sample_id in range(samples_len))
