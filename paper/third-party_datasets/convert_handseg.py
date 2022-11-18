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

dataset_dir = "D:\datasets\Handseg"

class_name = "hand"

def process_img(sample_id, index, files_len):
    print(f"Processing {index} from {files_len}")
    # 1 - right, 2 - left
    mask = cv2.imread(os.path.join(dataset_dir, "masks", sample_id),0)
    mask_l = np.array(mask)
    mask_r = np.array(mask)
    mask_r[mask_r==1] = 255
    mask_l[mask_l==2] = 255

    depth = np.array(Image.open(os.path.join(dataset_dir, "images", class_name+"_"+sample_id)))
    depth = depth.astype(np.float32)
    depth /= 10000.0
    depth[depth > max_range] = max_range
    depth[depth < min_range] = min_range
    depth = (depth - min_range) / (max_range - min_range)
    depth = depth * 255
    depth[depth>0]= 255 - depth[depth>0]
    depth_img = Image.fromarray(np.uint8(depth),mode="L")

    sample_id = sample_id.replace("user-", "hand_")

    depth_img.save(os.path.join(dataset_dir, "depth_formatted", sample_id))
    cv2.imwrite(os.path.join(dataset_dir, "mask_formatted", sample_id.replace(".png", "_i1.png")),mask_l)
    cv2.imwrite(os.path.join(dataset_dir, "mask_formatted", sample_id.replace(".png", "_i2.png")),mask_r)

    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax1.imshow(mask_r)
    ax2.imshow(mask_l)
    ax3.imshow(depth)
    plt.show()

    
    
samples = os.listdir(os.path.join(dataset_dir, "images"))
samples_len = int(len(samples))
Parallel(n_jobs=5)(delayed(process_img)(sample_id, index, samples_len) for index, sample_id in enumerate(samples))
