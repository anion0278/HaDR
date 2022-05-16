import common_settings as s
s.add_packages_paths()
from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import ws_specific_settings as wss


img_color_path = wss.storage +  r':\datasets\sim_train_320x256\color\0000001_gest1_hand1_date13-04-2022_19#00#59.png'

img_color = np.asarray(Image.open(img_color_path))
img_depth = np.asarray(Image.open(img_color_path.replace("color", "depth")))

for corruption in get_corruption_names('all'):
    for severity in range(1,6): #[1..5]
        tic = time.time()
        corrupted = corrupt(img_color, corruption_name=corruption, severity=severity)
        print(f"{corruption} - severity: {severity}", time.time() - tic)
        plt.imshow(corrupted)
        plt.show()


# for corruption in get_corruption_names('all'):
#     tic = time.time()
#     corrupted = corrupt(img_color, corruption_name=corruption, severity=severity)
#     print(f"{corruption} - severity: {severity}", time.time() - tic)
#     #plt.imshow(corrupted)
#     #plt.show()