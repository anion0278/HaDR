import common_settings as s
s.add_packages_paths()
from imagecorruptions import corrupt, get_corruption_names
import numpy as np
import matplotlib.pyplot as plt
import time
import ws_specific_settings as wss
import cv2

img_color_path = wss.storage +  r':\datasets\sim_train_320x256\color\0000001_gest1_hand3_date13-04-2022_22#06#10.png'

img_color = np.asarray(cv2.imread(img_color_path))
img_depth = np.asarray(cv2.imread(img_color_path.replace("color", "depth"), cv2.IMREAD_GRAYSCALE))

corruptions_with_severities =[
    ("motion_blur",2),
    ("brightness", 5),
    ("saturate", 5),
    # ("elastic_transform", 1),
    # ("contrast", 1),
    # ("fog", 1),
    # ("defocus_blur", 1),
]

added_corruptions = "(orig-above, aug-below)"

origs = np.hstack([img_color, np.stack((np.squeeze(img_depth),)*3, axis=-1)])

for corruption, severity in corruptions_with_severities:
    img_color = corrupt(img_color, corruption_name=corruption, severity=severity)
    img_depth = corrupt(img_depth, corruption_name=corruption, severity=severity)
    window_id = "win"
    augs = np.hstack([img_color, np.stack((np.squeeze(img_depth),)*3, axis=-1)])
    comparison_img = np.vstack([origs, augs])
    cv2.imshow(window_id, comparison_img)
    added_corruptions += f"{corruption}{severity}; "
    cv2.setWindowTitle(window_id, added_corruptions)
    cv2.waitKey(-1)


# for corruption in get_corruption_names('all'):
#     for severity in range(1,6): #[1..5]
#         tic = time.time()
#         corrupted = corrupt(img_color, corruption_name=corruption, severity=severity)
#         print(f"{corruption} - severity: {severity}", time.time() - tic)
#         plt.imshow(corrupted)
#         plt.show()