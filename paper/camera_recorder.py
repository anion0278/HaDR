import time, os
import numpy as np
import cv2
import ws_specific_settings as wss
import model_utils as utils
import camera
from datetime import datetime as dt

import common_settings as s

recorded_dataset_path = s.path_to_datasets + "real_ws2"

if __name__ == "__main__":
    index = 1
    cam = camera.RgbdCamera((640,480), 30)
    os.makedirs(recorded_dataset_path, exist_ok=True)
    os.makedirs(recorded_dataset_path+"/depth", exist_ok=True)
    os.makedirs(recorded_dataset_path+"/color", exist_ok=True)
    os.makedirs(recorded_dataset_path+"/mask2", exist_ok=True)
    while(True):
        color, depth = camera.separate_color_from_depth(cam.get_rgbd_image())
        depth_3ch = np.stack((np.squeeze(depth),)*3, axis=-1)
        window_id = "recorder"
        cv2.imshow(window_id, np.hstack([color, depth_3ch]))
        cv2.setWindowTitle(window_id, "Dataset recorder. [Esc] to quit, [Space] to save image")
        keyPressed = cv2.waitKey(1)
        if keyPressed == 27:    # Esc key to stop
            print("Closing...")
            break
        if keyPressed == 32:    # Space key to save img
            print("Saving image...")
            timestamp = dt.now().strftime("%a_D%d_M%m_%Hh_%Mm_%Ss") 
            img_name = f"{index}_hand_date{timestamp}.png"
            index += 1
            cv2.imwrite(f"{recorded_dataset_path}/color/{img_name}", color)
            cv2.imwrite(f"{recorded_dataset_path}/depth/{img_name}", depth)
            print("Saved " + img_name)

    cam.close()