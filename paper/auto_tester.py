import os
import ws_specific_settings as wss
import common_settings as s
import re

ALL_MODELS_DIR = s.path_to_models
MODEL_NAME = "final.pth"

for model_dir in os.listdir(ALL_MODELS_DIR):
    if (re.search("^\d[A-Z]-",model_dir)):
        dirname = os.path.join(ALL_MODELS_DIR, model_dir)
        if (os.path.exists(os.path.join(dirname,MODEL_NAME))):
            command = f"python paper/tester.py --checkpoint_path {dirname} --eval segm --out {dirname}\out.pkl"
            print(command)
            os.system(command)