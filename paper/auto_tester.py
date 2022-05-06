import os
import ws_specific_settings as wss
import re

MODELS_DIR = os.path.abspath(wss.storage+":/models/")
MODEL_NAME = "final.pth"

for model_dir in os.listdir(MODELS_DIR):
    if (re.search("^\d[A-Z]-",model_dir)):
        dirname = os.path.join(MODELS_DIR,model_dir)
        if (os.path.exists(os.path.join(dirname,MODEL_NAME))):
            command = f"python paper/tester.py --checkpoint_path " + dirname
            print(command)
            os.system(command)