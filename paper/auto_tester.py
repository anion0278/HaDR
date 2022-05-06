import os
import ws_specific_settings as wss
import re

MODEL_DIR = os.path.abspath(wss.storage+":/models/")

for filename in os.listdir(MODEL_DIR):
    if (re.search("^\d[A-Z]-",filename)):
        command = f"python paper/tester.py --checkpoint_path " + os.path.join(MODEL_DIR,filename)
        print(command)
        os.system(command)