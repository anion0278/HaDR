import os
import sys
import webbrowser
import common_settings as s
import model_utils as utils
import ws_specific_settings as wss

def start_and_open(start_dir):
    print("Starting tensorboard...")
    webbrowser.open_new("http://localhost:6006/#scalars")
    os.system('python -m tensorboard.main --max_reload_threads 4 --logdir='+ start_dir)

if __name__ == "__main__":
    model_dir = os.path.dirname(utils.ask_user_for_checkpoint(s.path_to_models))
    start_and_open(model_dir)