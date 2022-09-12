import os
import sys
import webbrowser
import common_settings as s
import model_utils as utils
import ws_specific_settings as wss
import tkinter as tk
import tkfilebrowser # pip tkfilebrowser, pywin32

def ask_user_for_log_dirs(default_checkpoint_path):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected_dirs = tkfilebrowser.askopendirnames(initialdir=default_checkpoint_path, title='Select a directory with logs')
    print(f"Selected {selected_dirs}")
    root.destroy()
    return selected_dirs

def start_and_open(log_dirs): 
    print("Starting tensorboard...")
    webbrowser.open_new("http://localhost:6006/#scalars")
    formatted_dirs = ""
    for dir in log_dirs: # todo somekind of LINQ
        name = os.path.basename(dir)
        formatted_dirs += f"{name}:{dir},"
    os.system('python -m tensorboard.main --max_reload_threads 4 --logdir_spec='+ formatted_dirs)

if __name__ == "__main__":
    log_dirs = ask_user_for_log_dirs(s.path_to_models) # we explicitly choose dirs, because it does not work well when root dir is chosen
    start_and_open(log_dirs)