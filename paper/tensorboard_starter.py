import os
import sys
import webbrowser
import common_settings as s
import model_utils as utils
import ws_specific_settings as wss
import tkinter as tk
from tkinter import filedialog

def ask_user_for_log_dir(default_checkpoint_path):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected_dir = filedialog.askdirectory(initialdir=default_checkpoint_path, title='Select a directory with logs')
    print(f"Selected {selected_dir}")
    root.destroy()
    return selected_dir

def start_and_open(start_dir):
    print("Starting tensorboard...")
    webbrowser.open_new("http://localhost:6006/#scalars")
    os.system('python -m tensorboard.main --max_reload_threads 4 --logdir='+ start_dir)

if __name__ == "__main__":
    model_dir = ask_user_for_log_dir(s.path_to_models)
    start_and_open(model_dir)