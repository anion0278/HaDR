import os
import sys
import webbrowser
import ws_specific_settings as wss

logs_path = wss.storage + ":/models"

def start_and_open():
    print("Starting tensorboard...")
    webbrowser.open_new("http://localhost:6006/#scalars")
    os.system('python -m tensorboard.main --logdir='+ logs_path)

if __name__ == "__main__":
    start_and_open()