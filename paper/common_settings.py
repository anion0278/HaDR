from tkinter import*
import os
import ws_specific_settings as wss

path_to_datasets= wss.storage + ":/datasets/"
path_to_models= wss.storage + ":/models/"
path_to_configs_formatted = "./paper/tested_configs/%s.py"
tested_checkpoint_file_name = "final.pth"
visualization_threshold = 0.2
batch_size = 8

# model_input_size = (256, 320) #TODO

# from custom train dataset - RGB-D
sim_train_mean=[96.924, 90.654, 88.884, 40.039] 
sim_train_std_rgbd=[69.349, 66.891, 66.718, 48.845]

# from custom val dataset - RGB-D
sim_val_mean=[102.951, 96.150, 98.884, 53.006] 
sim_val_std_rgbd=[68.324, 66.890, 68.474, 49.755]

# from COCO - RGB-D, represents real-life color distribution. Depth channel is obtained from custom real-cam dataset
test_train_mean_rgbd=[123.675, 116.28, 103.53, 35.3792] 
test_train_std_rgbd=[58.395, 57.12, 57.375, 45.978]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        import argparse 
        raise argparse.ArgumentTypeError("Boolean value expected.")


def add_packages_paths():
    import os, sys
    path = os.path.abspath("./modified_packges")
    sys.path.insert(0,path)

def get_norm_params(input_channels, mode):
    if mode == "train":
        mean_rgbd = sim_train_std_rgbd
        std_rgbd = sim_train_std_rgbd
    if mode == "val":
        mean_rgbd = sim_val_std_rgbd
        std_rgbd = sim_val_std_rgbd
    if mode == "test":
        mean_rgbd = test_train_mean_rgbd
        std_rgbd = test_train_std_rgbd

    options = {
        1: dict(mean=[mean_rgbd[3]], std=[std_rgbd[3]], to_rgb=False),
        3: dict(mean=mean_rgbd[0:3], std=std_rgbd[0:3], to_rgb=True),
        4: dict(mean=mean_rgbd, std=std_rgbd, to_rgb=False),
    }
    return options[input_channels]

class DropDownMenuChoice():
    def __init__(self) -> None:
        self.chosen_value = None
        self.__opt_var = None

    def submitForm(self, e = None):    
        self.chosen_value = self.__opt_var.get()
        self.root.quit()
        self.root.destroy()

    def show_options(self, options, title, default_value=None):
        self.root = Tk()
        self.root.geometry("200x100")
        self.root.title(title)
        self.root.attributes("-toolwindow", True)
        self.root.eval("tk::PlaceWindow . center")

        self.__opt_var = StringVar(self.root)
        self.__opt_var.set(default_value) # default value

        for (text, value) in options.items():
            Radiobutton(self.root, text = text, variable = self.__opt_var,
                value = value).pack(side = TOP, ipady = 0)
        
        Button(self.root, text="Select", command=self.submitForm, width=20,bg="gray",fg="white").place(x=30,y=70)
        self.root.bind('<Return>', self.submitForm)
        self.root.attributes("-topmost", True)

        self.root.mainloop()
        return self.chosen_value
