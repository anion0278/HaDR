from tkinter import*
import os
import ws_specific_settings as wss

path_to_datasets= wss.storage + ":/datasets/"
path_to_models= wss.storage + ":/models/"
path_to_configs = "./paper/tested_configs/%s.py"
tested_checkpoint_file_name = "final.pth"
visualization_threshold = 0.5

# model_input_size = (256, 320) #TODO

# from custom train dataset
sim_train_mean=[100.618, 99.171, 96.664, 37.286] 
sim_train_std=[67.012, 66.945, 68.539, 47.192]

# from custom val dataset
sim_val_mean=[99.857, 96.167, 97.676, 55.093] 
sim_val_std=[66.85, 66.183, 67.875, 55.906]

# from coco
test_train_mean=[123.675, 116.28, 103.53, 35.3792] 
test_train_std=[58.395, 57.12, 57.375, 45.978]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        import argparse 
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_packages_paths():
    import os, sys
    path = os.path.abspath("./modified_packges")
    sys.path.insert(0,path)

def get_norm_params(input_channels, mode):
    if mode == "train":
        mean = sim_train_std
        std = sim_train_std
    if mode == "val":
        mean = sim_val_std
        std = sim_val_std
    if mode == "test":
        mean = test_train_mean
        std = test_train_std

    options = {
        1: dict(mean=[mean[3]], std=[std[3]], to_rgb=False),
        3: dict(mean=mean[0:3], std=std[0:3], to_rgb=True),
        4: dict(mean=mean, std=std, to_rgb=False),
    }
    return options[input_channels]

class DropDownMenuChoice():
    def __init__(self) -> None:
        self.chosen_value = None
        self.__opt_var = None

    def submitForm(self):    
        self.chosen_value = self.__opt_var.get()
        self.root.quit()
        self.root.destroy()

    def show_options(self, options, title, default_value=None):
        self.root = Tk()
        self.root.geometry('200x100')
        self.root.title(title)
        self.root.attributes('-toolwindow', True)
        self.root.eval('tk::PlaceWindow . center')

        self.__opt_var = StringVar(self.root)
        self.__opt_var.set(default_value) # default value
 
                
        for (text, value) in options.items():
            Radiobutton(self.root, text = text, variable = self.__opt_var,
                value = value).pack(side = TOP, ipady = 0)

        Button(self.root, text='Select', command=self.submitForm, width=20,bg='gray',fg='white').place(x=30,y=70)
        self.root.attributes("-topmost", True)

        self.root.mainloop()
        return self.chosen_value
