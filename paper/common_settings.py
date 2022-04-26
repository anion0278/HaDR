path_to_configs = "./paper/tested_configs/%s.py"

sim_train_mean=[101.2224, 99.5584, 96.9216, 35.3792] 
sim_train_std=[67.3536, 67.2512, 68.8384, 45.978]

test_train_mean=[101.2224, 99.5584, 96.9216, 35.3792] 
test_train_std=[67.3536, 67.2512, 68.8384, 45.978]

sim_val_mean=[102.006, 98.065, 94.952, 37.56] 
sim_val_std=[39.619, 39.217, 39.583, 43.2]

def get_norm_params(input_channels):
    options = {
    1: dict(mean=[sim_train_mean[3]], std=[sim_train_std[3]], to_rgb=False),
    3: dict(mean=sim_train_mean[0:3], std=sim_train_std[0:3], to_rgb=True),
    4: dict(mean=sim_train_mean, std=sim_train_std, to_rgb=False),
    }
    return options[input_channels]

