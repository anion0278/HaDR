path_to_configs = "./paper/tested_configs/%s.py"

# from our train dataset
sim_train_mean=[101.2224, 99.5584, 96.9216, 35.3792] 
sim_train_std=[67.3536, 67.2512, 68.8384, 45.978]

# from our val dataset
sim_val_mean=[102.006, 98.065, 94.952, 37.56] 
sim_val_std=[39.619, 39.217, 39.583, 43.2]

# from coco
test_train_mean=[123.675, 116.28, 103.53, 35.3792] 
test_train_std=[58.395, 57.12, 57.375, 45.978]


def get_norm_params(input_channels, mode):
    if mode == "train":
        mean = sim_train_std
        std = sim_train_std
    if mode == "test":
        mean = test_train_mean
        std = test_train_std

    options = {
        1: dict(mean=[mean[3]], std=[std[3]], to_rgb=False),
        3: dict(mean=mean[0:3], std=std[0:3], to_rgb=True),
        4: dict(mean=mean, std=std, to_rgb=False),
    }
    return options[input_channels]

