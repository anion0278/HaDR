path_to_configs = "./paper/tested_configs/%s.py"

train_mean=[100.836, 99.179, 96.541, 35.251] 
train_std=[38.393, 38.168, 39.837, 42.428]

val_mean=[102.006, 98.065, 94.952, 37.56] 
val_std=[39.619, 39.217, 39.583, 43.2]

def get_norm_params(input_channels):
    options = {
    1: dict(mean=[train_mean[3]], std=[train_std[3]], to_rgb=False),
    3: dict(mean=train_mean[0:3], std=train_std[0:3], to_rgb=True),
    4: dict(mean=train_mean, std=train_std, to_rgb=False),
    }
    return options[input_channels]

