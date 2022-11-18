import os
archs = {
    "1": "solov2_light_448_r50_fpn",
    "2": "solov2_r101_fpn",
    # "3": "mask_rcnn_r50_fpn",
    # "4": "mask_rcnn_r101_fpn",
    }

ds_3rd_p = "third-party/"
our_ds = "sim_train_320x256"

data_configs = {
    "E": (3, False, ds_3rd_p+"egohands"),

    "H": (1, False, ds_3rd_p+"handseg"),

    "D": (1, False, ds_3rd_p+"densehands"),

    "O": (1, False, ds_3rd_p+"obman"),
    "O": (3, False, ds_3rd_p+"obman"),
    "O": (4, False, ds_3rd_p+"obman"),

    "R": (1, False, ds_3rd_p+"RHD"),
    "R": (3, False, ds_3rd_p+"RHD"),
    "R": (4, False, ds_3rd_p+"RHD"),

    "X": (1, False, our_ds),
    "Y": (3, False, our_ds),
    "Z": (4, False, our_ds),
    }

for tag_name, config in data_configs.items():
    for tag_id, arch in archs.items():
            channels, is_aug_enabled, ds = config
            command = f"python paper/trainer.py --tag {tag_id + tag_name} --arch {arch} --channels {channels} --aug {is_aug_enabled} --ds {ds}"
            print(command)
            os.system(command)


