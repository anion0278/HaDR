import os
archs = {
    "1": "solov2_light_448_r50_fpn",
    "2": "solov2_r101_fpn",
    # "3": "mask_rcnn_r50_fpn",
    # "4": "mask_rcnn_r101_fpn",
    }

ds_3rd_p = "third-party/"

datasets = {"sim_train_320x256", ds_3rd_p+"egohands", ds_3rd_p+"densehands", ds_3rd_p+"rhd", ds_3rd_p+"handseg"}

data_configs = {
    # "A": (1, True), # (channels, is_aug_enabled)
    # "B": (3, True),
    # "C": (4, True),
    "X": (1, False),
    "Y": (3, False),
    "Z": (4, False),
    }

for tag_name, config in data_configs.items():
    for tag_id, arch in archs.items():
        for ds in datasets:
            channels, is_aug_enabled = config
            command = f"python paper/trainer.py --tag {tag_id + tag_name} --arch {arch} --channels {channels} --aug {is_aug_enabled} --ds {ds}"
            print(command)
            os.system(command)

