import os
archs = {
    "1": "solov2_light_448_r50_fpn",
    "2": "solov2_r101_fpn",
    "3": "mask_rcnn_r50_fpn",
    "4": "mask_rcnn_r101_fpn",
    }

data_configs = {
    "A": (1, True), # (channels, is_aug_enabled)
    "B": (3, True),
    "C": (4, True),
    "D": (1, False),
    "E": (3, False),
    "F": (4, False),
    }

for tag_name, config in data_configs.items():
    for tag_id, arch in archs.items():
        channels, is_aug_enabled = config
        command = f"python paper/trainer.py --tag {tag_id + tag_name} --arch {arch} --channels {channels} --aug {is_aug_enabled}"
        print(command)
        os.system(command)

