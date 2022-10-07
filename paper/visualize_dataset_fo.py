import common_settings as s
s.add_packages_paths()
from tkinter import TRUE
import fiftyone as fo

if __name__ == "__main__":
    data_path = "D:\datasets\RHD/depth"
    labels_path = "D:\datasets\RHD/instances_hands_100.json"
    dataset_type = fo.types.COCODetectionDataset
    dataset = fo.Dataset.from_dir(
        data_path=data_path,
        labels_path=labels_path,
        dataset_type=dataset_type,
        # max_samples=10, # actually works
    )

    session = fo.launch_app(dataset,desktop=TRUE)
    session.wait()