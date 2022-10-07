#!/usr/bin/env python3
import common_settings as s
s.add_packages_paths()
import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

# ROOT_DIR = os.path.abspath(r"D:\datasets\RHD")
IMAGE_DIR = os.path.join(ROOT_DIR, "depth")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "mask")

INFO = {
    "description": "Hands Dataset",
    "url": "https://github.com/anion0278/hand_tracker",
    "version": "1.0.0",
    "year": 2022,
    "contributor": "VSB-TUO Department of Robotics",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "CC BY 4.0",
        "url": "https://creativecommons.org/licenses/by/4.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'hand',
        'supercategory': 'arm-regions',
    },
]

def filter_imgs(root, files):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files

# def filter_for_annotations(root, files, image_filename):
#     file_types = ['*.png']
#     file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
#     basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
#     file_name_prefix = basename_no_extension + '.*'
#     files = [os.path.join(root, f) for f in files]
#     files = [f for f in files if re.match(file_types, f)]
#     files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
#     return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    files = [basename_no_extension + "_i1.png",basename_no_extension + "_i2.png"]
    files = [os.path.join(root, f) for f in files]
    return files

def main(dataset_path):
    ROOT_DIR = dataset_path
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1
    
    files = list(os.walk(IMAGE_DIR))
    for root, _, files in files:
        image_files = filter_imgs(root, files)

        # go through each image
        for i, image_filename in enumerate(image_files):
            print("Processing %s from %s" % (i+1, len(image_files)))
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                print("Found %s related binary annotations" % len(annotation_files))
                # go through each associated annotation
                for annotation_filename in annotation_files:
                    
                    print("Processing annotation: %s " % annotation_filename)
                    class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                        .convert('1')).astype(np.uint8)
                    
                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id += 1
            if image_id%10000==0:
                with open('{}/instances_hands_{}.json'.format(ROOT_DIR,image_id), 'w') as output_json_file:
                    json.dump(coco_output, output_json_file)

    with open('{}/instances_hands_full.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    for ds in ["Handseg", "ObMan", "DenseHands"]:
        main("C:/datasets/third-party/"+ds)
