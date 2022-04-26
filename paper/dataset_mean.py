from torch.utils.data import DataLoader
import torch
import numpy as np
import os,sys
import mmcv
from mmcv import Config
import torchvision
from torchvision import transforms
from mmdet.datasets import build_dataset
path = os.path.abspath("./modified_packges")
sys.path.insert(0,path)

TRANSFORM_IMG = transforms.Compose([transforms.ToTensor()])
TRAIN_DATA_PATH = os.path.abspath('C:/datasets/sim_val_320x256/depth/') 
if __name__ == "__main__":

    dataset = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)


    loader = DataLoader(dataset=dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=0)
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for data in loader:
        data = torch.Tensor(data[0])
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    print(fst_moment, torch.sqrt(snd_moment - fst_moment ** 2))