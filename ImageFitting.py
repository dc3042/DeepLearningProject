import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.io import read_image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

import time



def get_mgrid(sidelen=256, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class ImageFitting(Dataset):

    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.coords = get_mgrid()

        self.len = len(os.listdir(self.img_dir))
        print(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, 'h5_f_{iter:010d}.{ext}'.format(iter=idx, ext='png'))
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)


        data_item = {'image': image,
                     'coords': self.coords,
                     'step': idx}


        return image

currentdir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(currentdir,'data')

cameraman = ImageFitting(data_dir, transform=Compose([
        Resize(256),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ]))

dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)
for i in range(len(dataloader)):
    d = next(iter(dataloader))
    print(d)