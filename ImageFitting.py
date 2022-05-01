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
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.transform = Compose([
            Resize(256),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])
        self.coords = get_mgrid()
        self.len = len(os.listdir(self.img_dir))
        print(self.len)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, 'h5_f_{iter:010d}.{ext}'.format(iter=idx, ext='png'))
        
        image = Image.open(img_path)
        image = self.transform(image)
        data_item = {'image': image,
                     'coords': self.coords,
                     'step': idx}
        return data_item