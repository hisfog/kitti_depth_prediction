import os
import random
import numpy as np
import copy
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms

def load_image(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class basicDataset(data.Dataset):
    def __init__(self,
                 data_path,
                 #filenames,
                 height,
                 width,
                 num_scales,
                 is_train=False,
                 img_ext='.png'):
    super(basicDataset, self).__init__()
    self.data_path = data_path
    #self.filenames = filenames
    self.height = height
    self.width = width
    self.num_scales = num_scales
    self.interp = Image.ANTIALIAS
    self.is_train = is_train
    self.img_ext = img_ext
    self.loader = load_image
    self.to_tensor = transforms.ToTensor()
    self.resize =  {}
    for i in range(self.num_scales):
        s = 2 ** i
        self.resize[i] = transforms.Resize((self.height // s, self.width // s), interpolation=self.interp)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}
        inputs['src'] = self.loader('1.jpg')
        return inputs

