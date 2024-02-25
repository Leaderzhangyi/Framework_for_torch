# -*- coding: utf-8 -*-
# @File    : base_dataset.py
# @Software: PyCharm
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

class TimeDataset(data.Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self,index):
        return self.x[index],self.y[index]