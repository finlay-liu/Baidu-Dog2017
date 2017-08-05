from __future__ import print_function

import os, gc, re, sys, glob, cv2, h5py, codecs
import numpy as np
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# Image batch generator
#   img, image path
#   label, image label
class dogloader(Dataset):
    def __init__(self, img, label, transform = None):
        self.img = img; self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        idx_img = Image.open(self.img[idx]).convert('RGB')
        if self.transform is not None:
            idx_img = self.transform(idx_img)
        label = torch.from_numpy(np.array([self.label[idx]]))
        return idx_img, label

# Normal batch generator
#   x and y are list or array
class Arrayloader(Dataset):
    def __init__(self, x, y):
        self.x = x; self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        idx_x = torch.from_numpy(np.array([self.x[idx]]))
        idx_y = torch.from_numpy(np.array([self.y[idx]]))
        return idx_x, idx_y

# Plot logging to file and image
def loging(str, log_file, plot = False):
    with codecs.open(log_file, 'a') as f:
        f.write(str + '\n')

    with codecs.open(log_file, 'r') as f:
        lines = f.readlines()

    if plot:
        train_loss = [float(re.split('/| \[| |\n', x)[3]) for x in lines[:-1]]
        val_loss = [float(re.split('/| \[| |\n', x)[4]) for x in lines[:-1]]

        train_acc = [1- float(re.split('/| \[| |\n', x)[6]) for x in lines[:-1]]
        val_acc = [1 - float(re.split('/| \[| |\n', x)[7]) for x in lines[:-1]]

        plt.figure(figsize = (10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_loss) + 1), train_loss)
        plt.plot(range(1, len(train_loss) + 1), val_loss)
        plt.legend(['train loss', 'val loss'], loc = 1)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(train_loss) + 1), train_acc)
        plt.plot(range(1, len(train_loss) + 1), val_acc)
        plt.legend(['train acc', 'val acc'], loc = 1)

        plt.savefig(log_file + '.png', dpi = 200, bbox_inches = 'tight')
        plt.close('all')
