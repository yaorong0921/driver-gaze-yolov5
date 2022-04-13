import os
import numpy as np
import math

import torch
from torch.utils.data import Dataset
import cv2
from utils.utils import *
import torchvision
from PIL import Image

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def img_id(self):
        return (self._data[0]) # image index starts with 1

    @ property
    def grids(self):
        grid=[]
        for item in self._data[1:]:
            grid.append(float(item))
        return grid


class BDDA(Dataset):
    """
    BDDA feature class.
    """
    def __init__(self, file, threshold, gazemap_path):
        """
        Args:

        """
        self.file = file
        self.gazemap_path = gazemap_path
        self.threshold = threshold
        self.mean = torch.zeros(1024)
        self.std = torch.ones(1024)
        self._parse_list()
        self.transform = torchvision.transforms.Compose(
                [torchvision.transforms.Resize([36,64]),
                torchvision.transforms.ToTensor()])



    def _parse_list(self):

        self.img_list = []

        tmp = [x.strip().split(',') for x in open(self.file)]
        img_list = [VideoRecord(item) for item in tmp]
        self.img_list = img_list

    def _normalizeData(self, data):
        return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        """
        """
        record = self.img_list[index]
        img_name = record.img_id.split('.')[0]
        name = record.img_id.split('_')
        gaze_file = name[0] + '_pure_hm_' + name[1]
        gaze_gt = Image.open(os.path.join(self.gazemap_path, gaze_file)).convert('L').crop((0,96,1024,672)) #left,top,right,bottom
        gaze_gt = self.transform(gaze_gt)
        gaze_gt = self._normalizeData(gaze_gt)

        return gaze_gt, img_name
