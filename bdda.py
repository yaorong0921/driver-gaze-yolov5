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
    def __init__(self, subset, file, feature_path, threshold, gazemap_path, lstm, seqlen):
        """
        Args:

        """
        self.subset = subset
        self.file = file
        self.feature_path = feature_path
        self.gazemap_path = gazemap_path
        self.threshold = threshold
        self.mean = torch.zeros(1024)
        self.std = torch.ones(1024)
        self.lstm = lstm
        self.seqlen = seqlen
        self._parse_list()
        self.transform = torchvision.transforms.Compose(
                [torchvision.transforms.Resize([36,64]),
                torchvision.transforms.ToTensor()])



    def _parse_list(self):

        self.img_list = []

        tmp = [x.strip().split(',') for x in open(self.file)]
        img_list = [VideoRecord(item) for item in tmp]

        if self.lstm:
            self.img_dict = {}

            clips = list(set([x.split('_')[0] for x in open(self.file)]))

            for clip in clips:
                self.img_dict[clip] = []

            for item in img_list:
                img_name = item.img_id.split('.')[0]
                feature_name = img_name + ".pt"
                clip = item.img_id.split('.')[0].split('_')[0]
                img_nr = item.img_id.split('.')[0].split('_')[1]
                grid = item.grids

                feature_path = os.path.join(self.feature_path,feature_name)

                if os.path.exists(feature_path) and not all(math.isnan(y) for y in grid):
                    self.img_list.append(item)
                    self.img_dict[clip].append(img_nr)
                else:
                    print('error loading feature:', feature_path)

            for key in self.img_dict:
                self.img_dict[key].sort()
            print('video number in %s: %d'%(self.subset,(len(self.img_list))))
        else:
            for item in img_list:
                img_name = item.img_id.split('.')[0]
                feature_name = img_name + ".pt"
                grid = item.grids

                feature_path = os.path.join(self.feature_path,feature_name)
                if os.path.exists(feature_path) and not all(math.isnan(y) for y in grid):
                    self.img_list.append(item)
                else:
                    print('error loading feature:', feature_path)


        print('video number in %s: %d'%(self.subset,(len(self.img_list))))


    def _normalizeData(self, data):
        return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        """
        """

        if self.lstm:
            record = self.img_list[index]
            img_name = record.img_id.split('.')[0]
            feature_name = img_name + ".pt"

            clip = record.img_id.split('.')[0].split('_')[0]
            img_nr = record.img_id.split('.')[0].split('_')[1]
            dict_idx = self.img_dict[clip].index(img_nr)

            feature_path = os.path.join(self.feature_path,feature_name)
            feature = torch.load(feature_path)

            # create list with previous features, last one is original
            feature_list = []
            first = dict_idx-(self.seqlen-1)
            duplicate = 0
            if first < 0:
                duplicate = abs(first) # if there are not enough previous features, we duplicate original to get seqlen
                first = 0
            for idx in range(first, dict_idx+1):
                feature_name2 = clip+'_'+self.img_dict[clip][idx]+ ".pt"
                feature_path2 = os.path.join(self.feature_path,feature_name2)
                feature2 = torch.load(feature_path2)
                feature_list.append(feature2)
            if duplicate:
                for i in range(duplicate):
                    feature_list.append(feature)
            feature = torch.stack(feature_list)
        else:
            record = self.img_list[index]
            img_name = record.img_id.split('.')[0]
            feature_name = img_name + ".pt"
            feature_path = os.path.join(self.feature_path,feature_name)
            feature = torch.load(feature_path)


        if self.subset == 'training':
            feature = feature + torch.randn(512,12,20)

        # set grid values <= 1/gridsize to 0, others to 1
        grid = np.array(record.grids)
        grid[grid>self.threshold] = 1.0
        grid[grid<=self.threshold] = 0.0
        grid = grid.astype(np.float32)


        if self.subset == 'test':
            name = record.img_id.split('_')
            gaze_file = name[0] + '_pure_hm_' + name[1]
            gaze_gt = Image.open(os.path.join(self.gazemap_path, gaze_file)).convert('L').crop((0,96,1024,672)) #left,top,right,bottom
            gaze_gt = self.transform(gaze_gt)
            gaze_gt = self._normalizeData(gaze_gt)

            return feature, grid, gaze_gt, img_name
        else:
            return feature, grid
