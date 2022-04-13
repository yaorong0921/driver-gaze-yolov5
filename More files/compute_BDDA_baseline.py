import os
from PIL import Image
import numpy as np
import math
import argparse

import os
import numpy as np
import math

import torch
from torch.utils.data import Dataset
import cv2
from utils.utils import *
import torchvision
from PIL import Image

parser = argparse.ArgumentParser(description='Create average baseline for given gaze map images')
parser.add_argument('--gazemaps', metavar='DIR', help='path to gaze map images folder')

def main():
	transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize([36,64]),
            torchvision.transforms.ToTensor()])
	args = parser.parse_args()

	count = 0
	for root, dirs, files in os.walk(args.gazemaps):
		for item in files:
			gt = Image.open(os.path.join(args.gazemaps,item)).convert('L').crop((0,96,1024,672)) #left,top,right,bottom
			gt = np.array(transform(gt))
			gt = normalizeData(gt)
			if np.isnan(np.sum(gt)):
			    continue
			if count == 0:
			    sum = gt
			else:
			    sum += gt
			count += 1
			if count%500 == 0:
				print("Count: %d"%count)
	sum = normalizeData(sum)

	a_file = open("avgBaseline.txt", "w")
	for row in sum:
		np.savetxt(a_file, row)

	a_file.close()


def normalizeData(s_map):
	norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
	return norm_s_map

if __name__ == '__main__':
    main()
