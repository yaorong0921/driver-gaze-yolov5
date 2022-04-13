import os
import argparse
import time
import shutil
import math

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import torchvision

import numbers

import network
from bdda_otherModels import BDDA

import numpy as np
from PIL import Image

from sklearn.metrics import f1_score,precision_score,recall_score, roc_curve, roc_auc_score



parser = argparse.ArgumentParser(description='Evalutaion of given Predictions')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--gazemaps', metavar='DIR', help='path to gaze map images folder')
parser.add_argument('--yolo5bb', metavar='DIR', help='path to folder of yolo5 bounding box txt files')
parser.add_argument('--predictions', metavar='DIR', help='path to predicted gaze maps folder')
parser.add_argument('--visualizations', metavar='DIR', help='path to folder for visalization of predicted gaze maps and target')
parser.add_argument('--threshhold', default=0.5, type=float, metavar='N', help='threshold for object-level evaluation')

def main():
    args = parser.parse_args()

    dim = 256
    th = 1/dim

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    testdir = os.path.join(args.data,'test')
    test_dataset = BDDA("test", testdir, th, args.gazemaps, (args.lstm or args.convlstm), args.sequence)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test(test_loader, args)

def test(test_loader, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    kld_losses = AverageMeter()
    cc_losses = AverageMeter()

    tp = 0
    fp = 0
    fn = 0
    all_count = 0

    hm_max_values = []
    gt = []

    i = 0

    transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize([36,64]),
            torchvision.transforms.ToTensor()])

    with torch.no_grad():
        end = time.time()
        for i, (gaze_gt, img_names) in enumerate(test_loader):
            if args.gpu is not None:
                gaze_gt = gaze_gt.cuda(args.gpu, non_blocking=True)

            first = True

            for img in img_names:

                heatfile = img+ '.jpg'

                heatmap = Image.open(os.path.join(args.predictions ,heatfile))#.convert('L')#.crop((0,96,1024,672)) #left,top,right,bottom
                heatmap = transform(heatmap)
                heatmap = normalizeData(heatmap)

                if first:
                    heatmap_batch = heatmap[None]
                    first = False
                else:
                    heatmap_batch = torch.cat((heatmap_batch, heatmap[None]), 0)

            heatmap = heatmap_batch

            for j in range(heatmap.size(0)):
                img_name = img_names[j]
                heatmap_img = heatmap[j] # predicted gaze map
                gt_img = gaze_gt[j] # original gaze map

                ##### compute object-level metrics

                filename  = os.path.join(args.yolo5bb, img_name+".txt")

                if os.path.exists(filename):
                    with open(filename) as f:

                        for linestring in f:
                            all_count += 1

                            line = linestring.split()

                            width = float(line[3])
                            height = float(line[4])
                            x_center = float(line[1])
                            y_center = float(line[2])

                            x_min, x_max, y_min, y_max = bb_mapping(x_center, y_center, width, height)

                            # find maximum pixel value within object bounding box
                            gt_obj = gt_img[0, y_min:y_max+1, x_min:x_max+1]
                            gt_obj_max = torch.max(gt_obj)
                            heatmap_obj = heatmap_img[0, y_min:y_max+1, x_min:x_max+1]
                            heatmap_obj_max = torch.max(heatmap_obj)
                            print(heatmap_obj_max)

                            # object is recognized if maximum pixel value is higher than th
                            gt_obj_recogn = gt_obj_max > 0.15
                            hm_obj_recogn = heatmap_obj_max > args.threshhold

                            hm_max_values.append(heatmap_obj_max)

                            if gt_obj_recogn:
                                gt.append(1)
                            else:
                                gt.append(0)

                            if (hm_obj_recogn and gt_obj_recogn):
                                tp +=1
                            elif (hm_obj_recogn and not gt_obj_recogn):
                                fp += 1
                            elif (not hm_obj_recogn and gt_obj_recogn):
                                fn += 1

                        visualization(heatmap_img.cpu(), gt_img.cpu(), args.visualizations, img_name)


            kld = kl(heatmap, gaze_gt)
            c = cc(heatmap,gaze_gt)

            kld_losses.update(kld, input.size(0))
            cc_losses.update(c, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'KL {kl.val:.4f} ({kl.avg:.4f})\t'
                      'CC {cc.val:.4f} ({cc.avg:.4f})\t'
                      .format(
                       i, len(test_loader), batch_time=batch_time, loss=losses, kl=kld_losses, cc=cc_losses))

        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'KL {kl.val:.4f} ({kl.avg:.4f})\t'
              'CC {cc.val:.4f} ({cc.avg:.4f})\t'
              .format(
               i, len(test_loader), batch_time=batch_time, loss=losses, kl=kld_losses, cc=cc_losses))

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        tn = all_count-tp-fp-fn
        acc = (tp+tn)/all_count
        f1 = 2*precision*recall/(precision+recall)
        print('Object-level results:')
        print('tp:', tp, 'fp:', fp, 'tn:', tn, 'fn:', fn, 'sum:', all_count)
        print('prec:', precision, 'recall:', recall, 'f1', f1, 'acc', acc)
        print('AUC:', roc_auc_score(gt, hm_max_values))

def bb_mapping(x_center_rel, y_center_rel, width_rel, height_rel, img_width = 64, img_height = 36):
    """
    Compute absolute bounding boxes values for given image size and given relative parameters

    :param x_center_rel: relative x value of bb center
    :param y_center_rel: relative y value of bb center
    :param width_rel: relative width
    :param height_rel: relative height
    :return: absolute values of bb borders
    """
    width_abs = width_rel*img_width
    height_abs = height_rel*img_height
    x_center_abs = x_center_rel*img_width
    y_center_abs = y_center_rel*img_height
    x_min = int(math.floor(x_center_abs - 0.5 * width_abs))
    x_max = int(math.floor(x_center_abs + 0.5 * width_abs))
    y_min = int(math.floor(y_center_abs - 0.5 * height_abs))
    y_max = int(math.floor(y_center_abs + 0.5 * height_abs))
    bb = [x if x>=0 else 0 for x in [x_min, x_max, y_min, y_max]]
    return bb


def cc(s_map_all,gt_all):
	eps = 1e-07
	bs = s_map_all.size()[0]
	r = 0
	for i in range(0, bs):
		s_map = s_map_all[i,:,:,:].squeeze()
		gt = gt_all[i,:,:,:].squeeze()
		s_map_norm = (s_map - torch.mean(s_map))/(eps + torch.std(s_map))
		gt_norm = (gt - torch.mean(gt))/(eps + torch.std(gt))
		a = s_map_norm.cpu()
		b = gt_norm.cpu()
		r += torch.sum(a*b) / (torch.sqrt(torch.sum(a*a) * torch.sum(b*b))+eps)
	return r/bs

def kl(s_map_all, gt_all):
	dims = len(s_map_all.size())
	bs = s_map_all.size()[0]
	eps = torch.tensor(1e-07)
	kl = 0

	if dims > 3:
		for i in range(0, bs):
			s_map = s_map_all[i,:,:,:].squeeze()
			gt = gt_all[i,:,:,:].squeeze()
			s_map = s_map/(torch.sum(s_map)*1.0 + eps)
			gt = gt/(torch.sum(gt)*1.0 + eps)
			gt = gt.to('cpu')
			s_map = s_map.to('cpu')
			kl += torch.sum(gt * torch.log(eps + gt/(s_map + eps)))
		return kl/bs


def normalizeData(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

def visualization(heatmap, gt, path, nr):
    heatmap = torchvision.transforms.functional.to_pil_image(heatmap)
    gt = torchvision.transforms.functional.to_pil_image(gt)

    heatmap.save(os.path.join(path, '%s_pred.png'%nr))
    gt.save(os.path.join(path, '%s_gt.png'%nr))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
