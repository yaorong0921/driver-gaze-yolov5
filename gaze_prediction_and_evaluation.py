"""
The code for computing the saliency metrics is adapted from
https://github.com/tarunsharma1/saliency_metrics/blob/master/salience_metrics.py
"""

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
from bdda import BDDA


from sklearn.metrics import f1_score,precision_score,recall_score, roc_curve, roc_auc_score



parser = argparse.ArgumentParser(description='Feature Training and Test')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--best', default='', type=str, metavar='PATH', help='path to best checkpoint (default: none)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--no_train', action='store_true', default=False)
parser.add_argument('--gridheight', default=16, type=int, metavar='N',
                    help='number of rows in grid')
parser.add_argument('--gridwidth', default=16, type=int, metavar='N',
                    help='number of columns in grid ')
parser.add_argument('--gazemaps', metavar='DIR', help='path to gaze map images folder')
parser.add_argument('--traingrid', default='', type=str, metavar='PATH', help='path to txt with grid entries for training images')
parser.add_argument('--valgrid', default='', type=str, metavar='PATH', help='path to txt with grid entries for validation images')
parser.add_argument('--testgrid', default='', type=str, metavar='PATH', help='path to txt with grid entries for test images')
parser.add_argument('--yolo5bb', metavar='DIR', help='path to folder of yolo5 bounding box txt files')
parser.add_argument('--visualizations', metavar='DIR', help='path to folder for visalization of predicted gaze maps and target')
parser.add_argument('--threshhold', default=0.5, type=float, metavar='N', help='threshold for object-level evaluation')
parser.add_argument('--lstm', default=False, action='store_true', help='use lstm module')
parser.add_argument('--convlstm', default=False, action='store_true', help='use convlstm module')
parser.add_argument('--sequence', default=6, type=int, metavar='N', help='sequence length for lstm module')


def main():
    args = parser.parse_args()

    dim = args.gridwidth*args.gridheight
    th = 1/dim

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    model = network.Net(args.gridheight, args.gridwidth)

    if args.lstm:
        model = network.LstmNet(args.gridheight, args.gridwidth)

    if args.convlstm:
        model = network.ConvLSTMNet(args.gridheight, args.gridwidth, args.sequence)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                betas=(0.9, 0.999), eps=1e-08,
                                weight_decay=args.weight_decay)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    if not args.no_train:
        traindir = os.path.join(args.data, 'training')
        valdir = os.path.join(args.data, 'validation')

        train_dataset = BDDA("training", args.traingrid, traindir, th, args.gazemaps,  (args.lstm or args.convlstm), args.sequence)
        val_dataset = BDDA("validation", args.valgrid, valdir, th, args.gazemaps,  (args.lstm or args.convlstm), args.sequence)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle= True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)


    testdir = os.path.join(args.data,'test')
    test_dataset = BDDA("test", args.testgrid, testdir, th, args.gazemaps, (args.lstm or args.convlstm), args.sequence)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    best_loss = 1000000

    if not args.no_train:

        for epoch in range(args.start_epoch, args.epochs):

            adjust_learning_rate(optimizer, epoch, args)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args)

            # evaluate on validation set
            loss1 = validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = loss1 < best_loss
            best_loss = min(loss1, best_loss)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, is_best, args.best)

    if args.best:
        if os.path.isfile(args.best):
            print("=> loading checkpoint '{}'".format(args.best))
            checkpoint = torch.load(args.best)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.best, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    test(test_loader, model, criterion, args)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, filename)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)

        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)

            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      .format(
                       i, len(val_loader), batch_time=batch_time, loss=losses))

    return loss


def test(test_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    kld_losses = AverageMeter()
    cc_losses = AverageMeter()

    model.eval()

    tp = 0
    fp = 0
    fn = 0
    all_count = 0

    hm_max_values = []
    gt = []

    i = 0

    heightfactor = 576//args.gridheight
    widthfactor = 1024//args.gridwidth

    smoothing = GaussianSmoothing(1, 5, 1).cuda(args.gpu)
    with torch.no_grad():
        end = time.time()
        for i, (input, target, gaze_gt, img_names) in enumerate(test_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
                gaze_gt = gaze_gt.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)

            loss = criterion(output, target)

            output = torch.sigmoid(output)

            heatmap = grid2heatmap(output,[heightfactor,widthfactor],[args.gridheight,args.gridwidth],args)
            heatmap = F.interpolate(heatmap, size=[36, 64], mode='bilinear', align_corners=False)
            heatmap = smoothing(heatmap)
            heatmap = F.pad(heatmap, (2, 2, 2, 2), mode='constant')
            heatmap = heatmap.view(heatmap.size(0),-1)
            heatmap = F.softmax(heatmap,dim=1)

            # normalize
            heatmap -= heatmap.min(1, keepdim=True)[0]
            heatmap /= heatmap.max(1, keepdim=True)[0]

            heatmap = heatmap.view(-1,1,36,64)

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

            losses.update(loss.item(), input.size(0))
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

def grid2heatmap(grid, size, num_grid, args):
    """
    Rearrange and expand gridvector of size (gridheight*gridwidth) to size (576 x 1024) by duplicating values

    :param grid: output vector
    :param size: (H,W) of one expanded grid cell
    :param num_grids: (H,W) = grid dimension
    :param args: parser arguments
    :return: 2D grid of size (576 x 1024)
    """
    new_heatmap = torch.zeros(grid.size(0),size[0]*num_grid[0],size[1]*num_grid[1])
    for i, item in enumerate(grid):
        idx = torch.nonzero(item)
        if idx.nelement() == 0:
            print('Empty')
            continue
        for x in idx:
            test = new_heatmap[i,x//num_grid[1]*size[0]:(x//num_grid[1]+1)*size[0],x%num_grid[1]*size[1]:(x%num_grid[1]+1)*size[1]]
            new_heatmap[i,x//num_grid[1]*size[0]:(x//num_grid[1]+1)*size[0],x%num_grid[1]*size[1]:(x%num_grid[1]+1)*size[1]] = item[x]
    output = new_heatmap.unsqueeze(1).cuda(args.gpu)

    return output

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

def kullback_leibler_divergence(y_true, y_pred, eps=1e-7):
    """
    Kullback-Leiber divergence (sec 4.2.3 of [1]). Assumes shape (b, 1, h, w) for all tensors.

    :param y_true: groundtruth.
    :param y_pred: prediction.
    :param eps: regularization epsilon.
    :return: loss value (one symbolic value per batch element).
    """
    P = y_pred
    P = P / (eps + torch.sum(P, dim=[1, 2, 3], keepdim=True))
    Q = y_true
    Q = Q / (eps + torch.sum(Q, dim=[1, 2, 3], keepdim=True))

    kld = torch.sum(Q * torch.log(eps + Q/(eps + P)), dim=[1, 2, 3])

    return kld

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


if __name__ == '__main__':
    main()
