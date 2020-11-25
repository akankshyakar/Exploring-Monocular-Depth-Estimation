import os
import time
import csv
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim

from dataloader.nyu_dataloader import NYUDataset
from dataloader.dataloader import MyDataloader
import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pdb
st = pdb.set_trace
cmap = plt.cm.viridis
cudnn.benchmark = True


@torch.no_grad()
def compute_depth_metrics(gt, pred, crop=False):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.shape[0]

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]


def create_data_loaders(args):
    print("=> creating data loaders ...")
    traindir = os.path.join(args.path, args.data, 'train')
    valdir = os.path.join(args.path, args.data, 'val')
    train_loader = None
    val_loader = None
    sparsifier = None #class for generating random sparse depth input from the ground truth
    if args.data == 'nyudepthv2':
        if not args.evaluate:
            train_dataset = NYUDataset(traindir, type='train')
        val_dataset = NYUDataset(valdir, type='val')
    else:
        raise RuntimeError('Dataset not found.' +
                           'The dataset must be either of nyudepthv2, kitti, or zed.')

    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id:np.random.seed(work_id))
            # worker_init_fn ensures different sampling patterns for each data loading thread

    print("=> data loaders created.")
    return train_loader, val_loader

def parse_command():
    data_names = ['nyudepthv2']
    modality_names = MyDataloader.modality_names


    import argparse
    parser = argparse.ArgumentParser(description='SSemi-Supervised')
    parser.add_argument('--save-path', default='', type=str, metavar='PATH',
                        help='path to save things')
    parser.add_argument('--path', default='', type=str, metavar='PATH',
                        help='path to load train and validation')
    parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
    parser.add_argument('--epoch-size', default=500, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
    parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=15)
    parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH',
                    help='path to pre-trained dispnet model')
    parser.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH',
                    help='path to pre-trained Exp Pose net model')
    parser.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=0)
    parser.add_argument('--max-depth', default=80, type=float, metavar='D',
                        help='cut-off depth of sparsifier, negative values means infinity (default: inf [m])')

    parser.add_argument('--data', metavar='DATA', default='nyudepthv2',
                        choices=data_names,
                        help='dataset: ' + ' | '.join(data_names) + ' (default: nyudepthv2)')
    # parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb', choices=modality_names,
    #                     help='modality: ' + ' | '.join(modality_names) + ' (default: rgb)')
    parser.add_argument('-s', '--num-samples', default=0, type=int, metavar='N',
                        help='number of sparse depth samples (default: 0)')

    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')

    parser.add_argument('-b', '--batch-size', default=8, type=int, help='mini-batch size (default: 8)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate (default 0.01)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='path to pretrained checkpoint to begin training from')
    parser.add_argument('-e', '--evaluate', dest='evaluate', type=str, default='',
                        help='evaluate model on validation set')
    parser.add_argument('--no-pretrain', dest='pretrained', action='store_false',
                        help='not to use ImageNet pre-trained weights')
    
    parser.add_argument('--photometric', type=float, help='weight for photometric loss', metavar='W', default=0.75)
    parser.add_argument('--vnl-loss', type=float, help='weight for VNL loss', metavar='W', default=0.5)
    parser.add_argument('--l1', type=float, help='weight for L1 loss', metavar='W', default=0.2)

    parser.add_argument('--lpg', action='store_true', help='to use LPG constraint')
    
    parser.add_argument('--export', default='', type=str, help='path to pre-trained model to load to export to ONNX')
    parser.set_defaults(pretrained=True)
    args = parser.parse_args()
    return args

def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch-1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)
def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0,1,low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0,max_value,resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:,i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)

COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}

def tensor2array(tensor, max_value=None, colormap='rainbow'):
    try:
        tensor = tensor.detach().cpu()
    except Exception as e:
        st()
        print(e)
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
    return array