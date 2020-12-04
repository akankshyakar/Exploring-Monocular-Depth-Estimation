import sys
sys.path.append('./../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
import models
from utils import save_checkpoint
import utils
import pdb
import numpy as np
import pdb
st = pdb.set_trace
import matplotlib.pyplot as plt
from loss.VNL import VNL_Loss
from loss.PhotoMetric import PhotoMetricLoss
from loss.im2pcl import CoordsRegressionLoss
from loss.ordinal import OrdinalRegressionLoss
from utils import tensor2array
torch.manual_seed(125)
torch.cuda.manual_seed_all(125) 
torch.backends.cudnn.deterministic = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



class Runner(nn.Module):
    def __init__(self, args):
        super(Runner, self).__init__()
        self.hidden_dim = 128
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.disp_net = models.DispNet(lpg_flag = args.lpg, max_depth = args.max_depth, \
            ord_num=args.ord_num, ordinal_flag=args.ordinal>0).to(self.device)
        self.pose_net = models.PoseExpNet(nb_ref_imgs=2,output_exp=False).to(self.device)

        if args.pretrained_exp_pose:
            print("=> using pre-trained weights for explainabilty and pose net")
            weights = torch.load(args.pretrained_exp_pose)
            self.pose_net.load_state_dict(weights['state_dict'], strict=False)
        else:
            self.pose_net.init_weights()

        if args.pretrained_disp:
            print("=> using pre-trained weights for Dispnet")
            weights = torch.load(args.pretrained_disp)
            self.disp_net.load_state_dict(weights['state_dict'])
        else:
            self.disp_net.init_weights()
        self.args = args
        self.l1_loss = nn.L1Loss()
        # TODO(abhorask): Fix focal_x, focal_y values
        # TODO(abhroask): change input size depending on dataset
        if args.data == 'nyudepthv2':
            input_size=(448, 448)
        else:
            input_size=(192, 256)
        self.virtual_normal_loss = VNL_Loss(focal_x= 519.0, focal_y= 519.0, input_size=input_size)
        self.photometric_loss = PhotoMetricLoss()
        self.coords_regression_loss = CoordsRegressionLoss()
        self.ordinal_regression_loss = OrdinalRegressionLoss(args.ord_num, args.max_depth)
    
    def forward(self, img, ref_imgs, intrinsics, gt_depth, mask_gt, world_coords_gt, log_losses, log_output, tb_writer, n_iter, ret_depth, mode = 'train', args=None):
        disp_and_params = self.disp_net(img) # 4 [8, 1, 448, 448]
        predicted_params_ranges = disp_and_params[-1]
        disps = disp_and_params[:-1]

        if args.ordinal > 0:
            log_prob = disps[-1]
            disps = disps[:-1]
        
        if type(disps) not in [tuple, list]:
            disps = [disps]
        
        depth = [1/disp for disp in disps[:-1]]

        if ret_depth: #inference call
            return depth
        
        if args.data == 'nyudepthv2' and args.photometric > 0:
            _, pose = self.pose_net(img, ref_imgs)	# pose = [seq_len-2, batch , 6]

        #### code for loss calculation
        loss = 0
        if args.l1 > 0:
            # TODO(abhorask): make more general
            l1_loss = self.l1_loss(depth[0], gt_depth)
            loss += args.l1 * l1_loss
        
        if args.vnl_loss > 0:
            vnl_loss = self.virtual_normal_loss(gt_depth, depth[0])
            loss += args.vnl_loss * vnl_loss
        
        if args.photometric > 0:
            photometric_loss = self.photometric_loss(img, ref_imgs, intrinsics, depth[0], pose)
            loss += args.photometric * photometric_loss

        if args.im2pcl > 0:
            d = depth[0].squeeze(1)
            im2pcl_loss = self.coords_regression_loss(predicted_params_ranges, d, world_coords_gt, mask_gt, args)
            loss += args.im2pcl * im2pcl_loss
        
        if args.ordinal > 0:
            gt_disp = gt_depth.clone()
            gt_disp[gt_depth>0] = 1/gt_disp[gt_depth>0]
            ordinal_loss = self.ordinal_regression_loss(log_prob, gt_disp)
            loss += args.ordinal * ordinal_loss

        # Logging
        if log_losses:
            # print("Logging Scalars")
            if args.l1 > 0:
                tb_writer.add_scalar(mode+'/l1_loss', l1_loss.item(), n_iter)
            if args.vnl_loss > 0:
                tb_writer.add_scalar(mode+'/vnl_loss', vnl_loss.item(), n_iter)
            if args.photometric > 0:
                tb_writer.add_scalar(mode+'/photometric_loss', photometric_loss.item(), n_iter)
            if args.im2pcl > 0:
                tb_writer.add_scalar(mode+'/coords_regression_loss', im2pcl_loss.item(), n_iter)
            if args.ordinal > 0:
                tb_writer.add_scalar(mode+'/ordinal_regression_loss', ordinal_loss.item(), n_iter)

            tb_writer.add_scalar(mode+'/total_loss', loss.item(), n_iter)

        if log_output: 
            print("Logging Training Images")
            tb_writer.add_image(mode+'/train_input', tensor2array(img[0]), n_iter)
            output_depth = depth[0][0,0,:,:]
            tb_writer.add_image(mode+'/train_depth', tensor2array(output_depth, max_value=None), n_iter)
            output_disp = 1.0/output_depth
            tb_writer.add_image(mode+'/train_disp', tensor2array(output_disp, max_value=None, colormap='magma'), n_iter)

        return loss
