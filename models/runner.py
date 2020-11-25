import sys
sys.path.append('./../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
import models
from utils import save_checkpoint, log_output_tensorboard
import utils
import pdb
import numpy as np
import pdb
st = pdb.set_trace
import matplotlib.pyplot as plt
from loss.VNL import VNL_Loss
from loss.PhotoMetric import PhotoMetricLoss
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
        self.disp_net = models.DispNet(lpg_flag = args.lpg, max_depth=args.max_depth).to(self.device)
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
        self.virtual_normal_loss = VNL_Loss(focal_x= 519.0, focal_y= 519.0, input_size=(448,448))
        self.photometric_loss = PhotoMetricLoss()

        self.w_photo = args.photometric
        self.w_vnl = args.vnl_loss
        self.w_l1 = args.l1
    
    def forward(self, img, ref_imgs, intrinsics, gt_depth, log_losses, log_output, tb_writer, n_iter, ret_depth, mode = 'train', args=None):
        disparities = self.disp_net(img) # 4 [8, 1, 448, 448]
        if type(disparities) not in [tuple, list]:
            disparities = [disparities]
        depth = [1/disp for disp in disparities]
        
        if ret_depth: #inference call
            return depth

        _, pose = self.pose_net(img, ref_imgs)	# pose = [seq_len-2, batch , 6]

        #### code for loss calculation
        loss = 0
        if self.w_l1 > 0:
            l1_loss = self.l1_loss(depth[0], gt_depth)
            loss += self.w_l1 * l1_loss
        
        if self.w_vnl > 0:
            vnl_loss = self.virtual_normal_loss(depth[0], gt_depth)
            loss += self.w_vnl * vnl_loss
        
        if self.w_photo > 0:
            photometric_loss = self.photometric_loss(img, ref_imgs, intrinsics, depth[0], pose)
            loss += self.w_photo * photometric_loss
        
        # Logging
        if log_losses:
            # print("Logging Scalars")
            if self.w_l1 > 0:
                tb_writer.add_scalar(mode+'/l1_loss', l1_loss.item(), n_iter)
            if self.w_vnl > 0:
                tb_writer.add_scalar(mode+'/vnl_loss', vnl_loss.item(), n_iter)
            if self.w_photo > 0:
                tb_writer.add_scalar(mode+'/photometric_loss', photometric_loss.item(), n_iter)
            
            tb_writer.add_scalar(mode+'/total_loss', loss.item(), n_iter)

        if log_output: 
            print("Logging Training Images")
            tb_writer.add_image(mode+'/train_input', tensor2array(img[0]), n_iter)
            output_depth = depth[0][0,0,:,:]
            tb_writer.add_image(mode+'/train_depth', tensor2array(output_depth, max_value=None), n_iter)
            output_disp = 1.0/output_depth
            tb_writer.add_image(mode+'/train_disp', tensor2array(output_disp, max_value=None, colormap='magma'), n_iter)

        return loss