from __future__ import division
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from loss.inverse_warp import inverse_warp, pose_vec2mat
from torch.autograd import Variable
import pdb

st = pdb.set_trace
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
torch.manual_seed(125)
torch.cuda.manual_seed_all(125) 
torch.backends.cudnn.deterministic = True
#import pdb
class PhotoMetricLoss(nn.Module):
    """
    Photometric loss Function.
    """
    def __init__(self, rotation_mode = 'euler', padding_mode='zeros'):
        super().__init__()
        self.rotation_mode = rotation_mode
        self.padding_mode = padding_mode
    
    def one_scale(self, tgt_img, ref_imgs, intrinsics, depth, pose):
        
        assert(pose.size(1) == len(ref_imgs))
        reconstruction_loss = 0
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h
        tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
        ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)

        warped_imgs = []
        diff_maps = []

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_img_warped, valid_points = inverse_warp(ref_img, depth[:,0], current_pose,
                                                        intrinsics_scaled,
                                                        self.rotation_mode, self.padding_mode)
            diff = (tgt_img_scaled - ref_img_warped) * valid_points.unsqueeze(1).float()

            reconstruction_loss += diff.abs().mean()

        return reconstruction_loss

    def forward(self, tgt_img, ref_imgs, intrinsics, depth, pose):
        if type(depth) not in [list, tuple]:
            depth = [depth]
        total_loss = 0
        for d in depth:
            loss = self.one_scale(tgt_img, ref_imgs, intrinsics, d, pose)
            total_loss += loss
        return total_loss