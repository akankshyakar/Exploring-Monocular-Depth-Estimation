import numpy as np
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalRegressionLayer(nn.Module):
    def __init__(self, ord_num, max_depth, min_depth):
        super(OrdinalRegressionLayer, self).__init__()
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.ord_num = ord_num

    def forward(self, x):
        """
        :param x: NxCxHxW, N is batch_size, C is channels of features
        :return: ord_label is ordinal outputs for each spatial locations , N x 1 x H x W
                 ord prob is the probability of each label, N x OrdNum x H x W
        """
        N, C, H, W = x.size()
        labels = torch.linspace(self.min_depth, self.max_depth, self.ord_num).to(x.device)

        log_prob = F.log_softmax(x, dim=1).view(N, C, H, W)

        ord_prob = F.softmax(x, dim=1)
        depth = ord_prob * labels[None, :, None, None]
        depth = torch.sum(depth, dim=1)
        # import pdb; pdb.set_trace()

        disp = 1/depth

        return log_prob, disp


class OrdinalRegressionLoss(object):

    def __init__(self, ord_num, max_depth):
        self.ord_num = ord_num
        self.max_depth = max_depth

    def __call__(self, log_prob, gt):
        """
        :param prob: ordinal regression probability, N x Ord Num x H x W, torch.Tensor
        :param gt: depth ground truth, NX1XHxW, torch.Tensor
        :return: loss: loss value, torch.float
        """
        N, C, H, W = log_prob.shape
        eps=1e-5
        
        # import pdb; pdb.set_trace()

        valid_mask = (gt > 0.).squeeze(1)

        gt_label = torch.linspace(0, self.max_depth, self.ord_num).view(1, self.ord_num, 1, 1).to(gt.device)
        gt_label = gt_label.repeat(N, 1, H, W)

        gt_label = gt_label - gt
        gt_label = -gt_label ** 2
        gt_label = F.softmax(gt_label, dim=1)

        entropy = - (gt_label * log_prob)
        loss = torch.sum(entropy, dim=1)[valid_mask]

        return loss.mean()