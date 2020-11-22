from __future__ import division
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from inverse_warp import inverse_warp, pose_vec2mat
from torch.autograd import Variable
import pytorch_ssim
import ipdb
st = ipdb.set_trace
# cuda = True if torch.cuda.is_available() else False
# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
adversarial_loss = torch.nn.MSELoss()
torch.manual_seed(125)
torch.cuda.manual_seed_all(125) 
torch.backends.cudnn.deterministic = True
#import pdb

def structured_sim_loss(pred_img, gt_img) :
    ssim_loss = pytorch_ssim.SSIM(window_size = 10)
    ssim_loss_val = ssim_loss(pred_img, gt_img)
    return ssim_loss_val

def photometric_reconstruction_loss(tgt_img, ref_imgs,
                                    depth, explainability_mask, pose, intrinsics=None,
                                    rotation_mode='euler', padding_mode='zeros'):
    def one_scale(depth, explainability_mask):
        assert(explainability_mask is None or depth.size()[2:] == explainability_mask.size()[2:])
        assert(pose.size(1) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        if not intrinsics:
            intrinsics = intrinsics_global
        print("intrinsics", intrinsics)

        tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
        ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)

        warped_imgs = []
        diff_maps = []

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_img_warped, valid_points = inverse_warp(ref_img, depth[:,0], current_pose,
                                                        intrinsics_scaled,
                                                        rotation_mode, padding_mode)
            diff = (tgt_img_scaled - ref_img_warped) * valid_points.unsqueeze(1).float()

            if explainability_mask is not None:
                diff = diff * explainability_mask[:,i:i+1].expand_as(diff)

            reconstruction_loss += diff.abs().mean()
            assert((reconstruction_loss == reconstruction_loss).item() == 1)

            warped_imgs.append(ref_img_warped[0])
            diff_maps.append(diff[0])

        return reconstruction_loss, warped_imgs, diff_maps

    warped_results, diff_results = [], []
    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    total_loss = 0
    for d, mask in zip(depth, explainability_mask):
        loss, warped, diff = one_scale(d, mask)
        total_loss += loss
        warped_results.append(warped)
        diff_results.append(diff)
    return total_loss, warped_results, diff_results

# def photometric_reconstruction_loss(args, discriminator, rgb_sequence_except_first,
#                                     depth, explainability_mask, poses, __p,__u, intrinsics=None,
#                                     rotation_mode='euler', padding_mode='zeros'):
#     def one_scale(depth, explainability_mask, tgt_imgs, ref_imgs, pose):
#         assert(explainability_mask is None or depth.size()[2:] == explainability_mask.size()[2:])
#         #st()
#         reconstruction_loss = 0
#         gan_loss=0
#         b, _, h, w = depth.size()
#         # h,w=128,416
#         downscale = tgt_imgs.size(2)/h
#         # st()
#         tgt_imgs_scaled = [F.interpolate(tgt_imgs, (h, w), mode='area')]
#         ref_imgs_scaled = [F.interpolate(ref_imgs, (h, w), mode='area')]
#         if not intrinsics:
#             intrinsics = intrinsics_global
#         print("intrinsics", intrinsics)
#         intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
#         valid = Variable(torch.cuda.FloatTensor(tgt_imgs_scaled[0].size(0), 1).fill_(1), requires_grad=False)
#         fake = Variable(torch.cuda.FloatTensor(tgt_imgs_scaled[0].size(0), 1).fill_(0), requires_grad=False)
#         warped_imgs = []
#         diff_maps = []
#         for i, (ref_img, tgt_img) in enumerate(zip(ref_imgs_scaled, tgt_imgs_scaled)):#p
#             # pdb.set_trace()
#             #current_pose = pose[:, i]
#             curr_ref = ref_img
#             curr_tgt = tgt_img
#             current_pose = pose[:,0,:]
#             current_depth = depth[:,0,:,:]
#             ref_img_warped, valid_points = inverse_warp(curr_ref, current_depth, current_pose,intrinsics_scaled,rotation_mode, padding_mode)
#             diff = (curr_tgt - ref_img_warped) * valid_points.unsqueeze(1).float()

#             if explainability_mask is not None:
#                 diff = diff * explainability_mask[:,i:i+1].expand_as(diff)

#             reconstruction_loss += diff.abs().mean()
#             #print (reconstruction_loss)
#             assert((reconstruction_loss == reconstruction_loss).item() == 1)
#             # pdb.set_trace()

#             warped_imgs.append(ref_img_warped)
#             diff_maps.append(diff)
#         # print(reconstruction_loss, warped_imgs, diff_maps,gan_loss)
#             if args.gan_loss_weight>0:
#                 ref_img_rescaled = F.interpolate(ref_img_warped, (128, 416), mode='area')#reshape
#                 tgt_img_rescaled = F.interpolate(curr_tgt, (128, 416), mode='area')

#                 real_loss = adversarial_loss(discriminator(tgt_img_rescaled.detach()), valid)
#                 fake_loss = adversarial_loss(discriminator(ref_img_rescaled.detach()), fake)
#                 gan_loss += (real_loss + fake_loss) / 2
#             else:
#                 gan_loss=0

#         return reconstruction_loss, warped_imgs, diff_maps, gan_loss

#     warped_results, diff_results = [], []
#     if type(explainability_mask) not in [tuple, list]:
#         explainability_mask = [explainability_mask]
#     if type(depth) not in [list, tuple]:
#         depth = [depth]
#     total_gan_loss=0
#     total_photo_loss = 0

#     ref_imgs = rgb_sequence_except_first[:-1,:,:,:,:]#barring last one
#     tgt_imgs = rgb_sequence_except_first[1:,:,:,:,:]#1st image can not be rpedicted
#     num_in_sequence = ref_imgs.size()[0]
#     for scale_idx, (d, mask) in enumerate(zip(depth, explainability_mask)):#s
#         # print(d.shape)
#         d = d[1:,:,:,:,:]
#         #print (num_in_sequence)
#         # st()
#         photo_loss = 0
#         gan_loss = 0
#         temp_warped_list = []
#         for seq in range(num_in_sequence):
#             seq_photo_loss, warped, diff ,seq_gan_loss= one_scale(d[seq,:,:,:,:], mask[seq,:,:,:,:], tgt_imgs[seq,:,:,:,:], ref_imgs[seq,:,:,:,:], poses[seq,:,:,:])
#             photo_loss += seq_photo_loss
#             gan_loss += seq_gan_loss
#             #print(type(warped), type(warped[0]))
#             temp_warped_list.append(warped[0])
#         total_photo_loss += photo_loss/num_in_sequence
#         total_gan_loss += gan_loss/num_in_sequence
#         diff_results.append(diff)
#         warped_results.append(temp_warped_list)
#     return total_photo_loss,total_gan_loss, warped_results, diff_results

'''
+pose : torch.Size([seq_len-2, batch, 1, 6])
+codes: torch.Size([seq_len-1, batch, 128])
+depths: torch.Size([seq_len-1, batch, 1, 128, 416])
+rgbs: torch.Size([seq_len-1, batch, 3, 128, 416])
'''
def trajectory_loss(pose, pose_net, __p, __u, codes, depths, rgbs, rotation_mode):

    # Calculate poses using pose net
    pose_t2, pose_t4, pose_t8 = pose_net(depths, rgbs, __p, __u, trajectory_loss_call=True)
    pose_t2_mat, pose_t4_mat, pose_t8_mat = __u(pose_vec2mat(__p(pose_t2).squeeze(1), rotation_mode)), __u(pose_vec2mat(__p(pose_t4).squeeze(1), rotation_mode)), __u(pose_vec2mat(__p(pose_t8).squeeze(1), rotation_mode))
    pose_t2_mat, pose_t4_mat, pose_t8_mat = append_row_in_matrix_4x4(pose_t2_mat), append_row_in_matrix_4x4(pose_t4_mat), append_row_in_matrix_4x4(pose_t8_mat)
    
    # Now calculate relative poses by multiplication. Try to do without for loop.
    pose_mat = __u(pose_vec2mat(__p(pose).squeeze(1), rotation_mode))  # [seq_len-2,B,3,4] # torch.Size([13, 4, 3, 4])
    
    # Make pose_mat 4x4
    pose_mat = append_row_in_matrix_4x4(pose_mat)

    # for t=2
    mat1, mat2 = __p(pose_mat[:-1]), __p(pose_mat[1:])
    mult_pose_t2 = mat2 @ mat1 
    mult_pose_t2 = __u(mult_pose_t2)

    # for t=4
    mat1, mat2, mat3, mat4 = __p(pose_mat[:-3]), __p(pose_mat[1:-2]), __p(pose_mat[2:-1]), __p(pose_mat[3:])
    mult_pose_t4 = mat4 @ mat3 @ mat2 @ mat1  
    mult_pose_t4 = __u(mult_pose_t4)

    # for t=8
    mat1, mat2, mat3, mat4, mat5, mat6, mat7, mat8 = __p(pose_mat[:-7]), __p(pose_mat[1:-6]), __p(pose_mat[2:-5]), __p(pose_mat[3:-4]), __p(pose_mat[4:-3]), __p(pose_mat[5:-2]), __p(pose_mat[6:-1]), __p(pose_mat[7:])
    mult_pose_t8 = mat8 @ mat7 @ mat6 @ mat5 @ mat4 @ mat3 @ mat2 @ mat1
    mult_pose_t8 = __u(mult_pose_t8)
    
    # Now take difference
    loss_t2 = torch.abs(mult_pose_t2 - pose_t2_mat).sum() / mult_pose_t2.shape[0]
    loss_t4 = torch.abs(mult_pose_t4 - pose_t4_mat).sum() / mult_pose_t4.shape[0]
    loss_t8 = torch.abs(mult_pose_t8 - pose_t8_mat).sum() / mult_pose_t8.shape[0]
    
    traj_loss = loss_t2 + loss_t4 + loss_t8
    return traj_loss


def append_row_in_matrix_4x4(pose_mat):
    assert pose_mat.ndim == 4
    # Make pose_mat 4x4
    lastrow = torch.tensor([0,0,0,1]).cuda().float()
    lastrow = lastrow.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    lastrow = lastrow.repeat(pose_mat.shape[0],pose_mat.shape[1],1,1)
    pose_mat = torch.cat((pose_mat, lastrow), dim=2)  # [seq_len-2,B,4,4]
    return pose_mat

def explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = torch.ones_like(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss

def gradient(pred):#get gradient
    D_dy = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    return D_dx, D_dy

def smooth_loss_sfm(pred_map):
    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]
    loss = 0
    weight = 1.
    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # don't ask me why it works better
    return loss

def smooth_loss(pred_map, input_image, __p, __u) : #4*[14, 4, 1, 128, 416] [56, 3, 128, 416]
    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]
    loss = 0
    weight=1
    for scaled_map in pred_map:
        N,_,H,W = __p(scaled_map).shape
        #st()
        depth_map_packed = __p(scaled_map)
        input_image_scaled = F.interpolate(input_image, size=(H,W))
        grad_x_depth       = torch.abs(depth_map_packed[:, :, :, 1:] - depth_map_packed[:, :, :, :-1])
        grad_y_depth 	   = torch.abs(depth_map_packed[:, :, 1:, :] - depth_map_packed[:, :, :-1, :])

        grad_x_image 	= torch.mean(torch.abs(input_image_scaled[:, :, :,1:] - input_image_scaled[:, :, :, :-1]), dim = 1, keepdim = True)
        grad_y_image    = torch.mean(torch.abs(input_image_scaled[:, :, 1:,:] - input_image_scaled[:, :, :-1, :]), dim = 1, keepdim = True)

        expDelXInImage = torch.exp(-grad_x_image)
        expDelYInImage = torch.exp(-grad_y_image)

        loss += (torch.mean(grad_x_depth*expDelXInImage) + torch.mean(grad_y_depth*expDelYInImage))*weight
        weight /= 2.3  # don't ask me why it works betters
    return loss

def smooth_loss_our(pred_map, input_image, __p, __u) : #4*[14, 4, 1, 128, 416] [56, 3, 128, 416]
    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]
    loss = 0
    weight = 1
    for scaled_map in pred_map:
        N,_,H,W = __p(scaled_map).shape
        # st()
        input_image_scaled = F.interpolate(input_image, size=(H,W))
        #print (scaled_map.shape)#(128, 416)
        dx_map, dy_map = gradient(__p(scaled_map))
        dx_map_val = torch.abs(dx_map)
        dy_map_val = torch.abs(dy_map)

        dx_image, dy_image = gradient(input_image_scaled)
        # dx_image_val = torch.abs(dx_image)
        dx_image_val = dx_image**2
        dx_image_val = torch.sum(dx_image_val,dim=1)
        dx_image_val = torch.sqrt(dx_image_val)
        dx_image_val = dx_image_val.unsqueeze(dim=1) #[56, 1, 127, 416]

        dy_image_val = dy_image**2
        dy_image_val = torch.sum(dy_image_val,dim=1)
        dy_image_val = torch.sqrt(dy_image_val)
        dy_image_val = dy_image_val.unsqueeze(dim=1)#[56, 1, 128, 415]

        dx_image_exp = torch.exp(-1*dx_image_val)
        dy_image_exp = torch.exp(-1*dy_image_val)

        x_mult_map = torch.mul(dx_map_val, dx_image_exp)
        x_mult_map = x_mult_map[:,:,:,1:]#[56, 1, 127, 415]

        y_mult_map = torch.mul(dy_map_val, dy_image_exp) 
        y_mult_map = y_mult_map[:,:,1:,:]#[56, 1, 127, 415]

        mult_map = x_mult_map + y_mult_map
        loss += mult_map.sum()*weight
        weight /= 2.3  # don't ask me why it works betters
    loss = loss/N
    # st()

    return loss

@torch.no_grad()
def compute_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2,x1:x2] = 1

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
