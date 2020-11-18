import sys
sys.path.append('./../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
import models
from models import optical_flow_extractor
from loss_functions import trajectory_loss, structured_sim_loss, photometric_reconstruction_loss, explainability_loss, smooth_loss, smooth_loss_sfm, compute_errors
from utils import tensor2array, save_checkpoint, save_path_formatter, log_output_tensorboard,vis_optflow
import utils
import pdb
import numpy as np
import ipdb
st = ipdb.set_trace
import matplotlib.pyplot as plt
torch.manual_seed(125)
torch.cuda.manual_seed_all(125) 
torch.backends.cudnn.deterministic = True
# import numpy as np
class LSTMRunner(nn.Module):
    def __init__(self, args):
        super(LSTMRunner, self).__init__()
        self.hidden_dim = 128
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.disp_net = models.DispNetS().to(self.device)
        self.pose_exp_net = models.PoseExpNet(nb_ref_imgs=args.sequence_length-1,output_exp=True).to(self.device)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim).to(self.device)

        if args.pretrained_exp_pose:
            print("=> using pre-trained weights for explainabilty and pose net")
            weights = torch.load(args.pretrained_exp_pose)
            self.pose_exp_net.load_state_dict(weights['state_dict'], strict=False)
        else:
            self.pose_exp_net.init_weights()

        if args.pretrained_disp:
            print("=> using pre-trained weights for Dispnet")
            weights = torch.load(args.pretrained_disp)
            self.disp_net.load_state_dict(weights['state_dict'])
        else:
            self.disp_net.init_weights()
        if args.discriminator_type=="Encoder":
            self.discriminator = models.Discriminator_encoder().to(self.device)
        else:
            self.discriminator = models.Discriminator().to(self.device)
        self.discriminator.init_weights()
        # Added encoder
        self.encoder = models.Encoder(args).to(self.device)
        if args.pretrained_net is None:
            self.encoder.init_weights()
        self.args = args

        self.__p = lambda x: utils.pack_seqdim(x, args.batch_size)
        self.__u = lambda x: utils.unpack_seqdim(x, args.batch_size)

    def get_optical_flow(self, imgs):
        # st()
        imgs=(imgs*0.5+0.5)*255 #unnormalise image for Optical flow
        return optical_flow_extractor.extract_OptFlow(imgs.cpu())

    
    def forward(self, tgt_img, ref_imgs, intrinsics, log_losses, log_output, tb_writer, n_iter, ret_depth, mode = 'train'):

        # compute output
        w1, w2, w3 ,w4 = self.args.appearance_loss_weight, self.args.smooth_loss_weight,\
                         self.args.trajectory_loss_weight, self.args.gan_loss_weight

        rgbs = torch.stack(ref_imgs).to(self.device)							# [seq_len , batch , imgSize]
        assert rgbs.shape[0] == self.args.sequence_length, "First dimension should be equal to sequence length."

        optical_flow_imgs = self.get_optical_flow(rgbs).to(self.device)#[14, 4, 2, 128, 416] # [seq_len-1 , batch , imgSize]
        # Convert optical flow images into codes using encoder.
        optical_codes = self.encoder(self.__p(optical_flow_imgs))#[56, 128]					# [seq_len-1 x batch, 128]
        optical_codes  = self.__u(optical_codes)#torch.Size([14, 4, 128])					# [seq_len-1, batch , 128]
        rgbs_from_first = rgbs[1:] # First image is not used.
        # Pass all optical codes through lstm
        if self.args.use_lstm:
            # print("LSTM is being used")
            lstm_out, _ = self.lstm(optical_codes)								# [seq_len-1, batch , 128]							
            assert rgbs_from_first.shape[0:2] == lstm_out.shape[0:2]
            # Calculate depth using corresponding images and LSTM seq
            disparities = self.disp_net(self.__p(rgbs_from_first), self.__p(lstm_out), self.__u) 		# 4*[seq_len-1 x batch , 128]
        else:
            #Calculate depth using corresponding images and optical codes
            assert rgbs_from_first.shape[0:2] == optical_codes.shape[0:2]
            disparities = self.disp_net(self.__p(rgbs_from_first), self.__p(optical_codes), self.__u)
        # pdb.set_trace()
        if type(disparities) not in [tuple, list]:
            disparities = [disparities]
        depth = [1/disp for disp in disparities]

        if ret_depth:
            return depth

        explainability_mask, pose = self.pose_exp_net(depth[0], rgbs_from_first, self.__p, self.__u)	# pose = [seq_len-2, batch , 6]
        if type(explainability_mask) not in [tuple, list]:
            explainability_mask = [explainability_mask]


        ###############################################
        # Calculating losses
        ###############################################

        # Trajectory loss
        if self.args.trajectory_loss_weight > 0:
            if self.args.use_lstm:  #for LSTM as input to traj loss
                traj_loss = trajectory_loss(pose, self.pose_exp_net, self.__p, self.__u,\
                                            lstm_out, depth[0], rgbs_from_first, self.args.rotation_mode)
            else:#for Optical code as input to traj loss
                traj_loss = trajectory_loss(pose, self.pose_exp_net, self.__p, self.__u, \
                                    optical_codes, depth[0], rgbs_from_first, self.args.rotation_mode)
        else:
            traj_loss=0



        # Appearance loss and Gan loss
        loss_1, gan_loss, warped, diff = photometric_reconstruction_loss(self.args, self.discriminator, \
                                         rgbs_from_first, intrinsics,depth, explainability_mask, pose,self.__p, \
                                         self.__u,self.args.rotation_mode, self.args.padding_mode)
        if w1 > 0:
            loss_2 = explainability_loss(explainability_mask)
        else:
            loss_2 = 0
       
        des_seq = self.__p(rgbs_from_first[1:])
        #src_seq = torch.Tensor(warped[0]).cuda()
        src_seq = warped[0]
        src_seq = [elt.unsqueeze(0) for elt in src_seq]
        src_seq = torch.cat(src_seq, dim=0)
        src_seq = self.__p(src_seq)
        loss_4 = structured_sim_loss(src_seq, des_seq)
        ap_loss = loss_2 + 0.15 * loss_1 + (0.425/rgbs_from_first.size()[1])*(1 - loss_4)
        
        if self.args.use_smoothlossSFM: # Depth smoothness loss from SFM 
            depth_reg_loss = smooth_loss_sfm(depth)
        else:
            # Depth smoothness loss
            depth_reg_loss = smooth_loss(depth, self.__p(rgbs_from_first), self.__p, self.__u)
        # Total loss

        loss = w1*ap_loss+ w2*depth_reg_loss+ w3*traj_loss+ w4*gan_loss

        ###############################################
        # Logging
        ###############################################
        if log_losses:
            # print("Logging Scalars")
            tb_writer.add_scalar(mode+'/photometric_loss', loss_1.item(), n_iter)
            if w1 > 0:
                tb_writer.add_scalar(mode+'/explanability_loss', loss_2.item(), n_iter)
                tb_writer.add_scalar(mode+'/appearance_loss', ap_loss.item(), n_iter)
                tb_writer.add_scalar(mode+'/SSIM_loss', loss_4.item(), n_iter)
            if w2 > 0:
                tb_writer.add_scalar(mode+'/disparity_smoothness_loss', depth_reg_loss.item(), n_iter)
            if w4 > 0:
                tb_writer.add_scalar(mode+'/GAN_loss', gan_loss.item(), n_iter)
            if w3 > 0:
                tb_writer.add_scalar(mode+'/traj_loss', traj_loss.item(), n_iter)
            tb_writer.add_scalar(mode+'/total_loss', loss.item(), n_iter)

        if log_output: 
            # print("Logging Images")
            tb_writer.add_image(mode+'/train_input', tensor2array(tgt_img[0]), n_iter)
            flow_to_show=vis_optflow(optical_flow_imgs[0][0].permute(1,2,0).cpu().numpy())
            tb_writer.add_image(mode+'/train_optical_flow', flow_to_show, n_iter)
            # st()
            output_depth = depth[0][-1,0,:,:,:]
            tb_writer.add_image(mode+'/train_depth', tensor2array(output_depth[0], max_value=None), n_iter)
            output_disp = 1.0/output_depth
            tb_writer.add_image(mode+'/train_disp', tensor2array(output_disp[0], max_value=None, colormap='magma'), n_iter)
            tb_writer.add_image(mode+'/train_warped', tensor2array(warped[0][-1][0,:,:,:]), n_iter)
            tb_writer.add_image(mode+'/train_diff', tensor2array(diff[0][-1][0,:,:,:]*0.5), n_iter)
            mask_to_show = tensor2array(explainability_mask[0][0][0], max_value=1, colormap='bone')
            tb_writer.add_image(mode+'/train_exp_mask', mask_to_show, n_iter)

        return loss, ap_loss, depth_reg_loss, traj_loss, gan_loss
