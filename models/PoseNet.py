import torch
import torch.nn as nn
from torch import sigmoid
from torch.nn.init import xavier_uniform_, zeros_


def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )


class PoseNet(nn.Module):

    def __init__(self, nb_ref_imgs=2, output_exp=False): #14
        super(PoseNet, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(8, conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])

        self.pose_pred = nn.Conv2d(conv_planes[6], 6, kernel_size=1, padding=0)

        if self.output_exp:
            upconv_planes = [256, 128, 64, 32, 16]
            self.upconv5 = upconv(conv_planes[4],   upconv_planes[0])
            self.upconv4 = upconv(upconv_planes[0], upconv_planes[1])
            self.upconv3 = upconv(upconv_planes[1], upconv_planes[2])
            self.upconv2 = upconv(upconv_planes[2], upconv_planes[3])
            self.upconv1 = upconv(upconv_planes[3], upconv_planes[4])

            self.predict_mask4 = nn.Conv2d(upconv_planes[1], 1, kernel_size=3, padding=1)
            self.predict_mask3 = nn.Conv2d(upconv_planes[2], 1, kernel_size=3, padding=1)
            self.predict_mask2 = nn.Conv2d(upconv_planes[3], 1, kernel_size=3, padding=1)
            self.predict_mask1 = nn.Conv2d(upconv_planes[4], 1, kernel_size=3, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    def posenet_for_trajectory_loss(self, rgbd, __p, __u):
        # for t=2
        rgbd1_t2 = rgbd[:-2]
        rgbd2_t2 = rgbd[2:]
        input_t2 = __p(torch.cat((rgbd1_t2, rgbd2_t2), dim=2))

        rgbd1_t4 = rgbd[:-4]
        rgbd2_t4 = rgbd[4:]
        input_t4 = __p(torch.cat((rgbd1_t4, rgbd2_t4), dim=2))

        rgbd1_t8 = rgbd[:-8]
        rgbd2_t8 = rgbd[8:]
        input_t8 = __p(torch.cat((rgbd1_t8, rgbd2_t8), dim=2))

        pose_t2 = __u(self.apply_pose_layers(input_t2))
        pose_t4 = __u(self.apply_pose_layers(input_t4))
        pose_t8 = __u(self.apply_pose_layers(input_t8))
        return pose_t2, pose_t4, pose_t8


    def apply_pose_layers(self, input):
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), 1, 6)
        return pose


    def forward(self, depth, ref_imgs, __p, __u, trajectory_loss_call = False):
        # st()

        assert(len(ref_imgs) == self.nb_ref_imgs)
        assert len(depth) == len(ref_imgs)

        concatenated_rgbd = torch.cat((ref_imgs, depth), dim=2)
        
        if trajectory_loss_call:
            return self.posenet_for_trajectory_loss(concatenated_rgbd, __p, __u)

        rgbd1 = concatenated_rgbd[:-1] 
        rgbd2 = concatenated_rgbd[1:]

        input = __p(torch.cat((rgbd1, rgbd2), dim=2))  # torch.Size([13, 4, 8, 128, 416])


        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), 1, 6)
        pose = __u(pose)

        if self.output_exp:
            out_upconv5 = self.upconv5(out_conv5  )[:, :, 0:out_conv4.size(2), 0:out_conv4.size(3)]
            out_upconv4 = self.upconv4(out_upconv5)[:, :, 0:out_conv3.size(2), 0:out_conv3.size(3)]
            out_upconv3 = self.upconv3(out_upconv4)[:, :, 0:out_conv2.size(2), 0:out_conv2.size(3)]
            out_upconv2 = self.upconv2(out_upconv3)[:, :, 0:out_conv1.size(2), 0:out_conv1.size(3)]
            out_upconv1 = self.upconv1(out_upconv2)[:, :, 0:input.size(2), 0:input.size(3)]

            exp_mask3 = __u(sigmoid(self.predict_mask3(out_upconv3)))
            exp_mask4 = __u(sigmoid(self.predict_mask4(out_upconv4)))
            exp_mask2 = __u(sigmoid(self.predict_mask2(out_upconv2)))
            exp_mask1 = __u(sigmoid(self.predict_mask1(out_upconv1)))
        else:
            exp_mask4 = None
            exp_mask3 = None
            exp_mask2 = None
            exp_mask1 = None

        if self.training:
            return [exp_mask1, exp_mask2, exp_mask3, exp_mask4], pose
        else:
            return exp_mask1, pose
