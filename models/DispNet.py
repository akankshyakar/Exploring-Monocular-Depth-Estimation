import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from loss.LPG import reduction_1x1, local_planar_guidance
import pdb
st = pdb.set_trace
def downsample_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )


def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )


def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]

def LPG_func(feat, reduction, lpg, max_depth, scale):
    '''
    feat: value from prev layer
    reduction: function which reduces to 3
    lpg: constraint of planar aaumption
    max_depth : set for dataset 
    '''
    # st()
    reduc = reduction(feat)
    plane_normal = reduc[:, :3, :, :]
    plane_normal = F.normalize(plane_normal, 2, 1)
    plane_dist = reduc[:, 3, :, :]
    plane_eq = torch.cat([plane_normal, plane_dist.unsqueeze(1)], 1)
    depth = lpg(plane_eq)
    depth_scaled = depth.unsqueeze(1) / max_depth
    depth_scaled_ds = F.interpolate(depth_scaled, scale_factor=scale, mode='nearest')
    depth = depth.unsqueeze(1)

    return depth, depth_scaled, depth_scaled_ds

class DispNet(nn.Module):

    def __init__(self, alpha=10, beta=0.01, lpg_flag = True, max_depth = 80):
        super(DispNet, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.lpg_flag = lpg_flag
        self.max_depth = max_depth
        num_features =512

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv(3,              conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_conv(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_conv(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_conv(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_conv(conv_planes[5], conv_planes[6])

        
        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv(upconv_planes[5], upconv_planes[6])

        self.iconv7 = conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv(1 + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = predict_disp(upconv_planes[3])
        self.predict_disp3 = predict_disp(upconv_planes[4])
        self.predict_disp2 = predict_disp(upconv_planes[5])
        self.predict_disp1 = predict_disp(upconv_planes[6])

        self.reduc8x8   = reduction_1x1(num_features // 4, num_features // 4, self.max_depth)
        self.lpg8x8     = local_planar_guidance(8)
        self.reduc4x4   = reduction_1x1(num_features // 8, num_features // 8, self.max_depth)
        self.lpg4x4     = local_planar_guidance(4)
        self.reduc2x2   = reduction_1x1(num_features // 16, num_features // 16, self.max_depth)
        self.lpg2x2     = local_planar_guidance(2)

        self.upconv1_lpg    = upconv(num_features // 16, num_features // 16)
        self.reduc1x1   = reduction_1x1(num_features // 32, num_features // 32, self.max_depth, is_final=True)
        self.conv1_lpg      = torch.nn.Sequential(nn.Conv2d(4, num_features // 16, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.get_depth  = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False),
                                              nn.Sigmoid())


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        # st() #[8, 3, 448, 448] 
        out_conv1 = self.conv1(x) #[8, 32, 224, 224]
        out_conv2 = self.conv2(out_conv1)  #[8, 64, 112, 112]
        out_conv3 = self.conv3(out_conv2) #[8, 128, 56, 56]
        out_conv4 = self.conv4(out_conv3) #[8, 256, 28, 28]
        out_conv5 = self.conv5(out_conv4) #[8, 512, 14, 14]
        out_conv6 = self.conv6(out_conv5) #[8, 512, 7, 7]
        out_conv7 = self.conv7(out_conv6) #[8, 512, 4, 4]

        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        concat7 = torch.cat((out_upconv7, out_conv6), 1) #[8, 1024, 7, 7]
        out_iconv7 = self.iconv7(concat7) #[8, 512, 7, 7]

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1) #[8, 1024, 14, 14]
        out_iconv6 = self.iconv6(concat6)  #[8, 512, 14, 14]

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1) #8, 512, 28, 28]
        out_iconv5 = self.iconv5(concat5) #[8, 256, 28, 28]

        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1) #[8, 256, 56, 56]
        out_iconv4 = self.iconv4(concat4) #[8, 128, 56, 56] *
        if self.lpg_flag:
            disp4, disp4_scaled, disp4_scaled_ds = LPG_func(out_iconv4, self.reduc8x8, self.lpg8x8, self.max_depth, 1/8)
        else:
            disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta #[8, 1, 56, 56]
            disp4_scaled_ds = disp4
        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(F.interpolate(disp4_scaled_ds, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)  #[8, 64, 112, 112]
        if self.lpg_flag:
            disp3, disp3_scaled, disp3_scaled_ds= LPG_func(out_iconv3, self.reduc4x4, self.lpg4x4, self.max_depth, 1/4)
        else:
            disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta
            disp3_scaled_ds = disp3

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(F.interpolate(disp3_scaled_ds, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2) #[8, 32, 224, 224] *
        if self.lpg_flag:
            disp2, disp2_scaled, disp2_scaled_ds = LPG_func(out_iconv2, self.reduc2x2, self.lpg2x2, self.max_depth, 1/2)
        else:
            disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta #[8, 1, 224, 224]
            disp2_scaled_ds =disp2

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(F.interpolate(disp2_scaled_ds, scale_factor=2, mode='bilinear', align_corners=False), x)
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1) #[8, 16, 448, 448] **
        
        if self.lpg_flag:
            reduc1x1 = self.reduc1x1(out_iconv1) #[8, 1, 448, 448]
            concat1 = torch.cat([reduc1x1, disp2_scaled, disp3_scaled, disp4_scaled], dim=1) #[8, 4, 448, 448]
            iconv1_lpg = self.conv1_lpg(concat1)
            disp1 = self.max_depth * self.get_depth(iconv1_lpg)
        else:
            disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta
        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1