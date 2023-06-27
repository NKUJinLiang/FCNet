import torch
import torch.nn as nn
import torch.nn.functional as F
from guided_filter_pytorch.guided_filter import FastGuidedFilter, GuidedFilter
from LaplacianPyramid import Lap_Pyramid
import time
import os
import math
import numpy as np
import cv2 as cv
import torchvision
from torch.nn import init
EPS = 1e-8

class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.in_norm = nn.InstanceNorm2d(n, affine=True, track_running_stats=False)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.in_norm(x)

class Block(nn.Module):
    """ implement conv+ReLU two times """
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        conv_relu = []
        conv_relu.append(nn.Conv2d(in_channels=in_channels, out_channels=middle_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.LeakyReLU())
        conv_relu.append(nn.Conv2d(in_channels=middle_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.LeakyReLU())
        self.conv_ReLU = nn.Sequential(*conv_relu)
    def forward(self, x):
        out = self.conv_ReLU(x)
        return out

class deconv_Block(nn.Module):
    """ implement conv+ReLU two times """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1)


    def forward(self, x,y):
        x = F.interpolate(x, (y.size()[2], y.size()[3]),mode='bilinear')
        out = self.conv(x)
        return out

#四层U_Net
class U_Net_4(nn.Module):
    def __init__(self,initialchanel):
        super().__init__()

        # 首先定义左半部分网络
        # left_conv_1 表示连续的两个（卷积+激活）
        # 随后进行最大池化
        self.ReLU = nn.LeakyReLU()
        self.left_conv_1 = Block(in_channels=3, middle_channels=initialchanel, out_channels=initialchanel)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_2 = Block(in_channels=initialchanel, middle_channels=2*initialchanel, out_channels=2*initialchanel)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_3 = Block(in_channels=2*initialchanel, middle_channels=4*initialchanel, out_channels=4*initialchanel)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_4 = Block(in_channels=4*initialchanel, middle_channels=8*initialchanel, out_channels=8*initialchanel)

        # 定义右半部分网络
        self.deconv_1 = deconv_Block(in_channels=8*initialchanel, out_channels=4*initialchanel)
        self.right_conv_1 = Block(in_channels=8*initialchanel, middle_channels=4*initialchanel, out_channels=4*initialchanel)

        self.deconv_2 = deconv_Block(in_channels=4*initialchanel, out_channels=2*initialchanel)
        self.right_conv_2 = Block(in_channels=4*initialchanel, middle_channels=2*initialchanel, out_channels=2*initialchanel)

        self.deconv_3 = deconv_Block(in_channels=2*initialchanel, out_channels=initialchanel)
        self.right_conv_3 = Block(in_channels=2*initialchanel, middle_channels=initialchanel, out_channels=initialchanel)
        # 最后是1x1的卷积，用于将通道数化为3
        self.right_conv_4 = nn.Conv2d(in_channels=initialchanel, out_channels=6, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        # 1：进行编码过程
        feature_1 = self.left_conv_1(x)
        #print("feature_1",feature_1.shape)
        feature_1_pool = self.pool_1(feature_1)
        #print("feature_1_pool",feature_1_pool.shape)

        feature_2 = self.left_conv_2(feature_1_pool)
        #print("feature_2",feature_2.shape)
        feature_2_pool = self.pool_2(feature_2)
        #print("feature_2_pool",feature_2_pool.shape)

        feature_3 = self.left_conv_3(feature_2_pool)
        #print("feature_3",feature_3.shape)
        feature_3_pool = self.pool_3(feature_3)
        #print("feature_3_pool",feature_3_pool.shape)

        feature_4 = self.left_conv_4(feature_3_pool)
        #print("feature_4",feature_4.shape)

        # 2：进行解码过程
        de_feature_1 = self.deconv_1(feature_4,feature_3)
        #print("de_feature_1",de_feature_1.shape)
        de_feature_1 = self.ReLU(de_feature_1)
        #print("de_feature_1",de_feature_1.shape)

        # 特征拼接
        temp = torch.cat((feature_3, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp)

        de_feature_2 = self.deconv_2(de_feature_1_conv,feature_2)
        de_feature_2 = self.ReLU(de_feature_2)
        temp = torch.cat((feature_2, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp)

        de_feature_3 = self.deconv_3(de_feature_2_conv,feature_1)
        de_feature_3 = self.ReLU(de_feature_3)
        temp = torch.cat((feature_1, de_feature_3), dim=1)
        de_feature_3_conv = self.right_conv_3(temp)

        out = self.right_conv_4(de_feature_3_conv )

        image = out[:,0:3,:,:]
        w_1 = out[:,3:6,:,:]

        map_1 = F.softmax(w_1, dim=0)
        out = torch.sum(map_1 * image, dim=0, keepdim=True).clamp(0, 1)

        return out

#三层U_Net
class U_Net_3(nn.Module):
    def __init__(self,initialchanel):
        super().__init__()
        self.ReLU = nn.LeakyReLU()

        # 首先定义左半部分网络
        # left_conv_1 表示连续的两个（卷积+激活）
        # 随后进行最大池化
        self.left_conv_1 = Block(in_channels=3, middle_channels=initialchanel, out_channels=initialchanel)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_2 = Block(in_channels=initialchanel, middle_channels=2*initialchanel, out_channels=2*initialchanel)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_3 = Block(in_channels=2*initialchanel, middle_channels=4*initialchanel, out_channels=4*initialchanel)

        # 定义右半部分网络
        self.deconv_1 = deconv_Block(in_channels=4*initialchanel, out_channels=2*initialchanel)
        self.right_conv_1 = Block(in_channels=4*initialchanel, middle_channels=2*initialchanel, out_channels=2*initialchanel)

        self.deconv_2 = deconv_Block(in_channels=2*initialchanel, out_channels=initialchanel)
        self.right_conv_2 = Block(in_channels=2*initialchanel, middle_channels=initialchanel, out_channels=initialchanel)
        # 最后是1x1的卷积，用于将通道数化为3
        self.right_conv_3 = nn.Conv2d(in_channels=initialchanel, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        # 1：进行编码过程
        feature_1 = self.left_conv_1(x)
        feature_1_pool = self.pool_1(feature_1)

        feature_2 = self.left_conv_2(feature_1_pool)
        feature_2_pool = self.pool_2(feature_2)

        feature_3 = self.left_conv_3(feature_2_pool)

        # 2：进行解码过程
        de_feature_1 = self.deconv_1(feature_3,feature_2)
        de_feature_1 = self.ReLU(de_feature_1)
        # 特征拼接
        temp = torch.cat((feature_2, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp)

        de_feature_2 = self.deconv_2(de_feature_1_conv,feature_1)
        de_feature_2 = self.ReLU(de_feature_2)
        temp = torch.cat((feature_1, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp)

        out = self.right_conv_3(de_feature_2_conv)

        return out

def build_lr_net(norm=AdaptiveNorm, layer=2, width=24):#lr = low resolution
    layers = [
        nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(width),
        nn.LeakyReLU(0.2, inplace=True),
    ]

    for l in range(1,layer):
        layers += [nn.Conv2d(width,  width, kernel_size=3, stride=1, padding=2**l,  dilation=2**l,  bias=False),
                   norm(width),
                   nn.LeakyReLU(0.2, inplace=True)]

    layers += [
        nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(width),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(width,  3, kernel_size=1, stride=1, padding=0, dilation=1)
    ]

    net = nn.Sequential(*layers)

    return net

class E2EMEF(nn.Module):
    # end-to-end mef model
    def __init__(self, depth,radius=1, eps=1e-4, is_guided=True):
        super(E2EMEF, self).__init__()
        self.lr = build_lr_net(layer=depth)
        self.is_guided = is_guided
        if is_guided:
            self.gf = FastGuidedFilter(radius, eps)

    def forward(self, x_lr, x_hr):
        w_lr = self.lr(x_lr)

        if self.is_guided:
            w_hr = self.gf(x_lr, w_lr, x_hr)
        else:
            w_hr = F.upsample(w_lr, x_hr.size()[2:], mode='bilinear')

        w_hr = torch.abs(w_hr)
        w_hr = (w_hr + EPS) / torch.sum((w_hr + EPS), dim=0)

        o_hr = torch.sum(w_hr * x_hr, dim=0, keepdim=True).clamp(0, 1)

        return o_hr, w_hr

#Multi_Scale_model
class Fusion_Model(nn.Module):
    """ implement conv+ReLU two times """
    def __init__(self, num_high):
        super().__init__()
        self.F_net_1 = E2EMEF(depth=4)
        self.F_net_2 = E2EMEF(depth = 3)
        self.F_net_3 =E2EMEF(depth = 2)
        self.F_net_4 =E2EMEF(depth = 1)
        self.C_net_1 = U_Net_4(24)#channel 24
        self.C_net_2 = U_Net_4(16)#channel 16
        self.C_net_3 = U_Net_3(16)#channel 16
        self.C_net_4 = U_Net_3(16)#channel 16
        self.decov = deconv_Block(in_channels=3, out_channels=3)

        self.lp = Lap_Pyramid(num_high)
        self.num_high = num_high

    def forward(self, img):
        #lp decomposition
        list = self.lp.pyramid_decom(img=img)
        #fusion on the 4th layer
        image_lr = nn.functional.interpolate(list[3], size=(list[3].shape[2] // 2, list[3].shape[3] // 2),
                                         mode='bilinear', align_corners=True)
        F_1,_ = self.F_net_1(image_lr, list[3])
        #correction
        out_1 = self.C_net_1(F_1)
        #upsampling
        U_1 = self.decov(out_1,list[2])
        U_1 = list[2] + U_1
        # fusion on the 3th layer
        image_lr = nn.functional.interpolate(U_1, size=(U_1.shape[2] // 2, U_1.shape[3] // 2),
                                         mode='bilinear', align_corners=True)
        F_2, _ =  self.F_net_2(image_lr, U_1)
        # correction
        out_2 = self.C_net_2(F_2)
        # upsampling
        U_2 = self.decov(out_2, list[1])
        U_2 = list[1] + U_2
        # fusion on the 2th layer
        image_lr = nn.functional.interpolate(U_2, size=(U_2.shape[2] // 2, U_2.shape[3] // 2),
                                         mode='bilinear', align_corners=True)
        F_3, _ = self.F_net_3(image_lr, U_2)
        # correction
        out_3 = self.C_net_3(F_3)
        # upsampling
        U_3 = self.decov(out_3, list[0])
        U_3 = list[0] + U_3
        # fusion on the 1th layer
        image_lr = nn.functional.interpolate(U_3, size=(U_3.shape[2] // 2, U_3.shape[3] // 2),
                                         mode='bilinear', align_corners=True)
        F_4, _ = self.F_net_4(image_lr, U_3)
        # correction
        out_4 = self.C_net_4(F_4)

        return out_1, out_2, out_3, out_4

'''
if __name__ == "__main__":
    #x_0 = torch.rand(size=(2,3,3))
    x_1 = torch.rand(size=(3, 3, 64, 64))
    x_2 = torch.rand(size=(3, 3, 128, 128))
    x_3 = torch.rand(size=(3, 3, 256, 256))
    x_4 = torch.rand(size=(3, 3, 512, 512))
    #out = x_0[0:3,:,:]
    net = PEC_model()
    out = net(x_1, x_2, x_3, x_4)
    for i in range(4):
        print(out[i].size())

    print(x_0)
    a = F.softmax(x_0, dim=0)
    print(a)
    print(torch.sum(a, dim=0, keepdim=True).shape)
    '''
if __name__ == "__main__":
    #x_0 = torch.rand(size=(2,3,3))
    x_1 = torch.rand(size=(7, 3, 75, 113)).cuda()
    x_2 = torch.rand(size=(7, 3, 150, 226)).cuda()
    x_3 = torch.rand(size=(7, 3, 300, 452)).cuda()
    x_4 = torch.rand(size=(7, 3, 600, 903)).cuda()
    x_5 = torch.rand(size=(7, 3, 523, 173)).cuda()

    net2 = Fusion_Model(num_high=3).cuda()
    start = time.time()
    #flops, params = thop.profile(net2,inputs=(x_5,))
    end_time = (time.time() - start)
    print("flops","params",flops,params)
    out1 , out2, out3, out4 = net2(x_5)
    print(end_time)
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
    print(out4.shape)
