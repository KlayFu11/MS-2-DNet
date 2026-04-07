import torch
import math

import itertools
import torch.nn.functional as F

from torch.nn import init
from torch.nn.init import trunc_normal_
from torch.nn.modules.utils import _triple
import torch.nn as nn

from deepmist.models.multiframe.STDBNet import base_STD as base

from thop import profile, clever_format
import functools
from deepmist.models.multiframe.LS_STDNet.ska import SKA 
from deepmist.models.multiframe.LS_STDNet.ska_mulch import SKA_GroupAgg


#卷积层 (Convolution) 和 批归一化层 (Batch Normalization, BN) 串联在一起的组合模块
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        
        self.sks = sks
        self.groups = groups
        self.dim = dim
        
    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        #x = self.act(self.cv3(self.act(self.cv1(x))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)

        return w

class LKP1(nn.Module):
    def __init__(self, dim, G, lks=5, sks=3):
        super(LKP1, self).__init__()
        self.dim = dim
        self.G = G
        self.sks = sks

        mid = dim // 2

        self.cv1 = Conv2d_BN(dim, mid)
        self.act = nn.ReLU(inplace=True)
        self.cv2 = Conv2d_BN(mid, mid, ks=lks, pad=(lks-1)//2, groups=mid)
        self.cv3 = Conv2d_BN(mid, mid)

        # output channels now = G * (ks*ks)
        self.cv4 = nn.Conv2d(mid, G * (sks*sks), kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=G, num_channels=G * (sks*sks))

    def forward(self, x):
        x = self.act(self.cv1(x))
        x = self.act(self.cv2(x))
        x = self.act(self.cv3(x))

        w = self.cv4(x)
        B, _, H, W = w.shape
        w = self.norm(w)
        return w.view(B, self.G, self.sks*self.sks, H, W)

#LSConv, 采用SKA和LKP
class LSConv(nn.Module):
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=1, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        
        return self.bn(self.ska(x, self.lkp(x))) + x

class LSGConv(nn.Module):
    def __init__(self, in_channel, out_channel, G=8, lks=7, sks=1):
        super(LSGConv, self).__init__()

        assert in_channel % G == 0
        assert out_channel % G == 0

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.G = G

        self.lkp = LKP1(in_channel, G=G, lks=lks, sks=sks)

        self.ska = SKA_GroupAgg()
        self.bn = nn.BatchNorm2d(out_channel)

        if in_channel != out_channel:
            self.res = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
            self.res_bn = nn.BatchNorm2d(out_channel)
        else:
            self.res = None

    def forward(self, x):
        w = self.lkp(x)  # (B, G, ks*ks, H, W)
        o = self.ska(x, w, self.out_channel)
        o = self.bn(o)
        if self.res is not None:
            return o + self.res_bn(self.res(x))
        else:
            return o + x


#顺序经过，采用LSConvBlock
class LSConvBlock(nn.Module):
    def __init__(self, num_feat):
        super(LSConvBlock, self).__init__()

        in_ch = 2 * num_feat

        #采用LSConv KL=7 KS=3 Group=8
        self.ls = LSConv(in_ch)
        self.PWconv = nn.Conv2d(in_ch, num_feat, kernel_size=1,bias=False)
        self.act = nn.ReLU(inplace=False)



    def forward(self, past_feat, curr_feat):
        x = torch.cat([past_feat, curr_feat], dim=1)

        x=self.ls(x)
        x=self.PWconv(x)

        return x

class SimpleConv(nn.Module):
    def __init__(self, num_feat):
        super(SimpleConv, self).__init__()
        
        in_ch = 2 * num_feat
        
        # 7x7 深度可分离卷积
        self.dwconv7x7 = nn.Conv2d(
            in_ch, in_ch, 
            kernel_size=7, 
            padding=3, 
            bias=False
        )
        self.bn7 = nn.BatchNorm2d(in_ch)
        
        # 3x3 深度可分离卷积
        self.dwconv3x3 = nn.Conv2d(
            in_ch, in_ch, 
            kernel_size=3, 
            padding=1, 
            groups=in_ch, 
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(in_ch)
        
        # Pointwise卷积降维
        self.pwconv = nn.Conv2d(in_ch, num_feat, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(num_feat)
        
        self.act = nn.ReLU(inplace=False)
        
    def forward(self, past_feat, curr_feat):
        # 拼接past和current特征
        x = torch.cat([past_feat, curr_feat], dim=1)
        
        # 7x7 DWConv
        x = self.dwconv7x7(x)
        x = self.bn7(x)
        x = self.act(x)
        
        # 3x3 DWConv
        x = self.dwconv3x3(x)
        x = self.bn3(x)
        x = self.act(x)
        
        # Pointwise降维
        x = self.pwconv(x)
        x = self.bn_pw(x)
        
        return x


class SimpleGConv(nn.Module):
    """
    设计思路：
    1. 用 7x7 普通卷积 替代 LKP 的 7x7 大核卷积
    2. 用 3x3 深度可分离卷积 替代 SKA 的动态 3x3 卷积
    3. 支持 in_channel != out_channel 的通道变换
    """
    def __init__(self, in_channel, out_channel, lks=7, sks=3):
        super(SimpleGConv, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        # 7x7 普通卷积（替代LKP的7x7大核卷积）
        self.conv7x7 = nn.Conv2d(
            in_channel, in_channel,
            kernel_size=lks,
            padding=(lks - 1) // 2,
            bias=False
        )
        self.bn7 = nn.BatchNorm2d(in_channel)
        
        # 3x3 深度可分离卷积（替代SKA的动态3x3卷积）
        self.dwconv3x3 = nn.Conv2d(
            in_channel, in_channel,
            kernel_size=sks,
            padding=(sks - 1) // 2,
            groups=in_channel,  # 深度可分离
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(in_channel)
        
        # Pointwise卷积进行通道变换 (in_channel -> out_channel)
        self.pwconv = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channel)
        
        # 残差连接（当 in_channel != out_channel 时）
        if in_channel != out_channel:
            self.res = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
            self.res_bn = nn.BatchNorm2d(out_channel)
        
        self.act = nn.ReLU(inplace=False)
        
    def forward(self, x):
        # 保存原始输入用于残差连接
        residual = x
        
        # 7x7 普通卷积
        x = self.conv7x7(x)
        x = self.bn7(x)
        x = self.act(x)
        
        # 3x3 深度可分离卷积
        x = self.dwconv3x3(x)
        x = self.bn3(x)
        x = self.act(x)
        
        # Pointwise通道变换
        x = self.pwconv(x)
        x = self.bn_pw(x)
        
        # 残差连接（使用保存的原始输入）
        if self.in_channel != self.out_channel:
            x = x + self.res_bn(self.res(residual))
        # 如果 in_channel == out_channel，直接加残差
        else:
            x = x + residual
        
        return x


#无动态参数
class StaticConvBlock(nn.Module):
    """
    1. 用 7x7 深度可分离卷积 替代 LKP 的 7x7 大核卷积
    2. 用 3x3 深度可分离卷积 替代 SKA 的动态 3x3 卷积
    3. 保持相同的通道变换和残差连接
    """
    def __init__(self, num_feat):
        super(StaticConvBlock, self).__init__()
        
        in_ch = 2 * num_feat
        
        # 对应 LKP 的处理：先降维到一半，7x7 DWConv，再保持
        self.cv1 = Conv2d_BN(in_ch, in_ch // 2)
        self.act1 = nn.ReLU(inplace=True)
        
        # 7x7 深度可分离卷积（替代LKP中的7x7 groups卷积）
        self.dwconv7x7 = nn.Conv2d(
            in_ch // 2, in_ch // 2, 
            kernel_size=7, 
            padding=3, 
            groups=in_ch // 2,  # 深度可分离
            bias=False
        )
        self.bn7 = nn.BatchNorm2d(in_ch // 2)
        self.act2 = nn.ReLU(inplace=True)
        
        # 对应 LKP 的 cv3
        self.cv3 = Conv2d_BN(in_ch // 2, in_ch // 2)
        self.act3 = nn.ReLU(inplace=True)
        
        # 对应 SKA 的 3x3 动态卷积 → 用静态 3x3 DWConv 替代
        self.dwconv3x3 = nn.Conv2d(
            in_ch, in_ch, 
            kernel_size=3, 
            padding=1,
            groups=in_ch,  # 深度可分离，对应groups=8的分组特性
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(in_ch)
        self.act4 = nn.ReLU(inplace=True)
        
        # 残差连接的BN
        self.res_bn = nn.BatchNorm2d(in_ch)
        
        # Pointwise卷积降维
        self.pwconv = nn.Conv2d(in_ch, num_feat, kernel_size=1, bias=False)
        
    def forward(self, past_feat, curr_feat):
        x = torch.cat([past_feat, curr_feat], dim=1)
        in_ch = x.size(1)
        
        # 分支1: 模拟 LKP 的路径
        branch = self.act1(self.cv1(x))
        branch = self.act2(self.bn7(self.dwconv7x7(branch)))
        branch = self.act3(self.cv3(branch))
        
        # 分支2: 模拟 SKA 的路径（这里简化处理）
        # SKA 原本是使用 LKP 生成的动态核，这里用静态卷积替代
        x = self.act4(self.bn3(self.dwconv3x3(x)))
        
        # 残差连接（对应 LSConv 中的 + x）
        x = self.res_bn(x) + x
        
        # Pointwise降维
        x = self.pwconv(x)
        
        return x


#差分相乘经过LSConv
# nn.GroupNorm，nn.ReLU都可以删掉，相加的跳阶也可以删掉试试
class LSConvDiff(nn.Module):
    def __init__(self, num_feat):
        super(LSConvDiff, self).__init__()
        self.ls = LSConv(num_feat)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=num_feat)
        
        self.PWconv = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.act = nn.ReLU(inplace=False)
    
    #这一版物理意义错误的差分效果还行
    # def forward(self, past_feat, curr_feat)

    #     # 差分
    #     diff = curr_feat - past_feat
    #     diff = self.norm(diff)

    #     # 动态门控
    #     dyn = self.ls(diff)                 # [B, C, H, W]

    #     # 稳定的 dyn_thr
    #     dyn_thr = torch.sigmoid(dyn)           # 限幅在[0,1]

    #     # 融合：避免信息消失（带残差）
    #     #fused = curr_feat * (dyn_thr + 1.0)    # gate∈[0,1] → [1,2]
    #     fused = curr_feat * dyn_thr 

    #     return self.act(fused)

    def forward(self, past_feat, curr_feat, same_frame=False):

        if same_frame:
            
            curr_norm =self.norm(curr_feat) 
            dyn = self.ls(curr_norm)
            dyn = self.PWconv(dyn)
            #dyn_thr = torch.sigmoid(dyn)
            fused = curr_feat * dyn
            fused_comb = fused + curr_feat
            return self.act(fused)

        else:
            #差分相乘
            diff = curr_feat - past_feat
            #加入归一化
            diff = self.norm(diff)

            dyn = self.ls(diff)
            dyn = self.PWconv(dyn)

            #加入稳定函数限幅度，防止梯度爆炸
            #dyn_thr = torch.sigmoid(dyn)
            #fused = (dyn_thr + 1.0) * curr_feat
            fused = dyn * past_feat

            fused_comb = fused + curr_feat
            return self.act(fused)


#多帧融合MFA
class MultiFrameAggregation(nn.Module):
    def __init__(self, block, nb_filter, num_frames, num_levels):
        super(MultiFrameAggregation, self).__init__()
        self.num_levels = num_levels
        self.aggregators = nn.ModuleList([
        #将num_frames个帧的特征图（每个帧特征图通道数为nb_filter[j]）拼接后的总通道数（nb_filter[j] * num_frames）
        #转换回单帧特征的通道数（nb_filter[j]
            base.make_layer(block, nb_filter[j] * num_frames, nb_filter[j])
            for j in range(num_levels)
        ])

    def forward(self, compensated_feats):
        return [self.aggregators[j](compensated_feats[j]) for j in range(self.num_levels)]


#采用Unet解码器 BaseDecoder
class BaseDecoder(nn.Module):
    def __init__(self, num_classes, block, nb_filter):
        super(BaseDecoder, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder_3 = base.make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.decoder_2 = base.make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.decoder_1 = base.make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])
        self.head = nn.Conv2d(nb_filter[0], num_classes, 1)

    def forward(self, observations):
        z_3 = self.decoder_3(torch.cat([observations[2], self.up(observations[3])], 1))
        z_2 = self.decoder_2(torch.cat([observations[1], self.up(z_3)], 1))
        z_1 = self.decoder_1(torch.cat([observations[0], self.up(z_2)], 1))
        pred_z_1 = self.head(z_1)

        return pred_z_1

#在U-Net BaseDecoder基础上将ResBlock换为CSHD
class CSBaseDecoder(nn.Module):
    def __init__(self, num_classes, nb_filter):
        super(CSBaseDecoder, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder_3 = LSGConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.decoder_2 = LSGConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.decoder_1 = LSGConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.head = nn.Conv2d(nb_filter[0], num_classes, 1)

    def forward(self, observations):
        z_3 = self.decoder_3(torch.cat([observations[2], self.up(observations[3])], 1))
        z_2 = self.decoder_2(torch.cat([observations[1], self.up(z_3)], 1))
        z_1 = self.decoder_1(torch.cat([observations[0], self.up(z_2)], 1))
        pred_z_1 = self.head(z_1)

        return pred_z_1

class PANBaseDecoder(nn.Module):
    def __init__(self, num_classes, nb_filter):
        super(PANBaseDecoder, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.down_1 =nn.Conv2d(nb_filter[0], nb_filter[0], 3, stride=2, padding=1, bias=False)
        # self.down_2 =nn.Conv2d(nb_filter[1], nb_filter[1], 3, stride=2, padding=1, bias=False)
        # self.down_3 =nn.Conv2d(nb_filter[2], nb_filter[2], 3, stride=2, padding=1, bias=False)


        self.downconv_1 = nn.Conv2d(nb_filter[0], nb_filter[0], 3, padding=1, bias=False)
        self.downconv_2 = nn.Conv2d(nb_filter[1], nb_filter[1], 3, padding=1, bias=False)
        self.downconv_3 = nn.Conv2d(nb_filter[2], nb_filter[2], 3, padding=1, bias=False)

        self.updecoder_3 = SimpleGConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.updecoder_2 = SimpleGConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.updecoder_1 = SimpleGConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        

        self.downdecoder_1 = SimpleGConv(nb_filter[1] + nb_filter[0], nb_filter[1])
        self.downdecoder_2 = SimpleGConv(nb_filter[2] + nb_filter[1], nb_filter[2])
        self.downdecoder_3 = SimpleGConv(nb_filter[3] + nb_filter[2], nb_filter[3])

        self.head_1 = nn.Conv2d(nb_filter[0], num_classes, 1)
        self.head_2 = nn.Conv2d(nb_filter[1], num_classes, 1)
        self.head_3 = nn.Conv2d(nb_filter[2], num_classes, 1)
        self.head_4 = nn.Conv2d(nb_filter[3], num_classes, 1)
    
    def forward(self, observations):    

        z_3 = self.updecoder_3(torch.cat([observations[2], self.up(observations[3])], 1))
        z_2 = self.updecoder_2(torch.cat([observations[1], self.up(z_3)], 1))
        z_1 = self.updecoder_1(torch.cat([observations[0], self.up(z_2)], 1))
        pred_z_1 = self.head_1(z_1)

        # #加入卷积下采样
        # down_z1 = F.interpolate(z_1, scale_factor=0.5, mode="bilinear", align_corners=True)
        # down_z1 = self.downconv_1(down_z1)   
        # d_2 = self.downdecoder_1(torch.cat([down_z1, z_2], 1))
        # pred_z_2 = self.head_2(d_2)

        # # scale 192 → 96
        # down_d2 = F.interpolate(d_2, scale_factor=0.5, mode="bilinear", align_corners=True)
        # down_d2 = self.downconv_2(down_d2)   
        # d_3 = self.downdecoder_2(torch.cat([down_d2, z_3], 1))
        # pred_z_3 = self.head_3(d_3)

        #下采样过程中不带卷积
        d_2 = self.downdecoder_1(torch.cat([F.interpolate(z_1, scale_factor=0.5, mode="bilinear", align_corners=True), z_2], 1))
        pred_z_2 = self.head_2(d_2)
        d_3 = self.downdecoder_2(torch.cat([F.interpolate(d_2, scale_factor=0.5, mode="bilinear", align_corners=True), z_3], 1))
        pred_z_3 = self.head_3(d_3)

        # scale 96 → 48
        # down_d3 = F.interpolate(d_3, scale_factor=0.5, mode="bilinear", align_corners=True)
        # down_d3 = self.downconv_3(down_d3)   
        # d_4 = self.downdecoder_3(torch.cat([down_d3, observations[3]], 1))
        # pred_z_4 = self.head_4(d_4)

        #采用三层
        # d_4 = self.downdecoder_3(torch.cat([F.interpolate(d_3, scale_factor=0.5, mode="bilinear", align_corners=True), observations[3]], 1))
        # pred_z_4 = self.head_4(d_4)

        return [pred_z_1, pred_z_2, pred_z_3]
        # return [pred_z_1, pred_z_2, pred_z_3, pred_z_4]

#将PANBaseDecoder中的Resblock换成LSConv
class PANLSDecoder(nn.Module):
    def __init__(self, num_classes, nb_filter):
        super(PANLSDecoder, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        
        self.updecoder_3 = LSConv(nb_filter[2] + nb_filter[3])
        self.PWupconv_3 = nn.Conv2d(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size=1,bias=False)
        
        self.updecoder_2 = LSConv(nb_filter[1] + nb_filter[2])
        self.PWupconv_2 = nn.Conv2d(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size=1,bias=False)
        
        self.updecoder_1 = LSConv(nb_filter[0] + nb_filter[1])
        self.PWupconv_1 = nn.Conv2d(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size=1,bias=False)
        
        self.downdecoder_1 = LSConv(nb_filter[1] + nb_filter[0])
        self.PWdownconv_1 = nn.Conv2d(nb_filter[1] + nb_filter[0], nb_filter[1], kernel_size=1,bias=False)
        self.downdecoder_2 = LSConv(nb_filter[2] + nb_filter[1])
        self.PWdownconv_2 = nn.Conv2d(nb_filter[2] + nb_filter[1], nb_filter[2], kernel_size=1,bias=False)

        self.head_1 = nn.Conv2d(nb_filter[0], num_classes, 1)
        self.head_2 = nn.Conv2d(nb_filter[1], num_classes, 1)
        self.head_3 = nn.Conv2d(nb_filter[2], num_classes, 1)

    def forward(self, observations):
        #上采样得到z_3, z_2, z_1
        z_3 = self.PWupconv_3(self.updecoder_3(torch.cat([observations[2], self.up(observations[3])], 1)))
        z_2 = self.PWupconv_2(self.updecoder_2(torch.cat([observations[1], self.up(z_3)], 1)))
        z_1 = self.PWupconv_1(self.updecoder_1(torch.cat([observations[0], self.up(z_2)], 1)))
        pred_z_1 = self.head_1(z_1)

        
        d_2 = self.PWdownconv_1(self.downdecoder_1(torch.cat([F.interpolate(z_1, scale_factor=0.5, mode="bilinear", align_corners=True), z_2], 1)))
        pred_z_2 = self.head_2(d_2)
        d_3 = self.PWdownconv_2(self.downdecoder_2(torch.cat([F.interpolate(d_2, scale_factor=0.5, mode="bilinear", align_corners=True), z_3], 1)))
        pred_z_3 = self.head_3(d_3)


        return [pred_z_1, pred_z_2, pred_z_3]

#不使用PWConv，直接在内部改变通道
class PANLSAGGDecoder(nn.Module):
    def __init__(self, num_classes,  nb_filter):
        super(PANLSAGGDecoder, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)#aligh_corners设置为TRUE

        
        self.updecoder_3 = LSGConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.updecoder_2 = LSGConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.updecoder_1 = LSGConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        # self.updecoder_3 = base.make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        # self.updecoder_2 = base.make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        # self.updecoder_1 = base.make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])
        
        self.downdecoder_1 = LSGConv(nb_filter[1] + nb_filter[0], nb_filter[1])
        self.downdecoder_2 = LSGConv(nb_filter[2] + nb_filter[1], nb_filter[2])
        # self.downdecoder_1 = base.make_layer(block, nb_filter[1] + nb_filter[0], nb_filter[1])
        # self.downdecoder_2 = base.make_layer(block, nb_filter[2] + nb_filter[1], nb_filter[2])
        

        self.head_1 = nn.Conv2d(nb_filter[0], num_classes, 1, bias=True)
        self.head_2 = nn.Conv2d(nb_filter[1], num_classes, 1, bias=True)
        self.head_3 = nn.Conv2d(nb_filter[2], num_classes, 1, bias=True)

    def forward(self, observations):
        #上采样得到z_3, z_2, z_1
        z_3 = self.updecoder_3(torch.cat([observations[2], self.up(observations[3])], 1))
        z_2 = self.updecoder_2(torch.cat([observations[1], self.up(z_3)], 1))
        z_1 = self.updecoder_1(torch.cat([observations[0], self.up(z_2)], 1))
        pred_z_1 = self.head_1(z_1)

        
        d_2 = self.downdecoder_1(torch.cat([F.interpolate(z_1, scale_factor=0.5, mode="area"), z_2], 1))
        pred_z_2 = self.head_2(d_2)
        d_3 = self.downdecoder_2(torch.cat([F.interpolate(d_2, scale_factor=0.5, mode="area"), z_3], 1))
        pred_z_3 = self.head_3(d_3)


        return [pred_z_1, pred_z_2, pred_z_3]


#nb_filter定义四个层级的特征通道数
class LS_STDNet(nn.Module):
    def __init__(self, num_frames=5, num_classes=1, in_channels=3, block=base.ResBlock, num_blocks=[2, 2, 2],
                 nb_filter=[8, 16, 32, 64],  modulator=base.CBAM, use_sufficiency_loss=True, deep_supervision=False):
        super(LS_STDNet, self).__init__()
        self.num_levels = len(nb_filter)    #四个层级
        self.num_frames = num_frames        #五帧一组
        self.use_sufficiency_loss = use_sufficiency_loss
        self.deep_supervision = deep_supervision

        self.encoder = base.ResNet(in_channels, block, num_blocks, nb_filter)
        # self.residual_layer = nn.ModuleList([
        #     self.make_layer(functools.partial(ResBlock_3d, nb_filter[j]), 1)
        #     for j in range(self.num_levels)
        # ])
        
        # self.IMC = ImplicitMotionCompensation(block, nb_filter, self.num_levels, shift_sizes, neighborhood_sizes)
        self.LSC = nn.ModuleList([
            LSConvBlock(num_feat=nb_filter[j]) for j in range(self.num_levels)
        ])
        self.MFA = MultiFrameAggregation(block, nb_filter, num_frames, self.num_levels)
        # self.decoder = ProgressiveDistillationDecoder(num_classes, block, nb_filter, modulator)
        self.decoder = PANLSAGGDecoder(num_classes, nb_filter)
        # self.decoder = PANBaseDecoder(num_classes, nb_filter)
        # self.decoder = CSBaseDecoder(num_classes, nb_filter)
        # self.decoder = BaseDecoder(num_classes, block, nb_filter)


    #特征均为4维张量
    def forward(self, x):
        #取时间维度上的最后一帧，得到[1，3,384,384]，然后进行ResNet编码，得到当前帧的四个特征
        #curr_frame_feats[0,1,2,3]=（[1, 8, 384, 384],[1, 16, 192, 192],[1, 32, 96, 96],[1, 64, 48, 48]）encode输出四个元素的元组
        # curr_frame_feats = self.encoder(x[:, :, -1, :, :])
        #循环四次，每次创建一个新的空列表，即创建一个包含4个空列表的外层列表[[], [], [], []]
        all_level_feats = [[] for _ in range(self.num_levels)]

        for i in range(self.num_frames):#0,1,2,3,4
            frame_feats = self.encoder(x[:, :, i, :, :])
            for j in range(self.num_levels):#0,1,2,3
                #4个层级每个层级分别有五帧同样大小的特征
                #即在[[], [], [], []]中按x轴添加，即[[1], [2], [3], [4]]，[[1，5], [2，6], [3，7], [4，8]]······
                all_level_feats[j].append(frame_feats[j])


        all_compensated_feats = [[] for _ in range(self.num_levels)]

        for i in range(self.num_levels):
            for j in range(self.num_frames):  
                all_compensated_feats[i].append(self.LSC[i](all_level_feats[i][j], all_level_feats[i][-1]))

        #仅差分的时候使用该形式
        # for i in range(self.num_levels):
        #     for j in range(self.num_frames):
        #         if j == self.num_frames - 1:
        #             all_compensated_feats[i].append(self.LSC[i](all_level_feats[i][j], all_level_feats[i][-1], same_frame=True))
        #         else:
        #             all_compensated_feats[i].append(self.LSC[i](all_level_feats[i][j], all_level_feats[i][-1], same_frame=False))
        #每个层级输出的observations和最初的对应层级的特征尺寸大小相同

        #observation[0,1,2,3] = [[1, 8, 384, 384],[1, 16, 192, 192],[1, 32, 96, 96],[1, 64, 48, 48]]
        observations = self.MFA([torch.cat(feats, dim=1) for feats in all_compensated_feats])

        # Progressive Distillation Decoder
        # ProgressiveDistillationDecoder(1, ResNet, [8,16,32,64], CBAM)
        # pred, pred_z_list, pred_v_list = self.decoder(observations)

        # if self.use_sufficiency_loss:
        #     return pred, pred_z_list, pred_v_list
        # return pred_z_list if self.deep_supervision else pred

        # Base Decoder
        pred = self.decoder(observations)

        return pred

if __name__ == '__main__':
    model = LS_STDNet().cuda()
    inputs = torch.randn((1, 3, 5, 384, 384)).cuda()  # Params = 0.57M FLOPs = 10.14G
    flops, params = profile(model, (inputs,))
    print('Params = ' + str(round(params / 1000 ** 2, 2)) + 'M')
    print('FLOPs = ' + str(round(flops / 1000 ** 3, 2)) + 'G')
    flops, params = clever_format([flops, params], '%.6f')
    print('Params = ' + params)
    print('FLOPs = ' + flops)