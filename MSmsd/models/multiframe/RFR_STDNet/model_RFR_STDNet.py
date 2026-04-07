import torch
import math

import torch.nn.functional as F

from torch.nn import init
from torch.nn.init import trunc_normal_
from torch.nn.modules.utils import _triple
import torch.nn as nn

from deepmist.models.multiframe.RFR_STDNet import base_STD as base

from thop import profile, clever_format
import functools

# from deepmist.models.multiframe.D3Dnet.code.dcn.functions.deform_conv_func import DeformConvFunction
# from deepmist.models.multiframe.D3Dnet.code.dcn.modules.deform_conv import *
# from .D3D import DeformConvFunction, DeformConvPack_d
#from deepmist.models.multiframe.RFR_STDNet import Deform_part as deform
#from deepmist.models.multiframe.RFR_STDNet.utils import *




#PDA模块，多尺度特征对齐与隐式运动补偿
#输入：两个 (1, 16, 384, 384) 的特征图
# 输出：一个 (1, 16, 384, 384) 的对齐特征图

# class PDA(nn.Module):
#     """Alignment module using Pyramid, Cascading and Deformable convolution
#     (PCD). It is used in EDVR.

#     Ref:
#         EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

#     Args:
#         num_feat (int): Channel number of middle features. Default: 64.
#         deformable_groups (int): Deformable groups. Defaults: 8.
#     """

#     def __init__(self, num_feat=64, deformable_groups=4):
#         super(PDA, self).__init__()

#         # Pyramid has three levels:
#         # L3: level 3, 1/4 spatial size
#         # L2: level 2, 1/2 spatial size
#         # L1: level 1, original spatial size

#         self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)  # 步长为2，将特征图下采样至1/2大小
#         self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 步长为1，保持特征图大小不变
#         self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)  # 再次下采样至1/4大小
#         self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 保持特征图大小不变

#         self.offset_conv1 = nn.ModuleDict()  # 第一级偏移量卷积 创建空字典
#         self.offset_conv2 = nn.ModuleDict()  # 第二级偏移量卷积
#         self.offset_conv3 = nn.ModuleDict()  # 第三级偏移量卷积
#         self.dcn_pack = nn.ModuleDict()      # 可变形卷积包
#         self.feat_conv = nn.ModuleDict()     # 特征融合卷积
#         # Pyramids
#         for i in range(3, 0, -1):       #3, 2, 1
#             level = f'l{i}'             ## 生成'l3', 'l2', 'l1'
#             #为每一层创建一个用于计算偏移量的卷积层offset_conv1
#             self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
#             if i == 3:
#                 self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             else:
#                 #仅为第一和第二层创建offset_conv3 
#                 self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
#                 self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             #为每一层都创建一个可变形卷积模块2D，输入输出通道数均不变
#             self.dcn_pack[level] = \
#                 deform.DeformConv(num_feat,
#                            num_feat,
#                            3,
#                            padding=1,
#                            stride=1,
#                            deformable_groups=deformable_groups)

#             #仅为第一和第二层创建feat_conv特征融合卷积
#             if i < 3:
#                 self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

#         self.upsample = nn.Upsample(
#             scale_factor=2, mode='bilinear', align_corners=False)
#         self.lrelu = nn.ReLU(inplace=True)

#     def forward(self, nbr_feat_l, ref_feat_l, video_dir=None):
#         """Align neighboring frame features to the reference frame features.

#         Args:
#             nbr_feat_l (list[Tensor]): Neighboring feature list. It
#                 contains three pyramid levels (L1, L2, L3),
#                 each with shape (b, c, h, w).
#             ref_feat_l (list[Tensor]): Reference feature list. It
#                 contains three pyramid levels (L1, L2, L3),
#                 each with shape (b, c, h, w).

#         Returns:
#             Tensor: Aligned features.
#         """
#         #输入[1,64,384,384]
#         nbr_feat_l1 = nbr_feat_l
#         nbr_feat_l2 = self.lrelu(self.conv_l2_1(nbr_feat_l1))#[1, 64, 192, 192]
#         nbr_feat_l2 = self.lrelu(self.conv_l2_2(nbr_feat_l2))#[1, 64, 192, 192]
#         # L3
#         nbr_feat_l3 = self.lrelu(self.conv_l3_1(nbr_feat_l2))#[1, 64, 96, 96]
#         nbr_feat_l3 = self.lrelu(self.conv_l3_2(nbr_feat_l3))#[1, 64, 96, 96]

#         nbr_feat_l = [nbr_feat_l1, nbr_feat_l2, nbr_feat_l3]

          #ref_feat尺寸变化同上
#         ref_feat_l1 = ref_feat_l
#         ref_feat_l2 = self.lrelu(self.conv_l2_1(ref_feat_l1))
#         ref_feat_l2 = self.lrelu(self.conv_l2_2(ref_feat_l2))
#         # L3
#         ref_feat_l3 = self.lrelu(self.conv_l3_1(ref_feat_l2))
#         ref_feat_l3 = self.lrelu(self.conv_l3_2(ref_feat_l3))

#         ref_feat_l = [ref_feat_l1, ref_feat_l2, ref_feat_l3]
#         #取ref_feat_l[i], i = 0,1,2即取ref_feat_l1, ref_feat_l2, ref_feat_l3
#         # Pyramids
#         upsampled_offset, upsampled_feat = None, None
#         for i in range(3, 0, -1):#3， 2， 1
#             level = f'l{i}'
#             offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
#             offset = self.lrelu(self.offset_conv1[level](offset))
#             if i == 3:
#                 offset = self.lrelu(self.offset_conv2[level](offset))
#             else:
#                 offset = self.lrelu(self.offset_conv2[level](torch.cat(
#                     [offset, upsampled_offset], dim=1)))
#                 offset = self.lrelu(self.offset_conv3[level](offset))

#             feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)

#             if i < 3:
#                 feat = self.feat_conv[level](
#                     torch.cat([feat, upsampled_feat], dim=1))
#             if i > 1:
#                 feat = self.lrelu(feat)

#             if i > 1:  # upsample offset and features
#                 # x2: when we upsample the offset, we should also enlarge
#                 # the magnitude.
#                 upsampled_offset = self.upsample(offset) * 2
#                 upsampled_feat = self.upsample(feat)
#         return feat


#PDA金字塔层数为2
# class PDA(nn.Module):
#     """Alignment module using Pyramid, Cascading and Deformable convolution
#     (PCD). It is used in EDVR.

#     Ref:
#         EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

#     Args:
#         num_feat (int): Channel number of middle features. Default: 64.
#         deformable_groups (int): Deformable groups. Defaults: 8.
#     """

#     def __init__(self, num_feat=64, deformable_groups=4):
#         super(PDA, self).__init__()

#         # Pyramid has three levels:
#         # L3: level 3, 1/4 spatial size
#         # L2: level 2, 1/2 spatial size
#         # L1: level 1, original spatial size

#         self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)  # 步长为2，将特征图下采样至1/2大小
#         self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 步长为1，保持特征图大小不变
#         # self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)  # 再次下采样至1/4大小
#         # self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 保持特征图大小不变

#         self.offset_conv1 = nn.ModuleDict()  # 第一级偏移量卷积 创建空字典
#         self.offset_conv2 = nn.ModuleDict()  # 第二级偏移量卷积
#         self.offset_conv3 = nn.ModuleDict()  # 第三级偏移量卷积
#         self.dcn_pack = nn.ModuleDict()      # 可变形卷积包
#         self.feat_conv = nn.ModuleDict()     # 特征融合卷积
#         # Pyramids
#         for i in range(2, 0, -1):       # 2, 1
#             level = f'l{i}'             ## 生成'l3', 'l2', 'l1'
#             #为每一层创建一个用于计算偏移量的卷积层offset_conv1
#             self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
#             if i == 2:
#                 self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             else:
#                 #仅为第一层创建offset_conv3 
#                 self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
#                 self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             #为每一层都创建一个可变形卷积模块2D，输入输出通道数均不变
#             self.dcn_pack[level] = \
#                 deform.DeformConv(num_feat,
#                            num_feat,
#                            3,
#                            padding=1,
#                            stride=1,
#                            deformable_groups=deformable_groups)

#             #仅为第一和第二层创建feat_conv特征融合卷积
#             if i < 2:
#                 self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

#         self.upsample = nn.Upsample(
#             scale_factor=2, mode='bilinear', align_corners=False)
#         self.lrelu = nn.ReLU(inplace=True)

#     def forward(self, nbr_feat_l, ref_feat_l, video_dir=None):
#         """Align neighboring frame features to the reference frame features.

#         Args:
#             nbr_feat_l (list[Tensor]): Neighboring feature list. It
#                 contains three pyramid levels (L1, L2, L3),
#                 each with shape (b, c, h, w).
#             ref_feat_l (list[Tensor]): Reference feature list. It
#                 contains three pyramid levels (L1, L2, L3),
#                 each with shape (b, c, h, w).

#         Returns:
#             Tensor: Aligned features.
#         """
#         nbr_feat_l1 = nbr_feat_l
#         nbr_feat_l2 = self.lrelu(self.conv_l2_1(nbr_feat_l1))
#         nbr_feat_l2 = self.lrelu(self.conv_l2_2(nbr_feat_l2))
#         # # L3
#         # nbr_feat_l3 = self.lrelu(self.conv_l3_1(nbr_feat_l2))
#         # nbr_feat_l3 = self.lrelu(self.conv_l3_2(nbr_feat_l3))

#         nbr_feat_l = [nbr_feat_l1, nbr_feat_l2]

#         ref_feat_l1 = ref_feat_l
#         ref_feat_l2 = self.lrelu(self.conv_l2_1(ref_feat_l1))
#         ref_feat_l2 = self.lrelu(self.conv_l2_2(ref_feat_l2))
#         # L3
#         # ref_feat_l3 = self.lrelu(self.conv_l3_1(ref_feat_l2))
#         # ref_feat_l3 = self.lrelu(self.conv_l3_2(ref_feat_l3))

#         ref_feat_l = [ref_feat_l1, ref_feat_l2]
#         #取ref_feat_l[i], i = 0,1,2即取ref_feat_l1, ref_feat_l2, ref_feat_l3
#         # Pyramids
#         upsampled_offset, upsampled_feat = None, None
#         for i in range(2, 0, -1):#2， 1
#             level = f'l{i}'
#             offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
#             offset = self.lrelu(self.offset_conv1[level](offset))
#             if i == 2:
#                 offset = self.lrelu(self.offset_conv2[level](offset))
#             else:
#                 offset = self.lrelu(self.offset_conv2[level](torch.cat(
#                     [offset, upsampled_offset], dim=1)))
#                 offset = self.lrelu(self.offset_conv3[level](offset))

#             feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)

#             if i < 2:
#                 feat = self.feat_conv[level](
#                     torch.cat([feat, upsampled_feat], dim=1))
#             if i > 1:
#                 feat = self.lrelu(feat)

#             if i > 1:  # upsample offset and features
#                 # x2: when we upsample the offset, we should also enlarge
#                 # the magnitude.
#                 upsampled_offset = self.upsample(offset) * 2
#                 upsampled_feat = self.upsample(feat)
#         return feat

#将DCN替换为普通卷积
# class PDA(nn.Module):

#     def __init__(self, num_feat=64, deformable_groups=4):
#         super(PDA, self).__init__()

#         # Pyramid has three levels:
#         # L3: level 3, 1/4 spatial size
#         # L2: level 2, 1/2 spatial size
#         # L1: level 1, original spatial size

#         self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)  # 步长为2，将特征图下采样至1/2大小
#         self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 步长为1，保持特征图大小不变
#         # self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)  # 再次下采样至1/4大小
#         # self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 保持特征图大小不变

#         self.offset_conv1 = nn.ModuleDict()  # 第一级偏移量卷积 创建空字典
#         self.offset_conv2 = nn.ModuleDict()  # 第二级偏移量卷积
#         self.offset_conv3 = nn.ModuleDict()  # 第三级偏移量卷积
#         self.dcn_pack = nn.ModuleDict()      # 存放普通卷积
#         self.feat_conv = nn.ModuleDict()     # 特征融合卷积
#         # Pyramids
#         for i in range(2, 0, -1):       # 2, 1
#             level = f'l{i}'             ## 生成'l3', 'l2', 'l1'
#             #为每一层创建一个用于计算偏移量的卷积层offset_conv1
#             self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
#             if i == 2:
#                 self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             else:
#                 #仅为第一层创建offset_conv3 
#                 self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
#                 self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             #为每一层都创建一个可变形卷积模块2D，输入输出通道数均不变
#             self.dcn_pack[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

#             #仅为第一和第二层创建feat_conv特征融合卷积
#             if i < 2:
#                 self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

#         self.upsample = nn.Upsample(
#             scale_factor=2, mode='bilinear', align_corners=False)
#         self.lrelu = nn.ReLU(inplace=True)

#     def forward(self, nbr_feat_l, ref_feat_l, video_dir=None):

#         nbr_feat_l1 = nbr_feat_l
#         nbr_feat_l2 = self.lrelu(self.conv_l2_1(nbr_feat_l1))
#         nbr_feat_l2 = self.lrelu(self.conv_l2_2(nbr_feat_l2))
#         # # L3
#         # nbr_feat_l3 = self.lrelu(self.conv_l3_1(nbr_feat_l2))
#         # nbr_feat_l3 = self.lrelu(self.conv_l3_2(nbr_feat_l3))

#         nbr_feat_l = [nbr_feat_l1, nbr_feat_l2]

#         ref_feat_l1 = ref_feat_l
#         ref_feat_l2 = self.lrelu(self.conv_l2_1(ref_feat_l1))
#         ref_feat_l2 = self.lrelu(self.conv_l2_2(ref_feat_l2))
#         # L3
#         # ref_feat_l3 = self.lrelu(self.conv_l3_1(ref_feat_l2))
#         # ref_feat_l3 = self.lrelu(self.conv_l3_2(ref_feat_l3))

#         ref_feat_l = [ref_feat_l1, ref_feat_l2]
#         #取ref_feat_l[i], i = 0,1,2即取ref_feat_l1, ref_feat_l2, ref_feat_l3
#         # Pyramids
#         upsampled_offset, upsampled_feat = None, None
#         for i in range(2, 0, -1):#2， 1
#             level = f'l{i}'
#             offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
#             offset = self.lrelu(self.offset_conv1[level](offset))
#             if i == 2:
#                 offset = self.lrelu(self.offset_conv2[level](offset))
#             else:
#                 offset = self.lrelu(self.offset_conv2[level](torch.cat(
#                     [offset, upsampled_offset], dim=1)))
#                 offset = self.lrelu(self.offset_conv3[level](offset))

#             feat = self.dcn_pack[level](torch.cat([offset, nbr_feat_l[i - 1]], dim=1))

#             if i < 2:
#                 feat = self.feat_conv[level](
#                     torch.cat([feat, upsampled_feat], dim=1))
#             if i > 1:
#                 feat = self.lrelu(feat)

#             if i > 1:  # upsample offset and features
#                 # x2: when we upsample the offset, we should also enlarge
#                 # the magnitude.
#                 upsampled_offset = self.upsample(offset)
#                 upsampled_feat = self.upsample(feat)
#         return feat

#替换为普通卷积后将下采计算偏移量支路删除
class PDA(nn.Module):

    def __init__(self, num_feat=64, deformable_groups=4):
        super(PDA, self).__init__()

        # Pyramid has three levels:
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size

        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)  # 步长为2，将特征图下采样至1/2大小
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 步长为1，保持特征图大小不变
        # self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)  # 再次下采样至1/4大小
        # self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 保持特征图大小不变

        self.offset_conv1 = nn.ModuleDict()  # 第一级偏移量卷积 创建空字典
        self.feat_conv = nn.ModuleDict()     # 特征融合卷积
        # Pyramids
        for i in range(2, 0, -1):       # 2, 1
            level = f'l{i}'             ## 生成'l2', 'l1'
            #为每一层创建一个用于计算偏移量的卷积层offset_conv1
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

            #仅为第一层创建feat_conv特征融合卷积
            if i < 2:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.ReLU(inplace=False)

    def forward(self, nbr_feat_l, ref_feat_l, video_dir=None):

        nbr_feat_l1 = nbr_feat_l
        nbr_feat_l2 = self.lrelu(self.conv_l2_1(nbr_feat_l1))
        nbr_feat_l2 = self.lrelu(self.conv_l2_2(nbr_feat_l2))
        # # L3
        # nbr_feat_l3 = self.lrelu(self.conv_l3_1(nbr_feat_l2))
        # nbr_feat_l3 = self.lrelu(self.conv_l3_2(nbr_feat_l3))

        nbr_feat_l = [nbr_feat_l1, nbr_feat_l2]

        ref_feat_l1 = ref_feat_l
        ref_feat_l2 = self.lrelu(self.conv_l2_1(ref_feat_l1))
        ref_feat_l2 = self.lrelu(self.conv_l2_2(ref_feat_l2))
        # L3
        # ref_feat_l3 = self.lrelu(self.conv_l3_1(ref_feat_l2))
        # ref_feat_l3 = self.lrelu(self.conv_l3_2(ref_feat_l3))

        ref_feat_l = [ref_feat_l1, ref_feat_l2]
        #取ref_feat_l[i], i = 0,1,2即取ref_feat_l1, ref_feat_l2, ref_feat_l3
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(2, 0, -1):#2， 1
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            feat = self.lrelu(self.offset_conv1[level](offset))

            if i < 2:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                upsampled_feat = self.upsample(feat)
        return feat


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

#调制过滤瓶颈MFB
class ModulationFilteringBottleneck(nn.Module):
    def __init__(self, block, in_channels, out_channels, modulator):
        super(ModulationFilteringBottleneck, self).__init__()
        self.modulator = modulator(in_channels)
        self.filter = base.make_layer(block, in_channels, out_channels)

    def forward(self, v):
        return self.filter(self.modulator(v))

#自蒸馏
class SelfDistillation(nn.Module):
    def __init__(self, block, in_channels, out_channels, num_classes, modulator):
        super(SelfDistillation, self).__init__()
        self.MFB = ModulationFilteringBottleneck(block, in_channels, out_channels, modulator)
        self.head_v = nn.Conv2d(in_channels, num_classes, 1)
        self.head_z = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, v):
        pred_v = self.head_v(v)  # p(y|v) 原始特征预测
        z = self.MFB(v)  # Variational Information Bottleneck: v -> z
        pred_z = self.head_z(z)  # p(y|z) 瓶颈特征预测

        return z, pred_z, pred_v


class ProgressiveDistillationDecoder(nn.Module):
    def __init__(self, num_classes, block, nb_filter, modulator):
        super(ProgressiveDistillationDecoder, self).__init__()
        #双线性插值上采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        self.decoder_3 = SelfDistillation(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_classes, modulator)
        self.decoder_2 = SelfDistillation(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_classes, modulator)
        self.decoder_1 = SelfDistillation(block, nb_filter[0] + nb_filter[1], nb_filter[0], num_classes, modulator)


    def forward(self, observations):
        #[1,64,48,48]上采样到[1,64,96,96]与[1,32,96,96]拼接得到[1,96,96,96]进入SelfDistillation
        #z_3=MFB([1,96,96,96])即CBAM（96）得到[1,32,96,96]
        #pred_z_3[1,1,384,384],pred_v_3[1,1,384,384]
        z_3, pred_z_3, pred_v_3 = self.decoder_3(torch.cat([observations[2], self.up(observations[3])], 1))  # [96, 96]
        pred_z_3, pred_v_3 = self.up_4(pred_z_3), self.up_4(pred_v_3)  # [384, 384]
        #z_2=MFB([1,48,192,192])即CBAM（48）得到[1,16,192,192]
        #pred_z_2[1,1,384,384],pred_v_2[1,1,384,384]
        z_2, pred_z_2, pred_v_2 = self.decoder_2(torch.cat([observations[1], self.up(z_3)], 1))  # [192, 192]
        pred_z_2, pred_v_2 = self.up(pred_z_2), self.up(pred_v_2)  # [384, 384]
        #z_1=MFB([1,24,384,384])即CBAM（24）得到[1,8,384,384]
        #pred_z_1[1,1,384,384],pred_v_1[1,1,384,384]        
        z_1, pred_z_1, pred_v_1 = self.decoder_1(torch.cat([observations[0], self.up(z_2)], 1))  # [384, 384]
        #pred_z_1用于预测掩码
        #[pred_z_1, pred_z_2, pred_z_3]用于深度监督的瓶颈特征预测列表
        #[pred_v_1, pred_v_2, pred_v_3]用于计算充分性损失的原始特征预测列表
        return pred_z_1, [pred_z_1, pred_z_2, pred_z_3], [pred_v_1, pred_v_2, pred_v_3]


#nb_filter定义四个层级的特征通道数
class RFR_STDNet(nn.Module):
    def __init__(self, num_frames=5, num_classes=1, in_channels=3, block=base.ResBlock, num_blocks=[2, 2, 2],
                 nb_filter=[8, 16, 32, 64],  modulator=base.CBAM, use_sufficiency_loss=True, deep_supervision=False):
        super(RFR_STDNet, self).__init__()
        self.num_levels = len(nb_filter)    #四个层级
        self.num_frames = num_frames        #五帧一组
        self.use_sufficiency_loss = use_sufficiency_loss
        self.deep_supervision = deep_supervision

        self.encoder = base.ResNet(in_channels, block, num_blocks, nb_filter)
        self.deform_aligns = nn.ModuleList([
            PDA(num_feat=nb_filter[j], deformable_groups=4) for j in range(self.num_levels)
        ])
        # self.residual_layer = nn.ModuleList([
        #     self.make_layer(functools.partial(ResBlock_3d, nb_filter[j]), 1)
        #     for j in range(self.num_levels)
        # ])

        # self.TA = nn.ModuleList([
        #     nn.Conv2d(num_frames * nb_filter[j], nb_filter[j], 1, 1, bias=True)
        #     for j in range(self.num_levels)
        # ])
        
        # self.IMC = ImplicitMotionCompensation(block, nb_filter, self.num_levels, shift_sizes, neighborhood_sizes)
        self.MFA = MultiFrameAggregation(block, nb_filter, num_frames, self.num_levels)
        self.decoder = ProgressiveDistillationDecoder(num_classes, block, nb_filter, modulator)

    #特征均为4维张量
    def forward(self, x):
        #取时间维度上的最后一帧，得到[1，3,384,384]，然后进行ResNet编码，得到当前帧的四个特征
        #curr_frame_feats[0,1,2,3]=（[1, 8, 384, 384],[1, 16, 192, 192],[1, 32, 96, 96],[1, 64, 48, 48]）encode输出四个元素的元组
        # curr_frame_feats = self.encoder(x[:, :, -1, :, :])
        #循环四次，每次创建一个新的空列表，即创建一个包含4个空列表的外层列表[[], [], [], []]
        all_level_feats = [[] for _ in range(self.num_levels)]

        # Implicit Motion Compensation
        for i in range(self.num_frames):#0,1,2,3,4
            frame_feats = self.encoder(x[:, :, i, :, :])
            for j in range(self.num_levels):#0,1,2,3
                #4个层级每个层级分别有五帧同样大小的特征
                #即在[[], [], [], []]中按x轴添加，即[[1], [2], [3], [4]]，[[1，5], [2，6], [3，7], [4，8]]······
                all_level_feats[j].append(frame_feats[j])
            # for i in range(self.num_frames):

        
        #PDA
        all_compensated_feats = [[] for _ in range(self.num_levels)]
        for i in range(self.num_levels):
            feat_current = all_level_feats[i][-1]
            for j in range(self.num_frames):
                feat_past = all_level_feats[i][j]
                compensated_feats = self.deform_aligns[i](feat_past, feat_current)
                all_compensated_feats[i].append(compensated_feats)
            


        #每个层级输出的observations和最初的对应层级的特征尺寸大小相同

        #observation[0,1,2,3] = [[1, 8, 384, 384],[1, 16, 192, 192],[1, 32, 96, 96],[1, 64, 48, 48]]
        observations = self.MFA([torch.cat(feats, dim=1) for feats in all_compensated_feats])

        # Progressive Distillation Decoder
        # ProgressiveDistillationDecoder(1, ResNet, [8,16,32,64], CBAM)
        pred, pred_z_list, pred_v_list = self.decoder(observations)

        if self.use_sufficiency_loss:
            return pred, pred_z_list, pred_v_list
        return pred_z_list if self.deep_supervision else pred


if __name__ == '__main__':
    model = RFR_STDNet().cuda()
    inputs = torch.randn((1, 3, 5, 384, 384)).cuda()  # Params = 0.85M FLOPs = 19.31G
    flops, params = profile(model, (inputs,))
    print('Params = ' + str(round(params / 1000 ** 2, 2)) + 'M')
    print('FLOPs = ' + str(round(flops / 1000 ** 3, 2)) + 'G')
    flops, params = clever_format([flops, params], '%.6f')
    print('Params = ' + params)
    print('FLOPs = ' + flops)