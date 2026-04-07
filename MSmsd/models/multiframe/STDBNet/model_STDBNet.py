import torch
import math

import torch.nn.functional as F

from torch.nn import init
from torch.nn.init import trunc_normal_
from torch.nn.modules.utils import _triple
import torch.nn as nn

from deepmist.models.multiframe.STDBNet import base_STD as base

from thop import profile, clever_format
import functools



#顺序经过卷积扩大感受野，采用标准卷积
class SEQUENCE(nn.Module):
    def __init__(self, num_feat):
        super(SEQUENCE, self).__init__()

        # 输入通道 = 2 * num_feat，切成4块后每块 = num_feat/2

        in_ch = 2 * num_feat

        # 只采用7*7/5*5的深度可分离卷积
        self.conv7 = nn.Conv2d(in_ch, in_ch, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(in_ch, in_ch, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_ch, num_feat, kernel_size=3, padding=1)

        self.act = nn.ReLU(inplace=False)
    def forward(self, past_feat, curr_feat):
        # B,C,H,W
        x = torch.cat([past_feat, curr_feat], dim=1)  # [B, 2C, H, W]

        x = self.act(self.conv7(x))
        #x = self.act(self.conv5(x))
        out = self.act(self.conv3(x))
        

        return out

#Depthwise + Pointwise
class DWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, padding):
        super(DWConvBlock, self).__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(x)


#顺序经过，采用深度可分卷积
class SEQUENCE_LITE(nn.Module):
    def __init__(self, num_feat):
        super(SEQUENCE_LITE, self).__init__()

        in_ch = 2 * num_feat

        # 三个深度卷积级联（比普通卷积省大量参数）
        #只采用7*7/5*5的深度可分离卷积
        self.act = nn.ReLU(inplace=False)

        self.conv9 = DWConvBlock(in_ch, in_ch, 9, padding=4)
        self.conv7 = DWConvBlock(in_ch, in_ch, 7, padding=3)
        self.conv5 = DWConvBlock(in_ch, in_ch, 5, padding=2)

        #通道合并为num_feat
        self.conv3 = nn.Conv2d(in_ch, num_feat, 3, padding=1)

    def forward(self, past_feat, curr_feat):
        x = torch.cat([past_feat, curr_feat], dim=1)

        #x = self.conv7(x)
        #x = self.conv5(x)
        x = self.conv9(x)
        x = self.conv3(x)

        return self.act(x)



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
class STDBNet(nn.Module):
    def __init__(self, num_frames=5, num_classes=1, in_channels=3, block=base.ResBlock, num_blocks=[2, 2, 2],
                 nb_filter=[8, 16, 32, 64],  modulator=base.CBAM, use_sufficiency_loss=True, deep_supervision=False):
        super(STDBNet, self).__init__()
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
        self.SEQ = nn.ModuleList([
            SEQUENCE(num_feat=nb_filter[j]) for j in range(self.num_levels)
        ])
        self.MFA = MultiFrameAggregation(block, nb_filter, num_frames, self.num_levels)
        self.decoder = ProgressiveDistillationDecoder(num_classes, block, nb_filter, modulator)

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
        #CHUNK
        for i in range(self.num_levels):
            for j in range(self.num_frames):
                all_compensated_feats[i].append(self.SEQ[i](all_level_feats[i][j], all_level_feats[i][-1]))
        
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
    model = STDBNet().cuda()
    inputs = torch.randn((1, 3, 5, 384, 384)).cuda()  # Params = 0.85M FLOPs = 19.31G
    flops, params = profile(model, (inputs,))
    print('Params = ' + str(round(params / 1000 ** 2, 2)) + 'M')
    print('FLOPs = ' + str(round(flops / 1000 ** 3, 2)) + 'G')
    flops, params = clever_format([flops, params], '%.6f')
    print('Params = ' + params)
    print('FLOPs = ' + flops)