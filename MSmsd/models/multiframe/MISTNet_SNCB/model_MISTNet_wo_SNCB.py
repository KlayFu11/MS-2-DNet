from deepmist.models.multiframe.MISTNet_SNCB.base import *
from thop import profile, clever_format


class MultiFrameAggregation(nn.Module):
    def __init__(self, block, nb_filter, num_frames, num_levels):
        super(MultiFrameAggregation, self).__init__()
        self.num_levels = num_levels
        self.aggregators = nn.ModuleList([
            make_layer(block, nb_filter[j] * num_frames, nb_filter[j])
            for j in range(num_levels)
        ])

    def forward(self, compensated_feats):
        return [self.aggregators[j](compensated_feats[j]) for j in range(self.num_levels)]


class ModulationFilteringBottleneck(nn.Module):
    def __init__(self, block, in_channels, out_channels, modulator):
        super(ModulationFilteringBottleneck, self).__init__()
        self.modulator = modulator(in_channels)
        self.filter = make_layer(block, in_channels, out_channels)

    def forward(self, v):
        return self.filter(self.modulator(v))


class SelfDistillation(nn.Module):
    def __init__(self, block, in_channels, out_channels, num_classes, modulator):
        super(SelfDistillation, self).__init__()
        self.MFB = ModulationFilteringBottleneck(block, in_channels, out_channels, modulator)
        self.head_v = nn.Conv2d(in_channels, num_classes, 1)
        self.head_z = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, v):
        pred_v = self.head_v(v)  # p(y|v)
        z = self.MFB(v)  # Variational Information Bottleneck: v -> z
        pred_z = self.head_z(z)  # p(y|z)

        return z, pred_z, pred_v


class ProgressiveDistillationDecoder(nn.Module):
    def __init__(self, num_classes, block, nb_filter, modulator):
        super(ProgressiveDistillationDecoder, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.decoder_3 = SelfDistillation(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_classes, modulator)
        self.decoder_2 = SelfDistillation(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_classes, modulator)
        self.decoder_1 = SelfDistillation(block, nb_filter[0] + nb_filter[1], nb_filter[0], num_classes, modulator)

    def forward(self, observations):
        z_3, pred_z_3, pred_v_3 = self.decoder_3(torch.cat([observations[2], self.up(observations[3])], 1))  # [96, 96]
        pred_z_3, pred_v_3 = self.up_4(pred_z_3), self.up_4(pred_v_3)  # [384, 384]
        z_2, pred_z_2, pred_v_2 = self.decoder_2(torch.cat([observations[1], self.up(z_3)], 1))  # [192, 192]
        pred_z_2, pred_v_2 = self.up(pred_z_2), self.up(pred_v_2)  # [384, 384]
        z_1, pred_z_1, pred_v_1 = self.decoder_1(torch.cat([observations[0], self.up(z_2)], 1))  # [384, 384]

        return pred_z_1, [pred_z_1, pred_z_2, pred_z_3], [pred_v_1, pred_v_2, pred_v_3]


class MISTNet_wo_SNCB(nn.Module):
    def __init__(self, num_frames=5, num_classes=1, in_channels=3, block=ResBlock, num_blocks=[2, 2, 2],
                 nb_filter=[8, 16, 32, 64], modulator=CBAM, use_sufficiency_loss=True, deep_supervision=False):
        super(MISTNet_wo_SNCB, self).__init__()
        self.num_levels = len(nb_filter)
        self.num_frames = num_frames
        self.use_sufficiency_loss = use_sufficiency_loss
        self.deep_supervision = deep_supervision

        self.encoder = ResNet(in_channels, block, num_blocks, nb_filter)
        self.MFA = MultiFrameAggregation(block, nb_filter, num_frames, self.num_levels)
        self.decoder = ProgressiveDistillationDecoder(num_classes, block, nb_filter, modulator)

    def forward(self, x):
        curr_frame_feats = self.encoder(x[:, :, -1, :, :])
        all_compensated_feats = [[] for _ in range(self.num_levels)]

        # Implicit Motion Compensation
        for i in range(self.num_frames):
            past_frame_feats = curr_frame_feats if i == self.num_frames - 1 else self.encoder(x[:, :, i, :, :])
            compensated_feats = past_frame_feats
            for j in range(self.num_levels):
                all_compensated_feats[j].append(compensated_feats[j])

        # Multi-frame Aggregation
        observations = self.MFA([torch.cat(feats, dim=1) for feats in all_compensated_feats])

        # Progressive Distillation Decoder
        pred, pred_z_list, pred_v_list = self.decoder(observations)

        if self.use_sufficiency_loss:
            return pred, pred_z_list, pred_v_list
        return pred_z_list if self.deep_supervision else pred


if __name__ == '__main__':
    model = MISTNet_wo_SNCB().cuda()
    inputs = torch.randn((1, 3, 5, 384, 384)).cuda()  # Params = 0.85M FLOPs = 19.31G
    flops, params = profile(model, (inputs,))
    print('Params = ' + str(round(params / 1000 ** 2, 2)) + 'M')
    print('FLOPs = ' + str(round(flops / 1000 ** 3, 2)) + 'G')
    flops, params = clever_format([flops, params], '%.6f')
    print('Params = ' + params)
    print('FLOPs = ' + flops)
