import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftIoULoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean'):
        super(SoftIoULoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, mask):
        pred = torch.sigmoid(pred)
        intersection = pred * mask
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        mask_sum = torch.sum(mask, dim=(1, 2, 3))
        iou = (intersection_sum + self.smooth) / (pred_sum + mask_sum - intersection_sum + self.smooth)
        if self.reduction == 'mean':
            return 1 - iou.mean()
        elif self.reduction == 'sum':
            return 1 - iou.sum()
        else:
            raise NotImplementedError(f'reduction type {self.reduction} not implemented')



# class MultiSoftIoULoss(nn.Module):
#     def __init__(self, smooth=1, reduction='mean'):
#         super(MultiSoftIoULoss, self).__init__()
#         self.smooth = smooth
#         self.reduction = reduction
#         #self.scale_weights = [1.0, 0.6, 0.3] #加入每一层对应的loss权重
#         self.scale_weights = [0.7, 0.2, 0.1]

#     def soft_iou(self, pred, target):
#         # pred: (B, C, H, W)
#         pred = torch.sigmoid(pred)

#         intersection = pred * target
#         intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
#         pred_sum = torch.sum(pred, dim=(1, 2, 3))
#         target_sum = torch.sum(target, dim=(1, 2, 3))

#         iou = (intersection_sum + self.smooth) / (pred_sum + target_sum - intersection_sum + self.smooth)
#         if self.reduction == 'mean':
#             return 1 - iou.mean()
#         elif self.reduction == 'sum':
#             return 1 - iou.sum()
#         else:
#             raise NotImplementedError(f'reduction type {self.reduction} not implemented')
    
    
#     def forward(self, pred_list, mask):

#         # 原始尺度
#         mask_384 = mask

#         # 下采样（align_corners=False 防止插值伪振荡）
#         mask_192 = F.interpolate(mask, size=(192,192), mode="bilinear", align_corners=True)
#         mask_96  = F.interpolate(mask, size=(96,96),  mode="bilinear", align_corners=True)
#         # mask_48  = F.interpolate(mask, size=(48,48),  mode="bilinear", align_corners=True)

#         masks = [mask_384, mask_192, mask_96]
#         # masks = [mask_384, mask_192, mask_96, mask_48]

#         # print("mask sizes:", [m.shape for m in masks])
#         # 三个尺度分别计算 IoU
#         losses = []
#         for p, g, w in zip(pred_list, masks, self.scale_weights):
#             losses.append(w * self.soft_iou(p, g))
#         total_weight = sum(self.scale_weights)
#         # 归一化，使 loss 大小稳定
#         return sum(losses) / total_weight

#         # for p, g in zip(pred_list, masks):
#         #     losses.append(self.soft_iou(p, g))
#         # # 最终 loss = 三个 IoU loss 的平均
#         # return sum(losses)/len(losses)

# class MultiSoftIoULoss(nn.Module):
#     def __init__(self, smooth=1.0, reduction='mean', num_scales=3):
#         super(MultiSoftIoULoss, self).__init__()
#         self.smooth = smooth
#         self.reduction = reduction
#         self.num_scales = num_scales

#         # learnable logits（不是直接的权重）
#         # 初始化为 0 -> softmax 后是均匀分布
#         self.scale_logits = nn.Parameter(torch.zeros(num_scales))

#     def soft_iou(self, pred, target):
#         # pred: (B, 1, H, W)
#         pred = torch.sigmoid(pred)

#         intersection = pred * target
#         intersection_sum = intersection.sum(dim=(1, 2, 3))
#         pred_sum = pred.sum(dim=(1, 2, 3))
#         target_sum = target.sum(dim=(1, 2, 3))

#         iou = (intersection_sum + self.smooth) / \
#               (pred_sum + target_sum - intersection_sum + self.smooth)

#         if self.reduction == 'mean':
#             return 1.0 - iou.mean()
#         elif self.reduction == 'sum':
#             return 1.0 - iou.sum()
#         else:
#             raise NotImplementedError

#     def forward(self, pred_list, mask):
#         """
#         pred_list: [pred_384, pred_192, pred_96]
#         mask:      [B, 1, 384, 384]
#         """

#         # 1. 构造多尺度 GT
#         masks = [
#             mask,
#             F.interpolate(mask, size=(192, 192), mode='bilinear', align_corners=False),
#             F.interpolate(mask, size=(96, 96), mode='bilinear', align_corners=False),
#         ]

#         # 2. Softmax 得到可学习权重
#         scale_weights = torch.softmax(self.scale_logits, dim=0)
#         # scale_weights: (3,) , sum = 1

#         # 3. 逐尺度计算 IoU loss
#         losses = []
#         for p, g in zip(pred_list, masks):
#             losses.append(self.soft_iou(p, g))

#         losses = torch.stack(losses)  # (3,)

#         # 4. 加权求和（已经天然归一化）
#         total_loss = torch.sum(scale_weights * losses)

#         return total_loss

#Original Loss

class MultiSoftIoULoss(nn.Module):
    def __init__(self, smooth=1.0, reduction='mean',
                 prior_weights=(0.8, 0.15, 0.05)):
        super().__init__()

        self.smooth = smooth
        self.reduction = reduction

        # ===== 固定先验=====
        prior = torch.tensor(prior_weights, dtype=torch.float32)
        prior = prior / prior.sum()
        self.register_buffer('log_prior', torch.log(prior))

        # ===== 可学习偏移（epoch1 内会被训练）=====
        self.scale_logits = nn.Parameter(torch.zeros(len(prior)))

        # ===== 冻结相关 =====
        self.freeze = False
        self.register_buffer('fixed_weights', prior.clone())

    def current_weights(self):
        """
        对 Trainer 暴露：当前真正使用的权重
        """
        if self.freeze:
            return self.fixed_weights
        return torch.softmax(self.log_prior + self.scale_logits, dim=0)

    def freeze_with_weights(self, weights: torch.Tensor):
        """
        在 epoch1 结束后调用
        """
        with torch.no_grad():
            self.fixed_weights.copy_(weights)
            self.freeze = True
            self.scale_logits.requires_grad_(False)

    def soft_iou(self, pred, target):
        pred = torch.sigmoid(pred)

        intersection = pred * target
        intersection_sum = intersection.sum(dim=(1, 2, 3))
        pred_sum = pred.sum(dim=(1, 2, 3))
        target_sum = target.sum(dim=(1, 2, 3))

        iou = (intersection_sum + self.smooth) / \
              (pred_sum + target_sum - intersection_sum + self.smooth)

        if self.reduction == 'mean':
            return 1.0 - iou.mean()
        elif self.reduction == 'sum':
            return 1.0 - iou.sum()
        else:
            raise NotImplementedError

    def forward(self, pred_list, mask):
        # pred_list: [384, 192, 96]，利用差值函数来执行尺寸下采样
        mask_384 = mask
        mask_192 = F.interpolate(mask, size=(192, 192), mode='bilinear', align_corners=True)
        mask_96  = F.interpolate(mask, size=(96, 96),  mode='bilinear', align_corners=True)

        masks = [mask_384, mask_192, mask_96]
        weights = self.current_weights()

        loss = 0.0
        for p, g, w in zip(pred_list, masks, weights):
            loss = loss + w * self.soft_iou(p, g)

        return loss

#L2 loss

# class MultiSoftIoULoss(nn.Module):
#     def __init__(self, smooth=1.0, reduction='mean',
#                  prior_weights=(0.8, 0.15, 0.05),
#                  prior_weight=0.5):  # 先验损失权重
#         super().__init__()
#         self.smooth = smooth
#         self.reduction = reduction
#         self.prior_weight = prior_weight

#         prior = torch.tensor(prior_weights, dtype=torch.float32)
#         prior = prior / prior.sum()
#         self.register_buffer('prior', prior)

#         self.scale_logits = nn.Parameter(torch.zeros(len(prior)))
#         self.freeze = False
#         self.register_buffer('fixed_weights', prior.clone())

#     def current_weights(self):
#         if self.freeze:
#             return self.fixed_weights
#         return torch.softmax(self.prior + self.scale_logits, dim=0)

#     def get_prior_loss(self):
#         """KL散度约束权重接近先验"""
#         if self.freeze:
#             return 0.0
#         w = self.current_weights()
#         return F.kl_div(torch.log(w + 1e-8), self.prior, reduction='batchmean')

#     def soft_iou(self, pred, target):
#         pred = torch.sigmoid(pred)

#         intersection = pred * target
#         intersection_sum = intersection.sum(dim=(1, 2, 3))
#         pred_sum = pred.sum(dim=(1, 2, 3))
#         target_sum = target.sum(dim=(1, 2, 3))

#         iou = (intersection_sum + self.smooth) / \
#               (pred_sum + target_sum - intersection_sum + self.smooth)

#         if self.reduction == 'mean':
#             return 1.0 - iou.mean()
#         elif self.reduction == 'sum':
#             return 1.0 - iou.sum()
#         else:
#             raise NotImplementedError

#     def forward(self, pred_list, mask):
#         mask_384 = mask
#         mask_192 = F.interpolate(mask, size=(192, 192), mode='bilinear', align_corners=True)
#         mask_96 = F.interpolate(mask, size=(96, 96), mode='bilinear', align_corners=True)
#         masks = [mask_384, mask_192, mask_96]
#         weights = self.current_weights()

#         loss = 0.0
#         for p, g, w in zip(pred_list, masks, weights):
#             loss = loss + w * self.soft_iou(p, g)

#         return loss

#渐进式权重调整 前3个epoch逐渐允许权重偏离先验，第5个epoch后自动冻结权重

# class MultiSoftIoULoss(nn.Module):
#     def __init__(self, smooth=1.0, reduction='mean',
#                  prior_weights=(0.8, 0.15, 0.05),
#                  warmup_epochs=3,
#                  freeze_after_epoch=5):
#         super().__init__()
#         self.smooth = smooth
#         self.reduction = reduction
#         self.warmup_epochs = warmup_epochs
#         self.freeze_after_epoch = freeze_after_epoch

#         prior = torch.tensor(prior_weights, dtype=torch.float32)
#         prior = prior / prior.sum()
#         self.register_buffer('prior', prior)

#         # 用较小的初始值，让学习更缓慢
#         self.scale_logits = nn.Parameter(torch.full((len(prior),), -0.1))

#         self.freeze = False
#         self.register_buffer('fixed_weights', prior.clone())
#         self.current_epoch = 0

#     def set_epoch(self, epoch):
#         self.current_epoch = epoch
#         # 自动冻结
#         if epoch >= self.freeze_after_epoch and not self.freeze:
#             w = self.current_weights()
#             self.fixed_weights.copy_(w)
#             self.freeze = True
#             self.scale_logits.requires_grad_(False)
    
#     def soft_iou(self, pred, target):
#         pred = torch.sigmoid(pred)

#         intersection = pred * target
#         intersection_sum = intersection.sum(dim=(1, 2, 3))
#         pred_sum = pred.sum(dim=(1, 2, 3))
#         target_sum = target.sum(dim=(1, 2, 3))

#         iou = (intersection_sum + self.smooth) / \
#               (pred_sum + target_sum - intersection_sum + self.smooth)

#         if self.reduction == 'mean':
#             return 1.0 - iou.mean()
#         elif self.reduction == 'sum':
#             return 1.0 - iou.sum()
#         else:
#             raise NotImplementedError

#     def current_weights(self):
#         if self.freeze:
#             return self.fixed_weights
        
#         # warmup 期间逐渐允许偏离先验
#         if self.current_epoch < self.warmup_epochs:
#             alpha = self.current_epoch / self.warmup_epochs
#             # 从先验逐渐过渡到可学习
#             adjusted_logits = torch.log(self.prior + 1e-8) + self.scale_logits * alpha
#         else:
#             adjusted_logits = torch.log(self.prior + 1e-8) + self.scale_logits
        
#         return torch.softmax(adjusted_logits, dim=0)

#     def forward(self, pred_list, mask):
#         """
#         pred_list: [pred_384, pred_192, pred_96] - 三个不同尺度的预测
#         mask: ground truth mask
#         """
#         # 创建多尺度 GT
#         mask_384 = mask
#         mask_192 = F.interpolate(mask, size=(192, 192), mode='bilinear', align_corners=True)
#         mask_96 = F.interpolate(mask, size=(96, 96), mode='bilinear', align_corners=True)

#         masks = [mask_384, mask_192, mask_96]
#         weights = self.current_weights()

#         loss = 0.0
#         for p, g, w in zip(pred_list, masks, weights):
#             loss = loss + w * self.soft_iou(p, g)

#         return loss


#DiceLoss计算的是Dice系数，也称为F1分数
class DiceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.reduction = reduction
        self.eps = 1e-6

    def forward(self, pred, mask):
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * mask, dim=(1, 2, 3))
        total_sum = torch.sum((pred + mask), dim=(1, 2, 3))
        dice = 2 * intersection / (total_sum + self.eps)
        if self.reduction == 'mean':
            return 1 - dice.mean()
        elif self.reduction == 'sum':
            return 1 - dice.sum()
        else:
            raise NotImplementedError(f'reduction type {self.reduction} not implemented')


class BceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, mask):
        loss_fn = nn.BCEWithLogitsLoss(reduction=self.reduction)
        return loss_fn(pred, mask)


class L1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(L1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, mask):
        loss_fn = nn.L1Loss(reduction=self.reduction)
        return loss_fn(pred, mask)
