from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _triple
from torch.autograd.function import once_differentiable
from torch.nn import init
from torch.nn.modules.utils import _pair


import DCN2 as DCN
#import D3D

class DeformConvFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, weight, bias,
                stride, padding, dilation, group, deformable_groups, im2col_step):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.group = group
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        output = DCN2.deform_conv_forward(input, weight, bias,
                                         offset,
                                         ctx.kernel_size[0], ctx.kernel_size[1],
                                         ctx.stride[0], ctx.stride[1],
                                         ctx.padding[0], ctx.padding[1],
                                         ctx.dilation[0], ctx.dilation[1],
                                         ctx.group,
                                         ctx.deformable_groups,
                                         ctx.im2col_step)
        ctx.save_for_backward(input, offset, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_weight, grad_bias = \
            DCN2.deform_conv_backward(input, weight,
                                     bias,
                                     offset,
                                     grad_output,
                                     ctx.kernel_size[0], ctx.kernel_size[1],
                                     ctx.stride[0], ctx.stride[1],
                                     ctx.padding[0], ctx.padding[1],
                                     ctx.dilation[0], ctx.dilation[1],
                                     ctx.group,
                                     ctx.deformable_groups,
                                     ctx.im2col_step)

        return grad_input, grad_offset, grad_weight, grad_bias,\
            None, None, None, None, None, None


            
class ModulatedDeformConvFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias,
                stride, padding, dilation, groups, deformable_groups, im2col_step):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        output = DCN.modulated_deform_conv_forward(input, weight, bias,
                                         offset, mask,
                                         ctx.kernel_size[0], ctx.kernel_size[1],
                                         ctx.stride[0], ctx.stride[1],
                                         ctx.padding[0], ctx.padding[1],
                                         ctx.dilation[0], ctx.dilation[1],
                                         ctx.groups,
                                         ctx.deformable_groups,
                                         ctx.im2col_step)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_mask, grad_weight, grad_bias = \
            DCN.modulated_deform_conv_backward(input, weight,
                                     bias,
                                     offset, mask,
                                     grad_output,
                                     ctx.kernel_size[0], ctx.kernel_size[1],
                                     ctx.stride[0], ctx.stride[1],
                                     ctx.padding[0], ctx.padding[1],
                                     ctx.dilation[0], ctx.dilation[1],
                                     ctx.groups,
                                     ctx.deformable_groups,
                                     ctx.im2col_step)

        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias,\
            None, None, None, None, None, None





# 定义DeformConvFunction
# class DeformConvFunction(Function):
#     @staticmethod
#     def forward(ctx, input, offset, weight, bias, 
#                 stride, padding, dilation, group, deformable_groups, im2col_step):
#         ctx.stride = _triple(stride)
#         ctx.padding = _triple(padding)
#         ctx.dilation = _triple(dilation)
#         ctx.kernel_size = _triple(weight.shape[2:5])
#         ctx.group = group
#         ctx.deformable_groups = deformable_groups
#         ctx.im2col_step = im2col_step
        
#         output = D3D.deform_conv_forward(input, weight, bias,
#                                         offset,
#                                         ctx.kernel_size[0], ctx.kernel_size[1], ctx.kernel_size[2],
#                                         ctx.stride[0], ctx.stride[1], ctx.stride[2],
#                                         ctx.padding[0], ctx.padding[1], ctx.padding[2],
#                                         ctx.dilation[0], ctx.dilation[1], ctx.dilation[2],
#                                         ctx.group,
#                                         ctx.deformable_groups,
#                                         ctx.im2col_step)
        
#         ctx.save_for_backward(input, offset, weight, bias)
#         return output

#     @staticmethod
#     @once_differentiable
#     def backward(ctx, grad_output):
#         input, offset, weight, bias = ctx.saved_tensors
        
#         grad_input, grad_offset, grad_weight, grad_bias = \
#             D3D.deform_conv_backward(input, weight,
#                                     bias,
#                                     offset,
#                                     grad_output,
#                                     ctx.kernel_size[0], ctx.kernel_size[1], ctx.kernel_size[2],
#                                     ctx.stride[0], ctx.stride[1], ctx.stride[2],
#                                     ctx.padding[0], ctx.padding[1], ctx.padding[2],
#                                     ctx.dilation[0], ctx.dilation[1], ctx.dilation[2],
#                                     ctx.group,
#                                     ctx.deformable_groups,
#                                     ctx.im2col_step)
        
#         return grad_input, grad_offset, grad_weight, grad_bias,\
#             None, None, None, None, None, None

# 定义DeformConv_d类
# class DeformConv_d(nn.Module):
    
#     def __init__(self, in_channels, out_channels,
#                  kernel_size, stride, padding, dimension='THW', dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True):
#         super(DeformConv_d, self).__init__()

#         if in_channels % groups != 0:
#             raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
#         if out_channels % groups != 0:
#             raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = _triple(kernel_size)
#         self.stride = _triple(stride)
#         self.padding = _triple(padding)
#         self.dilation = _triple(dilation)
#         self.dimension = dimension
#         self.length = len(dimension)
#         self.groups = groups
#         self.deformable_groups = deformable_groups
#         self.im2col_step = im2col_step
#         self.use_bias = bias

#         self.weight = nn.Parameter(torch.Tensor(
#             out_channels, in_channels // groups, *self.kernel_size))
#         self.bias = nn.Parameter(torch.Tensor(out_channels))
#         self.reset_parameters()
#         if not self.use_bias:
#             self.bias.requires_grad = False

#     def reset_parameters(self):
#         n = self.in_channels
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)

#     def forward(self, input, temp):
#         dimension_T = 'T' in self.dimension
#         dimension_H = 'H' in self.dimension
#         dimension_W = 'W' in self.dimension
#         b, c, t, h, w = temp.shape
#         if self.length == 2:
#             offset = temp.new_zeros(b, 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2], t, h, w)
#             if dimension_T == False:
#                 for i in range(
#                         self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
#                     offset[:, i * 3, :, :, :] = 0  # T
#                     offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2, :, :, :]
#                     offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
#             if dimension_H == False:
#                 for i in range(
#                         self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
#                     offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
#                     offset[:, i * 3 + 1, :, :, :] = 0  # H
#                     offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
#             if dimension_W == False:
#                 for i in range(
#                         self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
#                     offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
#                     offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2 + 1, :, :, :]
#                     offset[:, i * 3 + 2, :, :, :] = 0  # W

#         if self.length == 1:
#             offset = temp.new_zeros(b, 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2], t, h, w)
#             if dimension_T == True:
#                 for i in range(
#                         self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
#                     offset[:, i * 3, :, :, :] = temp[:, i, :, :, :]  # T
#                     offset[:, i * 3 + 1, :, :, :] = 0
#                     offset[:, i * 3 + 2, :, :, :] = 0
#             if dimension_H == True:
#                 for i in range(
#                         self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
#                     offset[:, i * 3, :, :, :] = 0
#                     offset[:, i * 3 + 1, :, :, :] = temp[:, i, :, :, :]  # H
#                     offset[:, i * 3 + 2, :, :, :] = 0
#             if dimension_W == True:
#                 for i in range(
#                         self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
#                     offset[:, i * 3, :, :, :] = 0
#                     offset[:, i * 3 + 1, :, :, :] = 0
#                     offset[:, i * 3 + 2, :, :, :] = temp[:, i, :, :, :]  # W

#         return DeformConvFunction.apply(input, offset,
#                                         self.weight,
#                                         self.bias,
#                                         self.stride,
#                                         self.padding,
#                                         self.dilation,
#                                         self.groups,
#                                         self.deformable_groups,
#                                         self.im2col_step)

# # 定义DeformConvPack_d类
# class DeformConvPack_d(DeformConv_d):
    
#     def __init__(self, in_channels, out_channels,
#                  kernel_size, stride, padding, dimension='THW',
#                  dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1):
#         super(DeformConvPack_d, self).__init__(in_channels, out_channels,
#                                              kernel_size, stride, padding, dimension, dilation, groups, deformable_groups,
#                                              im2col_step, bias)
        
#         self.dimension = dimension
#         self.length = len(dimension)
#         out_channels = self.deformable_groups * self.length * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
#         self.conv_offset = nn.Conv3d(self.in_channels,
#                                      out_channels,
#                                      kernel_size=self.kernel_size,
#                                      stride=self.stride,
#                                      padding=self.padding,
#                                      bias=True)
#         self.conv_offset.lr_mult = lr_mult
#         self.init_offset()

#     def init_offset(self):
#         self.conv_offset.weight.data.zero_()
#         self.conv_offset.bias.data.zero_()

#     def forward(self, input):
#         temp = self.conv_offset(input)
#         dimension_T = 'T' in self.dimension
#         dimension_H = 'H' in self.dimension
#         dimension_W = 'W' in self.dimension
#         b, c, t, h, w = temp.shape
#         if self.length == 2:
#             offset = temp.new_zeros(b, 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2], t, h, w)
#             if dimension_T == False:
#                 for i in range(
#                         self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
#                     offset[:, i * 3, :, :, :] = 0  # T
#                     offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2, :, :, :]
#                     offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
#             if dimension_H == False:
#                 for i in range(
#                         self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
#                     offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
#                     offset[:, i * 3 + 1, :, :, :] = 0  # H
#                     offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
#             if dimension_W == False:
#                 for i in range(
#                         self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
#                     offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
#                     offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2 + 1, :, :, :]
#                     offset[:, i * 3 + 2, :, :, :] = 0  # W

#         if self.length == 1:
#             offset = temp.new_zeros(b, 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2], t, h, w)
#             if dimension_T == True:
#                 for i in range(
#                         self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
#                     offset[:, i * 3, :, :, :] = temp[:, i, :, :, :]  # T
#                     offset[:, i * 3 + 1, :, :, :] = 0
#                     offset[:, i * 3 + 2, :, :, :] = 0
#             if dimension_H == True:
#                 for i in range(
#                         self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
#                     offset[:, i * 3, :, :, :] = 0
#                     offset[:, i * 3 + 1, :, :, :] = temp[:, i, :, :, :]  # H
#                     offset[:, i * 3 + 2, :, :, :] = 0
#             if dimension_W == True:
#                 for i in range(
#                         self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
#                     offset[:, i * 3, :, :, :] = 0
#                     offset[:, i * 3 + 1, :, :, :] = 0
#                     offset[:, i * 3 + 2, :, :, :] = temp[:, i, :, :, :]  # W

#         return DeformConvFunction.apply(input, offset,
#                                         self.weight,
#                                         self.bias,
#                                         self.stride,
#                                         self.padding,
#                                         self.dilation,
#                                         self.groups,
#                                         self.deformable_groups,
#                                         self.im2col_step)

class ConvOffset2d(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, im2col_step=128, bias=True):
        super(ConvOffset2d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
               offset.shape[1]
        return DeformConvFunction.apply(input, offset,
                                        self.weight,
                                        self.bias,
                                        self.stride,
                                        self.padding,
                                        self.dilation,
                                        self.groups,
                                        self.deformable_groups,
                                        self.im2col_step)


_DeformConv = DeformConvFunction.apply


class DeformConvPack(ConvOffset2d):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=128, bias=True, lr_mult=0.1):
        super(DeformConvPack, self).__init__(in_channels, out_channels,
                                             kernel_size, stride, padding, dilation, groups, deformable_groups,
                                             im2col_step, bias)

        out_channels = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = nn.Conv2d(self.in_channels,
                                     out_channels,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input):
        offset = self.conv_offset(input)
        return DeformConvFunction.apply(input, offset,
                                        self.weight,
                                        self.bias,
                                        self.stride,
                                        self.padding,
                                        self.dilation,
                                        self.groups,
                                        self.deformable_groups,
                                        self.im2col_step)


class DeformConv(ConvOffset2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=128, bias=True, lr_mult=0.1):
        super(DeformConv, self).__init__(in_channels, out_channels,
                                         kernel_size, stride, padding, dilation, groups, deformable_groups, im2col_step,
                                         bias)
        out_channels = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = nn.Conv2d(self.in_channels,
                                     out_channels,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        # constant_init(self.conv_offset[-1], val=0, bias=0)
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input, offset):
        out = self.conv_offset(offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        # offset_absmean = torch.mean(torch.abs(offset))
        # if offset_absmean > 50:
        #     print(
        #         f'Offset abs mean is {offset_absmean}, larger than 50.'
        #     )
        # offset = self.max_residual_magnitude * torch.tanh(offset)

        return ModulatedDeformConvFunction.apply(input, offset, mask,
                                                 self.weight,
                                                 self.bias,
                                                 self.stride,
                                                 self.padding,
                                                 self.dilation,
                                                 self.groups,
                                                 self.deformable_groups,
                                                 self.im2col_step)


class DeformConv_split(ConvOffset2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=128, bias=True, lr_mult=0.1):
        super(DeformConv_split, self).__init__(in_channels, out_channels,
                                               kernel_size, stride, padding, dilation, groups, deformable_groups,
                                               im2col_step, bias)
        out_channels0 = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        out_channels1 = self.deformable_groups * 1 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset0 = nn.Conv2d(self.in_channels // 2,
                                      out_channels0,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      padding=self.padding,
                                      bias=True)
        self.conv_offset1 = nn.Conv2d(self.in_channels // 2,
                                      out_channels1,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      padding=self.padding,
                                      bias=True)
        self.conv_offset0.lr_mult = lr_mult
        self.conv_offset1.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        # constant_init(self.conv_offset[-1], val=0, bias=0)
        self.conv_offset0.weight.data.zero_()
        self.conv_offset0.bias.data.zero_()
        self.conv_offset1.weight.data.zero_()
        self.conv_offset1.bias.data.zero_()

    def forward(self, input, offset):
        offset_fea, mask_fea = torch.chunk(offset, 2, dim=1)
        offset = self.conv_offset0(offset_fea)
        mask = self.conv_offset1(mask_fea)
        mask = torch.sigmoid(mask)
        # offset_absmean = torch.mean(torch.abs(offset))
        # if offset_absmean > 50:
        #     print(
        #         f'Offset abs mean is {offset_absmean}, larger than 50.'
        #     )
        # offset = self.max_residual_magnitude * torch.tanh(offset)

        return ModulatedDeformConvFunction.apply(input, offset, mask,
                                                 self.weight,
                                                 self.bias,
                                                 self.stride,
                                                 self.padding,
                                                 self.dilation,
                                                 self.groups,
                                                 self.deformable_groups,
                                                 self.im2col_step)



