#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*

import math
import numpy as np
from torch.autograd import Function, Variable
# from .quant_modules import *
import torch
# torch._C._jit_set_bailout_depth(1)
# torch._C._jit_set_profiling_mode(False)


def clamp(input, min, max, inplace=False):
    """
    Clamp tensor input to (min, max).
    input: input tensor to be clamped
    """

    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)

@torch.jit.script
def linear_quantize(input, scale, zero_point, inplace:bool=False):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    input: single-precision input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    # print(input)
    if len(input.shape) == 4:
        # print("input shape",input.shape)
        # print("scale shape before view",scale.shape)
        # print("zero_point shape before view", zero_point.shape)
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
        # print("scale shape after view",scale.shape)
        # print("zero_point shape after view", zero_point.shape)
        # print()
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping single-precision input to integer values with the given scale and zeropoint
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point)

@torch.jit.script
def linear_dequantize(input, scale, zero_point, inplace:bool=False):
    """
    Map integer input tensor to fixed point float point with given scaling factor and zeropoint.
    input: integer input tensor to be mapped
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping integer input to fixed point float point value with given scaling factor and zeropoint
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale

@torch.jit.script
def asymmetric_linear_quantization_params(num_bits:int,
                                          saturation_min,
                                          saturation_max,
                                          integral_zero_point:bool=True,
                                          signed:bool=True):
    """
    Compute the scaling factor and zeropoint with the given quantization range.
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    """
    n = 2**num_bits - 1
    scale = n / torch.clamp((saturation_max - saturation_min), min=1e-8)
    zero_point = scale * saturation_min

    if integral_zero_point:
        if isinstance(zero_point, torch.Tensor):
            zero_point = zero_point.round()
        else:
            zero_point = float(round(zero_point))
    if signed:
        zero_point += 2**(num_bits - 1)
    return scale, zero_point

@torch.jit.script
def symmetric_linear_quantization_params(num_bits:int,
                                          saturation_min,
                                          saturation_max,
                                          integral_zero_point:bool=True,
                                          signed:bool=True):
    """
    Compute the scaling factor and zeropoint with the given quantization range.
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    """
    n = 2**num_bits - 1
    max_max = torch.max(-saturation_min, saturation_max)
    scale = n / torch.clamp(2*max_max, min=1e-8)
    zero_point = torch.zeros_like(scale)

    return scale, zero_point

@torch.jit.script
def merged_quantization_internal(x,k:int,x_min,x_max):
    scale, zero_point = symmetric_linear_quantization_params(
            k, x_min, x_max)
    new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
    n = 2**(k - 1)
    new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
    quant_x = linear_dequantize(new_quant_x,
                                scale,
                                zero_point,
                                inplace=False)
    return quant_x

@torch.jit.script
def merged_quantization_internal_asym(x,k:int,x_min,x_max):
    scale, zero_point = asymmetric_linear_quantization_params(
            k, x_min, x_max)
    new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
    n = 2**(k - 1)
    new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
    quant_x = linear_dequantize(new_quant_x,
                                scale,
                                zero_point,
                                inplace=False)
    return quant_x

@torch.jit.script
def merged_quantization_internal_act(x,k:int,scale,zero_point):
    new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
    n = 2**(k - 1)
    new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
    quant_x = linear_dequantize(new_quant_x,
                                scale,
                                zero_point,
                                inplace=False)
    return quant_x

class AsymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values with given range and bit-setting.
    Currently only support inference, but not support back-propagation.
    """
    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None):
        """
        x: single-precision value to be quantized
        k: bit-setting for x
        x_min: lower bound for quantization range
        x_max=None
        """

        # if x_min is None or x_max is None or (sum(x_min == x_max) == 1
        #                                       and x_min.numel() == 1):
        #     x_min, x_max = x.min(), x.max()
        # scale, zero_point = asymmetric_linear_quantization_params(
        #     k, x_min, x_max)
        # new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        # n = 2**(k - 1)
        # new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
        # quant_x = linear_dequantize(new_quant_x,
        #                             scale,
        #                             zero_point,
        #                             inplace=False)
        quant_x = merged_quantization_internal_asym(x,k,x_min,x_max)
        return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        # print("backward shape",grad_output)
        return grad_output, None, None, None

class SymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values with given range and bit-setting.
    Currently only support inference, but not support back-propagation.
    """
    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None):
        """
        x: single-precision value to be quantized
        k: bit-setting for x
        x_min: lower bound for quantization range
        x_max=None
        """

        # if x_min is None or x_max is None or (sum(x_min == x_max) == 1
        #                                       and x_min.numel() == 1):
        #     x_min, x_max = x.min(), x.max()
        # scale, zero_point = asymmetric_linear_quantization_params(
        #     k, x_min, x_max)
        # new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        # n = 2**(k - 1)
        # new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
        # quant_x = linear_dequantize(new_quant_x,
        #                             scale,
        #                             zero_point,
        #                             inplace=False)
        quant_x = merged_quantization_internal(x,k,x_min,x_max)
        return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        # print("backward shape",grad_output)
        return grad_output, None, None, None
    
class AsymmetricQuantFunctionAct(Function):
    """
    Class to quantize the given floating-point values with given range and bit-setting.
    Currently only support inference, but not support back-propagation.
    """
    @staticmethod
    def forward(ctx, x, k, scale=None, zero_point=None):
        """
        x: single-precision value to be quantized
        k: bit-setting for x
        x_min: lower bound for quantization range
        x_max=None
        """

        # if x_min is None or x_max is None or (sum(x_min == x_max) == 1
        #                                       and x_min.numel() == 1):
        #     x_min, x_max = x.min(), x.max()
        # scale, zero_point = asymmetric_linear_quantization_params(
        #     k, x_min, x_max)
        # new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        # n = 2**(k - 1)
        # new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
        # quant_x = linear_dequantize(new_quant_x,
        #                             scale,
        #                             zero_point,
        #                             inplace=False)
        quant_x = merged_quantization_internal_act(x,k,scale,zero_point)
        return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        # print("backward shape",grad_output)
        return grad_output, None, None, None

class AsymmetricQuantFunctionPerturb(Function):
    """
    Class to quantize the given floating-point values with given range and bit-setting.
    Currently only support inference, but not support back-propagation.
    """
    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None, perturb_portion=None):
        """
        x: single-precision value to be quantized
        k: bit-setting for x
        x_min: lower bound for quantization range
        x_max=None
        """

        # if x_min is None or x_max is None or (sum(x_min == x_max) == 1
        #                                       and x_min.numel() == 1):
        #     x_min, x_max = x.min(), x.max()
        scale, zero_point = asymmetric_linear_quantization_params(
            k, x_min, x_max)
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        n = 2**(k - 1)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)

        assert x.grad is not None
        current_grad_flat = x.grad.view(x.shape[0],-1)
        val, idx = torch.topk(current_grad_flat, k=round(current_grad_flat.shape[-1]*perturb_portion), dim=1, largest=False, sorted=False)

        perturbation = torch.zeros_like(current_grad_flat, device=x.device).scatter_(dim=1,index=idx,src=val.sign()).view(x.grad.shape)

        quant_x = linear_dequantize(new_quant_x+perturbation,
                                    scale,
                                    zero_point,
                                    inplace=False)
        return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        # print("backward shape",grad_output)
        return grad_output, None, None, None, None


class AsymmetricQuantFunctionPerturbNorm(Function):
    """
    Class to quantize the given floating-point values with given range and bit-setting.
    Currently only support inference, but not support back-propagation.
    """
    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None, perturb_portion=None):
        """
        x: single-precision value to be quantized
        k: bit-setting for x
        x_min: lower bound for quantization range
        x_max=None
        """

        # if x_min is None or x_max is None or (sum(x_min == x_max) == 1
        #                                       and x_min.numel() == 1):
        #     x_min, x_max = x.min(), x.max()
        scale, zero_point = asymmetric_linear_quantization_params(
            k, x_min, x_max)
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        n = 2**(k - 1)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)

        assert x.grad is not None
        current_grad_flat = x.grad.flatten()
        val, idx = torch.sort(current_grad_flat.abs(), descending=True)
        if len(x.shape) == 4:
            inverse_scale_square = ((1/(scale+1e-8))**2).view(-1,1,1,1).expand(x.grad.shape).flatten()
        elif len(x.shape) == 2:
            inverse_scale_square = ((1/(scale+1e-8))**2).view(-1,1).expand(x.grad.shape).flatten()
        scale_by_gradient_order = torch.gather(inverse_scale_square, dim=0, index=idx)
        cumulative_sum = torch.cumsum(scale_by_gradient_order, dim=0).sqrt()

        idx_found = torch.searchsorted(cumulative_sum, perturb_portion**2)
        # idx_found = min(idx_found, round(current_grad_flat.shape[0]*0.1))

        print(x.shape, idx_found.item(), (idx_found/x.flatten().shape[0])*100, x.norm())
        if x.norm() > 1000:
            print(x)
            print(x.grad)
            raise TypeError

        perturbation = torch.zeros_like(current_grad_flat, device=x.device).scatter_(dim=0,index=idx[:idx_found+1],src=-val.sign()).view(x.grad.shape) # -gradient 

        quant_x = linear_dequantize(new_quant_x+perturbation,
                                    scale,
                                    zero_point,
                                    inplace=False)
        return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        # print("backward shape",grad_output)
        return grad_output, None, None, None, None

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad
