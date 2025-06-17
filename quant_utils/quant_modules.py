                                                        # *
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
# *

## attention quantization : idea from ptq4vit <https://github.com/hahnyuan/PTQ4ViT>

import torch
import time
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from .quant_utils import AsymmetricQuantFunction,SymmetricQuantFunction,AsymmetricQuantFunctionAct, asymmetric_linear_quantization_params, linear_quantize, linear_dequantize, grad_scale, round_pass
import sys
from types import MethodType

from timm.models.vision_transformer import Attention
from timm.models.swin_transformer import WindowAttention


class QuantAct(Module):
    """
    Class to quantize given activations
    """
    
    def __init__(self,
                activation_bit,
                full_precision_flag=False,
                running_stat=True,
                beta=0.9):
        """
        activation_bit: bit-setting for activation
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(QuantAct, self).__init__()
        self.activation_bit = activation_bit
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))
        self.register_buffer('beta', torch.Tensor([beta]))
        self.register_buffer('beta_t', torch.ones(1))
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero_point', torch.zeros(1))
        self.act_function = AsymmetricQuantFunctionAct.apply
        
        if self.activation_bit >= 32: # FP setting
            self.full_precision_flag = True 
    
    def __repr__(self):
        return "{0}(activation_bit={1}, full_precision_flag={2}, running_stat={3}, Act_min: {4:.2f}, Act_max: {5:.2f})".format(
            self.__class__.__name__, self.activation_bit,
            self.full_precision_flag, self.running_stat, self.x_min.item(),
            self.x_max.item())
    
    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False
    
    def unfix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = True
        
    def asymmetric_linear_quantization_params(self,num_bits:int,
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
    
    def forward(self, x):
        """
        quantize given activation x
        """

        if self.running_stat:
            x_min = x.data.min()
            x_max = x.data.max()
            # in-place operation used on multi-gpus
            # self.x_min += -self.x_min + min(self.x_min, x_min)
            # self.x_max += -self.x_max + max(self.x_max, x_max)

            self.beta_t = self.beta_t * self.beta
            self.x_min = (self.x_min * self.beta + x_min * (1 - self.beta))/(1 - self.beta_t)
            self.x_max = (self.x_max * self.beta + x_max * (1 - self.beta)) / (1 - self.beta_t)
            self.scale, self.zero_point = self.asymmetric_linear_quantization_params(self.activation_bit, self.x_min, self.x_max)
        # print(self.__repr__)
        if not self.full_precision_flag:
            quant_act = self.act_function(x, self.activation_bit, self.scale, self.zero_point)
            return quant_act
        else:
            return x
        

class QuantAct_lsq(Module):
    """
    Class to quantize given activations
    """
    
    def __init__(self,
                activation_bit,
                full_precision_flag=False,
                lsq_g_scale=1.0):
        """
        activation_bit: bit-setting for activation
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(QuantAct_lsq, self).__init__()
        self.activation_bit = activation_bit
        self.full_precision_flag = full_precision_flag

        self.alpha = Parameter(torch.ones(1))
        self.zero_point = Parameter(torch.zeros(1))
        self.register_buffer('signed', torch.zeros(1))
        self.register_buffer('init_flag', torch.zeros(1))
        
        self.q_signed_minmax = [- (2**(self.activation_bit-1)), 2**(self.activation_bit-1)-1]
        self.q_unsigned_minmax = [0, 2**self.activation_bit - 1]
        
        if self.activation_bit >= 32: # FP setting
            self.full_precision_flag = True 
            
        self.lsq_g_scale = lsq_g_scale
    
    
    def __repr__(self):
        return "{0}(activation_bit={1}, full_precision_flag={2}, alpha={3:.2f}, zero_point={4:.2f}, signed: {5:.1f})".format(
            self.__class__.__name__, self.activation_bit,
            self.full_precision_flag, self.alpha.item(), self.zero_point.item(),
            self.signed.item())
    
    
    ## DUMMY FUNCTIONS (Fix, unfix) : MEANINGLESS
    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False
    
    def unfix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = True

    
    def forward(self, x):
        """
        quantize given activation x
        """ 
        
        if self.full_precision_flag:
            return x
        
        if not self.init_flag:
            if x.min() < 1e-5:
                self.signed.data.fill_(1)
                
            if self.signed == 1: 
                q_min, q_max = self.q_signed_minmax
            else:
                q_min, q_max = self.q_unsigned_minmax
                
            self.alpha.data.copy_(2*x.abs().mean()/math.sqrt(q_max))
            self.zero_point.data.copy_(self.zero_point.data*0.9 + 0.1* (torch.min(x.detach()) - self.alpha.data*q_max))
            
            self.init_flag.fill_(1)
            self.g_scale = self.lsq_g_scale / math.sqrt(x.numel()*q_max)
            
        if self.signed == 1: 
            q_min, q_max = self.q_signed_minmax
        else:
            q_min, q_max = self.q_unsigned_minmax
            
        alpha = grad_scale(self.alpha,self.g_scale)
        zero_point = grad_scale(self.zero_point,self.g_scale)
        
        zero_point = round_pass(self.zero_point)
        
        if len(x.shape)==2:
            alpha = alpha.view(-1,1)
            zero_point = zero_point.view(-1,1)
        elif len(x.shape)==4:
            alpha = alpha.view(-1,1,1,1)
            zero_point = zero_point.view(-1,1,1,1)
            
        x = round_pass( (x/alpha + zero_point).clamp(q_min,q_max))
        x = (x - zero_point) *alpha
        
        return x
    
    def update_bit(self):
        ''' caution: use right after the quant init only'''
        self.q_signed_minmax = [- (2**(self.activation_bit-1)), 2**(self.activation_bit-1)-1]
        self.q_unsigned_minmax = [0, 2**self.activation_bit - 1]
            




class Quant_Linear(Module):
    """
    Class to quantize given linear layer weights
    """
    
    def __init__(self, weight_bit, activation_bit, 
                weight_full_precision_flag=False,
                act_full_precision_flag=False,
                weight_q_mode='lsq', act_q_mode='lsq',
                lsq_g_scale=1.0): ######### lsq or minmax
        """
        weight: bit-setting for weight
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(Quant_Linear, self).__init__()
        self.weight_full_precision_flag = weight_full_precision_flag
        self.act_full_precision_flag = act_full_precision_flag

        self.weight_bit = weight_bit
        self.activation_bit = activation_bit
        
        self.weight_q_mode = weight_q_mode
        self.act_q_mode = act_q_mode

        if self.weight_q_mode == "lsq": 
            self.weight_function = self._lsquant
            self.q_min = - (2**self.weight_bit - 1)
            self.q_max = 2**(self.weight_bit-1) - 1
        elif self.weight_q_mode == "minmax":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.weight_q_mode == "minmax_asym":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise TypeError

        if self.act_q_mode == "lsq":
            self.quant_act = QuantAct_lsq(self.activation_bit, full_precision_flag=act_full_precision_flag,lsq_g_scale=lsq_g_scale)
        elif self.act_q_mode == "minmax":
            self.quant_act = QuantAct(self.activation_bit, full_precision_flag=act_full_precision_flag)
        else:
            raise TypeError

        self.eval_mode = False
        self.quant_weight = None
        
    def _lsquant(self, w): ## symmetric quantization
        ## careful, in lsq mode, alpha and zeropoint is reversed (scale=1/alpha, zp=-zp)
        
        alpha = grad_scale(self.alpha, self.grad_scale)
        alpha = alpha.unsqueeze(1)
        w_q = round_pass((self.weight/alpha).clamp(self.q_min, self.q_max)) * alpha
        return w_q

    
    def __repr__(self):
        s = super(Quant_Linear, self).__repr__()
        s = "(" + s + " weight_bit={}, weight_fp={}, wq_mode={})".format(
            self.weight_bit, self.weight_full_precision_flag, self.weight_q_mode)
        return s
    
    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = Parameter(linear.weight.data.clone())
        try:
            self.bias = Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None
        
        if self.weight_q_mode == "lsq":
            self.alpha = Parameter(torch.Tensor(self.out_features))
            # self.register_buffer('alpha_init', torch.zeros(1))
            self.alpha.data.copy_(2*self.weight.abs().mean() / math.sqrt(self.q_max))
            self.grad_scale = 1.0 / math.sqrt(self.weight.numel()*self.q_max)
            

    
    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        if not self.act_full_precision_flag:
            x_processed = self.quant_act(x)
            # x_processed = torch.utils.checkpoint(self.quant_act, x, use_reentrant=False)
        else:
            x_processed = x

        if self.eval_mode and self.quant_weight != None:
            w = self.quant_weight
            # print("YES!")
        else:
            # print(self.__repr__)
            if not self.weight_full_precision_flag:
                if "minmax" in self.weight_q_mode:
                    w = self.weight
                    x_transform = w.data.detach()
                    w_min = x_transform.min(dim=1).values
                    w_max = x_transform.max(dim=1).values
                    w = self.weight_function(self.weight, self.weight_bit, w_min,
                                        w_max)
                elif self.weight_q_mode == 'lsq':
                    w = self._lsquant(self.weight)
                
            else:
                w = self.weight
        return F.linear(x_processed, weight=w, bias=self.bias)

    def get_quant_weight(self, full_precision=False):
        if not full_precision:
            w = self.weight
            x_transform = w.data.detach()
            w_min = x_transform.min(dim=1).values
            w_max = x_transform.max(dim=1).values
            # print(self.__repr__)
            if not self.weight_full_precision_flag:
                w = self.weight_function(self.weight, self.weight_bit, w_min,
                                        w_max)
        else:
            w = self.weight
        return w
    
    # def quant_weight(self,full_precision=False):
    #     quant_dict = {}
    #     if not full_precision:
    #         # w = self.weight_function(self.weight, self.weight_bit, w_min,
    #         #                          w_max)
    #         w = self.weight
    #         x_transform = w.data.detach()
    #         w_min = x_transform.min(dim=1).values
    #         w_max = x_transform.max(dim=1).values
    #         scale, zero_point = asymmetric_linear_quantization_params(self.weight_bit, w_min, w_max)
    #         new_quant_x = linear_quantize(w, scale, zero_point, inplace=False)
    #         n = 2**(self.weight_bit - 1)
    #         new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
    #         quant_x = linear_dequantize(new_quant_x,
    #                                     scale,
    #                                     zero_point,
    #                                     inplace=False)
    #         quant_dict["w"] = self.weight
    #         quant_dict["scale"] = scale
    #         quant_dict["zero_point"] = zero_point
    #         quant_dict["quant"] = new_quant_x
    #         quant_dict["dequant"] = quant_x
    #     else:
    #         quant_dict["w"] = self.weight
    #         quant_dict["scale"] = None 
    #         quant_dict["zero_point"] = None 
    #         quant_dict["quant"] = None 
    #         quant_dict["dequant"] = None 
            
    #     return quant_dict

    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.quant_act.fix()
    
    def unfix(self):
        """
        fix the activation range by setting running stat
        """
        self.quant_act.unfix()
        
    def eval_mode_init(self):

        if "minmax" in self.weight_q_mode:
            w = self.weight
            x_transform = w.data.detach()
            w_min = x_transform.min(dim=1).values
            w_max = x_transform.max(dim=1).values
            
            self.quant_weight = self.weight_function(self.weight, self.weight_bit, w_min, w_max)
        elif self.weight_q_mode == "lsq":
            self.quant_weight = self._lsquant(self.weight)
            
    def update_bit(self):
        ''' caution: use right after the quant init only'''
        if self.weight_q_mode == "lsq":
            self.q_min = - (2**self.weight_bit - 1)
            self.q_max = 2**(self.weight_bit-1) - 1
            self.alpha.data.copy_(2*self.weight.abs().mean() / math.sqrt(self.q_max))
            self.grad_scale = 1.0 / math.sqrt(self.weight.numel()*self.q_max)
        
        if self.act_q_mode == "lsq":
            self.quant_act.update_bit()
        

class Quant_AvgPool2d(Module):
    """
    Class to quantize AvgPool2d layer
    """
    
    def __init__(self, weight_bit, activation_bit, 
                weight_full_precision_flag=False,
                act_full_precision_flag=False): #########
        """
        weight: bit-setting for weight
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(Quant_AvgPool2d, self).__init__()
        self.weight_full_precision_flag = weight_full_precision_flag #dummy
        self.act_full_precision_flag = act_full_precision_flag

        self.weight_bit = weight_bit
        self.activation_bit = activation_bit

        self.quant_act = QuantAct(self.activation_bit, full_precision_flag=act_full_precision_flag)

    
    def __repr__(self):
        s = super(Quant_AvgPool2d, self).__repr__()
        s = "(" + s + " weight_bit={}, weight_full_precision_flag={}, act_fp={})".format(
            self.weight_bit, self.weight_full_precision_flag, self.act_full_precision_flag)
        return s
    
    def set_param(self, pool):
        self.kernel_size = pool.kernel_size
        self.stride = pool.stride
        self.padding = pool.padding
    
    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        if not self.act_full_precision_flag:
            x_processed = self.quant_act(x)
        else:
            x_processed = x

        return F.avg_pool2d(x_processed, kernel_size=self.kernel_size,stride=self.stride,padding=self.padding)


    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.quant_act.fix()
    
    def unfix(self):
        """
        fix the activation range by setting running stat
        """
        self.quant_act.unfix()

class Quant_MaxPool2d(Module):
    """
    Class to quantize MaxPool2d layer
    """
    
    def __init__(self, weight_bit, activation_bit, 
                weight_full_precision_flag=False,
                act_full_precision_flag=False): #########
        """
        weight: bit-setting for weight
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(Quant_MaxPool2d, self).__init__()
        self.weight_full_precision_flag = weight_full_precision_flag #dummy
        self.act_full_precision_flag = act_full_precision_flag

        self.weight_bit = weight_bit
        self.activation_bit = activation_bit

        self.quant_act = QuantAct(self.activation_bit, full_precision_flag=act_full_precision_flag)

    
    def __repr__(self):
        s = super(Quant_MaxPool2d, self).__repr__()
        s = "(" + s + " weight_bit={}, weight_full_precision_flag={}, act_fp={})".format(
            self.weight_bit, self.weight_full_precision_flag, self.act_full_precision_flag)
        return s
    
    def set_param(self, pool):
        self.kernel_size = pool.kernel_size
        self.stride = pool.stride
        self.padding = pool.padding
    
    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        if not self.act_full_precision_flag:
            x_processed = self.quant_act(x)
        else:
            x_processed = x

        return F.max_pool2d(x_processed, kernel_size=self.kernel_size,stride=self.stride,padding=self.padding)


    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.quant_act.fix()
    
    def unfix(self):
        """
        fix the activation range by setting running stat
        """
        self.quant_act.unfix()

class Quant_Matmul(Module):
    """
    Class to quantize Matmul layer
    """
    
    def __init__(self, weight_bit, activation_bit, 
                weight_full_precision_flag=False,
                act_full_precision_flag=False,
                weight_q_mode='lsq', act_q_mode='lsq',
                lsq_g_scale=1.0): ######### weight_q_mode never be used
        """
        weight: bit-setting for weight
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(Quant_Matmul, self).__init__()
        self.weight_full_precision_flag = weight_full_precision_flag #dummy
        self.act_full_precision_flag = act_full_precision_flag

        self.weight_bit = weight_bit
        self.activation_bit = activation_bit
        
        self.weight_q_mode=weight_q_mode
        self.act_q_mode=act_q_mode
        
        if self.act_q_mode == "lsq":
            self.quant_act1 = QuantAct_lsq(self.activation_bit, full_precision_flag=act_full_precision_flag, lsq_g_scale=lsq_g_scale)
            self.quant_act2 = QuantAct_lsq(self.activation_bit, full_precision_flag=act_full_precision_flag, lsq_g_scale=lsq_g_scale)
        elif self.act_q_mode == "minmax":
            self.quant_act1 = QuantAct(self.activation_bit, full_precision_flag=act_full_precision_flag)
            self.quant_act2 = QuantAct(self.activation_bit, full_precision_flag=act_full_precision_flag)
        else:
            raise TypeError

    
    def __repr__(self):
        s = super(Quant_Matmul, self).__repr__()
        s = "(" + s + " weight_bit={}, weight_full_precision_flag={}, act_fp={})".format(
            self.weight_bit, self.weight_full_precision_flag, self.act_full_precision_flag)
        return s
    
    def forward(self, x1, x2):
        """
        using quantized weights to forward activation x
        """
        if not self.act_full_precision_flag:
            x1_processed = self.quant_act1(x1)
            x2_processed = self.quant_act2(x2)
        else:
            x1_processed = x1
            x2_processed = x2

        return x1_processed @ x2_processed


    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.quant_act1.fix()
        self.quant_act2.fix()
    
    def unfix(self):
        """
        fix the activation range by setting running stat
        """
        self.quant_act1.unfix()
        self.quant_act2.unfix()


class Quant_Conv2d(Module):
    """
    Class to quantize given convolutional layer weights
    """
    
    def __init__(self, weight_bit, activation_bit, 
                weight_full_precision_flag=False,
                act_full_precision_flag=False,
                weight_q_mode='lsq', act_q_mode='lsq',
                lsq_g_scale=1.0): #########
        super(Quant_Conv2d, self).__init__()
        self.weight_full_precision_flag = weight_full_precision_flag
        self.act_full_precision_flag = act_full_precision_flag

        self.weight_bit = weight_bit
        self.activation_bit = activation_bit
        
        self.weight_q_mode = weight_q_mode
        self.act_q_mode = act_q_mode
        
        if self.weight_q_mode == "lsq": 
            self.weight_function = self._lsquant
            self.q_min = - (2**self.weight_bit - 1)
            self.q_max = 2**(self.weight_bit-1) - 1
        elif self.weight_q_mode == "minmax":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.weight_q_mode == "minmax_asym":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise TypeError
        
        if self.act_q_mode == "lsq":
            self.quant_act = QuantAct_lsq(self.activation_bit, full_precision_flag=act_full_precision_flag,lsq_g_scale=lsq_g_scale)
        elif self.act_q_mode == "minmax":
            self.quant_act = QuantAct(self.activation_bit, full_precision_flag=act_full_precision_flag)
        else:
            raise TypeError
        
        self.eval_mode = False
        self.quant_weight = None
        
    def _lsquant(self, w): ## symmetric quantization
        ## careful, in lsq mode, alpha and zeropoint is reversed (scale=1/alpha, zp=-zp)
        
        alpha = grad_scale(self.alpha, self.grad_scale)
        alpha = alpha.view(-1,1,1,1)
        w_q = round_pass((self.weight/alpha).clamp(self.q_min, self.q_max)) * alpha
        return w_q
    
    def __repr__(self):
        s = super(Quant_Conv2d, self).__repr__()
        s = "(" + s + " weight_bit={}, weight_fp={}, wq_mode={})".format(
            self.weight_bit, self.weight_full_precision_flag, self.weight_q_mode)
        return s
    
    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = Parameter(conv.weight.data.clone())
        try:
            self.bias = Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None
            
        if self.weight_q_mode == "lsq":
            self.alpha = Parameter(torch.Tensor(self.out_channels))
            # self.register_buffer('alpha_init', torch.zeros(1))
            self.alpha.data.copy_(2*self.weight.abs().mean() / math.sqrt(self.q_max))
            self.grad_scale = 1.0 / math.sqrt(self.weight.numel()*self.q_max)
            
    
    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        if not self.act_full_precision_flag:
            x_processed = self.quant_act(x)
        else:
            x_processed = x
            
        if self.eval_mode and self.quant_weight != None:
            w = self.quant_weight
            # print("YES!")
        else:
            # print(self.__repr__)
            if not self.weight_full_precision_flag:
                if "minmax" in self.weight_q_mode:
                    w = self.weight
                    x_transform = w.data.contiguous().view(self.out_channels, -1)
                    w_min = x_transform.min(dim=1).values
                    w_max = x_transform.max(dim=1).values
                    w = self.weight_function(self.weight, self.weight_bit, w_min,
                                        w_max)
                elif self.weight_q_mode == 'lsq':
                    w = self._lsquant(self.weight)
            else:
                w = self.weight
        
        return F.conv2d(x_processed, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


    def quant_weight(self,full_precision=False):
        quant_dict = {}
        if not full_precision:
            # w = self.weight_function(self.weight, self.weight_bit, w_min,
            #                          w_max)
            w = self.weight
            x_transform = w.data.contiguous().view(self.out_channels, -1)
            w_min = x_transform.min(dim=1).values
            w_max = x_transform.max(dim=1).values
            scale, zero_point = asymmetric_linear_quantization_params(self.weight_bit, w_min, w_max)
            new_quant_x = linear_quantize(w, scale, zero_point, inplace=False)
            n = 2**(self.weight_bit - 1)
            new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
            quant_x = linear_dequantize(new_quant_x,
                                        scale,
                                        zero_point,
                                        inplace=False)
            quant_dict["w"] = self.weight
            quant_dict["scale"] = scale
            quant_dict["zero_point"] = zero_point
            quant_dict["quant"] = new_quant_x
            quant_dict["dequant"] = quant_x
        else:
            quant_dict["w"] = self.weight
            quant_dict["scale"] = None 
            quant_dict["zero_point"] = None 
            quant_dict["quant"] = None 
            quant_dict["dequant"] = None 
            
        return quant_dict
    
    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.quant_act.fix()
    
    def unfix(self):
        """
        fix the activation range by setting running stat
        """
        self.quant_act.unfix()
        
    def eval_mode_init(self):

        if "minmax" in self.weight_q_mode:
            w = self.weight
            x_transform = w.data.contiguous().view(self.out_channels, -1)
            w_min = x_transform.min(dim=1).values
            w_max = x_transform.max(dim=1).values
            
            self.quant_weight = self.weight_function(self.weight, self.weight_bit, w_min, w_max)
        elif self.weight_q_mode == "lsq":
            self.quant_weight = self._lsquant(self.weight)
    
    def update_bit(self):
        ''' caution: use right after the quant init only'''
        if self.weight_q_mode == "lsq":
            self.q_min = - (2**self.weight_bit - 1)
            self.q_max = 2**(self.weight_bit-1) - 1
            self.alpha.data.copy_(2*self.weight.abs().mean() / math.sqrt(self.q_max))
            self.grad_scale = 1.0 / math.sqrt(self.weight.numel()*self.q_max)
            
        if self.act_q_mode == "lsq":
            self.quant_act.update_bit()


##### from PTQ4VIT https://github.com/hahnyuan/PTQ4ViT/blob/f2959814559d674808b00afeb93dd0214fa0ad96/utils/models.py#L1

class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B

def attention_forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

    # attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    del q, k

    # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    del attn, v
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def window_attention_forward(self, x, mask = None):
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    q = q * self.scale
    # attn = (q @ k.transpose(-2, -1))
    attn = self.matmul1(q, k.transpose(-2,-1))

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)

    attn = self.attn_drop(attn)

    # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def prepare_vit_model(module):
    for n, child in module.named_children():
        if len(list(child.children())) > 0:
            prepare_vit_model(child)
        
        if isinstance(child, Attention):
            setattr(child, "matmul1", MatMul())
            setattr(child, "matmul2", MatMul())
            child.forward = MethodType(attention_forward,child)
        elif isinstance(child, WindowAttention):
            setattr(child, "matmul1", MatMul())
            setattr(child, "matmul2", MatMul())
            child.forward = MethodType(window_attention_forward,child)


##############

def quantize_model(module,qw=4,qa=4, weight_q_mode='lsq', act_q_mode='lsq', lsq_g_scale=1.0):

    for n, child in module.named_children():
        if len(list(child.children())) > 0:
            quantize_model(child,qw,qa,weight_q_mode=weight_q_mode, act_q_mode=act_q_mode)
            
        if isinstance(child, nn.Conv2d):
            quant_conv = Quant_Conv2d(weight_bit=qw, activation_bit=qa, weight_q_mode=weight_q_mode, act_q_mode=act_q_mode, lsq_g_scale=lsq_g_scale)
            quant_conv.set_param(child)
            setattr(module,n,quant_conv)
        elif isinstance(child, nn.Linear):
            quant_linear = Quant_Linear(weight_bit=qw, activation_bit=qa, weight_q_mode=weight_q_mode, act_q_mode=act_q_mode, lsq_g_scale=lsq_g_scale)
            quant_linear.set_param(child)
            setattr(module,n,quant_linear)
        # elif isinstance(child, nn.AvgPool2d):
        #     quant_avgpool = Quant_AvgPool2d(weight_bit=qw, activation_bit=qa)
        #     quant_avgpool.set_param(child)
        #     setattr(module,n,quant_avgpool)
        # elif isinstance(child, nn.MaxPool2d):
        #     quant_maxpool = Quant_MaxPool2d(weight_bit=qw, activation_bit=qa)
        #     quant_maxpool.set_param(child)
        #     setattr(module,n,quant_maxpool)
        # elif isinstance(child, nn.Identity):
        #     quant_act = QuantAct(activation_bit=qa)
            # quant_act.set_param(child)
            # setattr(module,n,quant_act)
        elif isinstance(child, MatMul):
            quant_matmul = Quant_Matmul(weight_bit=qw, activation_bit=qa, weight_q_mode=weight_q_mode, act_q_mode=act_q_mode, lsq_g_scale=lsq_g_scale)
            # quant_act.set_param(child)
            setattr(module,n,quant_matmul)
            
def quantize_model_only_attn(module,qw=4,qa=4, weight_q_mode='lsq', act_q_mode='lsq', lsq_g_scale=1.0):
    for n, child in module.named_children():
        if len(list(child.children())) > 0:
            quantize_model_only_attn(child,qw,qa,weight_q_mode=weight_q_mode, act_q_mode=act_q_mode)
            
        if isinstance(child, Attention):
            quantize_model(child,qw,qa,weight_q_mode=weight_q_mode, act_q_mode=act_q_mode)
        elif isinstance(child, WindowAttention):
            quantize_model(child,qw,qa,weight_q_mode=weight_q_mode, act_q_mode=act_q_mode)


def quantize_model_origin(module,qw=4,qa=4):

    # if type(module) == nn.Sequential:
    #     print(module)
    # print(module)
    for attr_str in dir(module):
        # print(attr_str)
        if "norm" in attr_str:
            continue # fast and dirty solution
        target_attr = getattr(module,attr_str)
        if type(target_attr) == nn.Conv2d:
            print("replaced: ", attr_str)
            quant_conv = Quant_Conv2d(weight_bit=qw, activation_bit=qa)
            quant_conv.set_param(target_attr)
            setattr(module, attr_str, quant_conv)
        elif type(target_attr) == nn.Linear:
            print("replaced: ", attr_str)
            quant_linear = Quant_Linear(weight_bit=qw, activation_bit=qa)
            quant_linear.set_param(target_attr)
            setattr(module, attr_str, quant_linear)
        # elif type(target_attr) == nn.ReLU or type(target_attr) == nn.ReLU6:
        #     print("replaced: ", attr_str)
        #     quant_act = nn.Sequential(*[target_attr, QuantAct(activation_bit=7)])
        #     setattr(module, attr_str, quant_act)
    
    for child in module.children():
        # if type(module) == nn.Sequential:
        print(child)
        quantize_model(child)

def set_first_last_layer(model, fl8bit=False):
    module_list = []
    for m in model.modules():
        if isinstance(m, Quant_Conv2d):
            module_list += [m]
        if isinstance(m, Quant_Linear):
            module_list += [m]
    module_list[0].act_full_precision_flag = True
    module_list[0].quant_act.full_precision_flag = True
    if fl8bit:
        module_list[0].weight_bit = 8
        module_list[-1].weight_bit = 8
        module_list[-1].quant_act.activation_bit = 8
    module_list[0].update_bit()
    module_list[-1].update_bit()
    # module_list[-1].activation_bit = 8 #torch.tensor(8)
    # module_list[-1].act_full_precision_flag = True

def freeze_model(model):
    module_list = []
    for m in model.modules():
        if isinstance(m, Quant_Conv2d):
            m.fix()
        if isinstance(m, Quant_Linear):
            m.fix()
        if isinstance(m, Quant_AvgPool2d):
            m.fix()
        if isinstance(m, Quant_MaxPool2d):
            m.fix()
        if isinstance(m, Quant_Matmul):
            m.fix()
        # if isinstance(m, QuantAct):
        #     m.fix()

def unfreeze_model(model):
    module_list = []
    for m in model.modules():
        if isinstance(m, Quant_Conv2d):
            m.unfix()
        if isinstance(m, Quant_Linear):
            m.unfix()
        if isinstance(m, Quant_AvgPool2d):
            m.unfix()
        if isinstance(m, Quant_MaxPool2d):
            m.unfix()
        if isinstance(m, Quant_Matmul):
            m.unfix()
        # if isinstance(m, QuantAct):
        #     m.unfix()

def full_precision_model(model):
    module_list = []
    for m in model.modules():
        if isinstance(m, Quant_Conv2d):
            m.weight_full_precision_flag = True
            m.act_full_precision_flag = True
        if isinstance(m, Quant_Linear):
            m.weight_full_precision_flag = True
            m.act_full_precision_flag = True
        if isinstance(m, Quant_Matmul):
            m.weight_full_precision_flag = True
            m.act_full_precision_flag = True
            
def un_full_precision_model(model):
    module_list = []
    for m in model.modules():
        if isinstance(m, Quant_Conv2d):
            m.weight_full_precision_flag = False
            m.act_full_precision_flag = False
        if isinstance(m, Quant_Linear):
            m.weight_full_precision_flag = False
            m.act_full_precision_flag = False
        if isinstance(m, Quant_Matmul):
            m.weight_full_precision_flag = False
            m.act_full_precision_flag = False

            
def to_train_mode(model):
    """
    unfreeze the activation range
    """
    if type(model) == Quant_Linear:
        model.eval_mode = False
        model.quant_weight = None
    elif type(model) == Quant_Conv2d:
        model.eval_mode = False
        model.quant_weight = None
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            to_train_mode(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                to_train_mode(mod)
        return model

def to_eval_mode(model):
    """
    unfreeze the activation range
    """
    if type(model) == Quant_Linear:
        model.eval_mode = True
        model.eval_mode_init() 
    elif type(model) == Quant_Conv2d:
        model.eval_mode = True
        model.eval_mode_init() 
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            to_eval_mode(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                to_eval_mode(mod)
        return model