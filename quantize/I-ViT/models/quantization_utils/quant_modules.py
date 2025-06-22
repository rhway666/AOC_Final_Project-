import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn import Parameter
import logging

from .quant_utils import *


class QuantLinear(nn.Linear):
    """
    Class to quantize weights of given Linear layer

    Parameters:
    ----------
    weight_bit : int
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 weight_bit=8,
                 bias_bit=32,
                 per_channel=True,
                 quant_mode='symmetric'):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.weight_bit = weight_bit
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.quant_mode = quant_mode

        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        self.register_buffer('fc_scaling_factor', torch.zeros(self.out_features))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        if self.bias is not None:
            self.register_buffer('bias_integer', torch.zeros_like(self.bias))

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = "(" + s + " weight_bit={}, quant_mode={})".format(
            self.weight_bit, self.quant_mode)
        return s

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, prev_act_scaling_factor=None):
        with torch.no_grad():
            w = self.weight
            if self.per_channel:
                
                v = w.reshape(w.shape[0], -1)
                cur_min = v.min(axis=1).values
                cur_max = v.max(axis=1).values
                self.min_val = cur_min
                self.max_val = cur_max
            else:
                raise Exception('For weight, we only support per_channel quantization.')

            self.fc_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, self.min_val, self.max_val)

        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.fc_scaling_factor, True)

        bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

        if self.bias is not None:
            self.bias_integer = self.weight_function(
                self.bias, self.bias_bit, bias_scaling_factor, True)
        else:
            self.bias_integer = None

        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor

        return F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer) \
               * bias_scaling_factor, bias_scaling_factor


class QuantAct(nn.Module):
    """
    Class to quantize given activations
    Parameters:
    ----------
    activation_bit : int
        Bitwidth for quantized activations.
    act_range_momentum : float, default 0.95
        Momentum for updating the activation quantization range.
    running_stat : bool, default True
        Whether to use running statistics for activation quantization range.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    channel_len : int, default None
        Specify the channel length when using the per_channel mode.
    quant_mode : 'none' or 'asymmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """

    def __init__(self,
                 activation_bit=8,
                 act_range_momentum=0.95,
                 running_stat=True,
                 per_channel=False,
                 quant_mode="symmetric"):
        super(QuantAct, self).__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.per_channel = per_channel

        self.min_val = torch.zeros(1)
        self.max_val = torch.zeros(1)
        self.register_buffer('act_scaling_factor', torch.zeros(1))

        self.quant_mode = quant_mode
        self.per_channel = per_channel

        if self.quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def __repr__(self):
        return "{0}(activation_bit={1}, " \
               "quant_mode: {2}, Act_min: {3:.2f}, " \
               "Act_max: {4:.2f})".format(self.__class__.__name__, self.activation_bit,
                                          self.quant_mode, self.x_min.item(), self.x_max.item())

    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False

    def unfix(self):
        """
        unfix the activation range by setting running stat
        """
        self.running_stat = True

    def forward(self, x,
                pre_act_scaling_factor=None,
                identity=None,
                identity_scaling_factor=None):
        # collect runnng stats
        with torch.no_grad():
            x_act = x if identity is None else identity + x
            if self.running_stat:
                if len(x_act.shape) == 4:
                    x_act = x_act.permute(0, 2, 3, 1)
                # linear [N,C], CNN [N,C,H,W]
                v = x_act.reshape(-1, x_act.shape[-1]) #  # shape: [N*H*W, C]
                v = v.transpose(0, 1) # [C, N*H*W]

                cur_min = v.min(axis=1).values # 沿著channel方向找最大最小值
                cur_max = v.max(axis=1).values
                if torch.eq(self.min_val, self.max_val).all():  # PyTorch 中一種用來**檢查兩個 Tensor 是否「完全相等」 min 和 max 一樣 表示是 第一個 batch，還沒統計過任何輸入 用來初始化
                    """
                    a = torch.tensor([1.0, 2.0, 3.0])
                    b = torch.tensor([1.0, 9.0, 3.0])
                    print(torch.eq(a, b))
                    # tensor([True, False, True])
                    """
                    self.min_val = cur_min
                    self.max_val = cur_max
                else:
                    # batch間平滑的更新 min_val = 0 * 0.95 + cur_min * 0.05 
                    self.min_val = self.min_val * self.act_range_momentum + \
                                 cur_min * (1 - self.act_range_momentum)
                    self.max_val = self.max_val * self.act_range_momentum + \
                                   cur_max * (1 - self.act_range_momentum)
                self.max_val = self.max_val.max()
                self.min_val = self.min_val.min()

            self.act_scaling_factor = symmetric_linear_quantization_params(
                self.activation_bit, self.min_val, self.max_val)

        if pre_act_scaling_factor is None:
            # this is for the input quantization
            # 第一次量化輸入（例如模型的第一層）
            quant_act_int = self.act_function(x, self.activation_bit, self.act_scaling_factor, False)
            """
            forward(ctx, x, k, specified_scale, is_weight):
            x: floating point tensor to be quantized
            k: quantization bitwidth
            Note that the current implementation of SymmetricQuantFunction requires pre-calculated scaling factor.
            specified_scale: pre-calculated scaling factor for the tensor x
            """
        else:
            # 中間層，已經有前一層的 scaling factor（pre_act_scaling_factor）
            quant_act_int = fixedpoint_mul.apply(
                x, pre_act_scaling_factor,
                self.activation_bit, self.quant_mode,
                self.act_scaling_factor,
                identity, identity_scaling_factor)

        correct_output_scale = self.act_scaling_factor.view(-1)

        return quant_act_int * correct_output_scale, self.act_scaling_factor


class QuantMatMul(nn.Module):
    """
    Class to quantize weights of given matmul layer
    """
    def __init__(self):
        super(QuantMatMul, self).__init__()
        self.register_buffer('act_scaling_factor', torch.zeros(1))
    
    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, A, pre_act_scaling_factor_A, B, pre_act_scaling_factor_B):
        A_int = A / pre_act_scaling_factor_A
        B_int = B / pre_act_scaling_factor_B
        
        act_scaling_factor = pre_act_scaling_factor_A * pre_act_scaling_factor_B
        self.act_scaling_factor = act_scaling_factor
        return (A_int @ B_int) * act_scaling_factor, act_scaling_factor


class QuantConv2d(nn.Conv2d):
    """
    Class to quantize weights of given convolutional layer
    Parameters:
    ----------
    weight_bit : int, default 4
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    full_precision_flag : bool, default False
        If True, use fp32 and skip quantization
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    weight_percentile : float, default 0
        The percentile to setup quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 weight_bit=8,
                 bias_bit=32,
                 quant_mode="symmetric",
                 per_channel=True,
                 weight_percentile=0):
        super(QuantConv2d, self).__init__(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias
                                          )
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)

        self.register_buffer('conv_scaling_factor', torch.zeros(self.out_channels))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        self.register_buffer('bias_integer', torch.zeros_like(self.bias))

    def __repr__(self):
        s = super(QuantConv2d, self).__repr__()
        s = "(" + s + " weight_bit={}, quant_mode={})".format(self.weight_bit, self.quant_mode)
        return s

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, pre_act_scaling_factor=None):
        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        with torch.no_grad():
            w = self.weight
            if self.per_channel:
                v = w.reshape(w.shape[0], -1)
                cur_min = v.min(axis=1).values
                cur_max = v.max(axis=1).values
                self.min_val = cur_min
                self.max_val = cur_max
            else:
                raise Exception('For weight, we only support per_channel quantization.')

            self.conv_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, self.min_val, self.max_val)

        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.conv_scaling_factor, True)
        bias_scaling_factor = self.conv_scaling_factor * pre_act_scaling_factor
        self.bias_integer = self.weight_function(
            self.bias, self.bias_bit, bias_scaling_factor, True)

        pre_act_scaling_factor = pre_act_scaling_factor.view(1, -1, 1, 1)
        x_int = x / pre_act_scaling_factor
        correct_output_scale = bias_scaling_factor.view(1, -1, 1, 1)

        return (F.conv2d(x_int, self.weight_integer, self.bias_integer, self.stride, self.padding,
                         self.dilation, self.groups) * correct_output_scale, correct_output_scale)


class IntLayerNorm(nn.LayerNorm):
    """
    Implementation of I-LayerNorm
    Class to quantize given LayerNorm layer
    """
    def __init__(self, 
                normalized_shape, 
                eps=1e-5,
                elementwise_affine=True):
        super(IntLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
        self.dim_sqrt = None
        self.register_buffer('norm_scaling_factor', torch.zeros(1))
        self.register_buffer('bias_integer', torch.zeros_like(self.bias))

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, scaling_factor=None):
        if self.dim_sqrt is None:
            n = torch.tensor(x.shape[2], dtype=torch.float)
            self.dim_sqrt = torch.sqrt(n).cuda()

        # Normalization: computes mean and variance(std)
        x_int = x / scaling_factor
        mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
        y_int = x_int - mean_int
        y_sq_int = y_int ** 2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

        # Integer Iteration
        k = 2 ** 16
        for _ in range(10):
            k_1 = floor_ste.apply((k + floor_ste.apply(var_int/k))/2)
            k = k_1
        std_int = k

        factor = floor_ste.apply((2 ** 31-1) / std_int)
        y_int = floor_ste.apply(y_int * factor / 2)
        scaling_factor = self.dim_sqrt / 2 ** 30

        # scaling and shifting
        bias = self.bias.data.detach() / (self.weight.data.detach())
        bias_int = floor_ste.apply(bias / scaling_factor)

        self.bias_integer = bias_int
        

        y_int = y_int + bias_int
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor
        self.norm_scaling_factor = scaling_factor
        return x, scaling_factor


class IntLayerNorm_LUT(nn.LayerNorm):
    """
    Implementation of I-LayerNorm with lookup table for square root
    Class to quantize given LayerNorm layer using table lookup instead of Newton's method
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, log_path: str = "var_int_log.txt"):
        super().__init__(normalized_shape, eps, elementwise_affine)
        self.dim_sqrt = None
        self.register_buffer('norm_scaling_factor', torch.zeros(1))
        
        # Handle bias initialization properly
        if elementwise_affine and self.bias is not None:
            self.register_buffer('bias_integer', torch.zeros_like(self.bias))
        else:
            self.register_buffer('bias_integer', torch.zeros(normalized_shape))
        
        # Lookup table parameters for sqrt(x) approximation using y = ax + b
        # Based on the segments you provided
        
        # self.register_buffer('x_min', torch.tensor([0, 309676, 619353, 929030, 1238707, 1548384, 1858060, 2167737, 2477414, 2787091], dtype=torch.float32))
        # self.register_buffer('x_max', torch.tensor([309676, 619353, 929030, 1238707, 1548384, 1858060, 2167737, 2477414, 2787091, 3096768], dtype=torch.float32))
        # self.register_buffer('slope', torch.tensor([0.00179699, 0.00074434, 0.00057115, 0.00048150, 0.00042421, 0.00038352, 0.00035268, 0.00032827, 0.00030831, 0.00029161], dtype=torch.float32))
        # self.register_buffer('intercept', torch.tensor([0.000, 325.982, 433.246, 516.532, 587.498, 650.510, 707.806, 760.728, 810.157, 856.711], dtype=torch.float32))
        self.register_buffer('x_min',     torch.tensor([   100000,   1650210,   5075929,  10377157,  17553893,  26606139,  37533893,  50337157,  65015929,  81570210], dtype=torch.float32))
        self.register_buffer('x_max',     torch.tensor([  1650210,   5075929,  10377157,  17553893,  26606139,  37533893,  50337157,  65015929,  81570210, 100000000], dtype=torch.float32))
        self.register_buffer('slope',     torch.tensor([0.00062467, 0.00028268, 0.00018267, 0.00013493, 0.00010698, 0.00008862, 0.00007564, 0.00006597, 0.00005850, 0.00005254], dtype=torch.float32))
        self.register_buffer('intercept', torch.tensor([ 253,   818,  1325,  1821,  2311.883,  2800,  3287,  3774,  4260,  4745], dtype=torch.float32))

        self._global_var_min = float('inf')
        self._global_var_max = float('-inf')
        # self.var_int_stats = {
        #     'min': [],
        #     'max': [],
        #     'mean': []
        # }
        self.log_path = log_path
    def fix(self):
        pass
        
    def unfix(self):
        pass
    
    def lookup_sqrt(self, var_int):
        """
        Lookup table implementation of sqrt(x) using linear approximation y = ax + b
        """
        result = torch.zeros_like(var_int, dtype=torch.float32)
        
        # Process each segment
        for i in range(len(self.x_min)):
            # Create mask for values in current segment
            mask = (var_int >= self.x_min[i]) & (var_int < self.x_max[i])
            
            # Apply linear approximation y = ax + b for this segment
            if torch.any(mask):
                a_val = self.slope[i].to(var_int.device)
                b_val = self.intercept[i].to(var_int.device)
                result[mask] = a_val * var_int[mask] + b_val
        
        # Handle values >= last segment boundary (extrapolate using last segment)
        mask = var_int >= self.x_max[-1]
        if torch.any(mask):
            a_val = self.slope[-1].to(var_int.device)
            b_val = self.intercept[-1].to(var_int.device)
            result[mask] = a_val * var_int[mask] + b_val
            print("Warning: Values >= last segment boundary, using last segment slope and intercept.")
        # Handle values < first segment boundary (use first segment)
        mask = var_int < self.x_min[0]
        if torch.any(mask):
            a_val = self.slope[0].to(var_int.device)
            b_val = self.intercept[0].to(var_int.device)
            result[mask] = a_val * var_int[mask] + b_val
            
        return result
        
    def forward(self, x, scaling_factor=None):
        if self.dim_sqrt is None:
            n = torch.tensor(x.shape[2], dtype=torch.float)
            self.dim_sqrt = torch.sqrt(n).cuda()
            
        # Normalization: computes mean and variance(std)
        x_int = x / scaling_factor
        mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
        y_int = x_int - mean_int
        y_sq_int = y_int ** 2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
        
        # Ensure variance is positive and clamp to avoid edge cases
        var_int = torch.clamp(var_int, min=1.0)
       
        N = x.shape[2]
        
        var_int = var_int / N
        # with torch.no_grad():
        #     vmin = var_int.min().item()
        #     vmax = var_int.max().item()

        # # 2) 如果有設定 log_path，就把這筆寫入檔案
        # if self.log_path is not None:
        #     # 這裡每次都以 append 模式打開、寫入一行
        #     with open(self.log_path, 'a') as f:
        #         f.write(f"var_int range: {vmin:.2f} ~ {vmax:.2f}\n")
       
        # changed = False
       
        # with torch.no_grad():
        #     if vmin < self._global_var_min:
        #         self._global_var_min = vmin
        #         changed = True
        #     if vmax > self._global_var_max:
        #         self._global_var_max = vmax
        #         changed = True
        #     # print(f"[IntLayerNorm_LUT] var_int range: {vmin:.2f} ~ {vmax:.2f}")
        
        # if changed:
        #     print(f"[IntLayerNorm_LUT] Updated global var_int range: {self._global_var_min:.2f} ~ {self._global_var_max:.2f}")
        
        
        # print("var_int:", var_int)
        # # Use lookup table to get sqrt(var_int) instead of Newton's method
        std_int = self.lookup_sqrt(var_int)
        # std_int = (var_int ** 0.5)

        # Apply floor operation to maintain integer arithmetic
        std_int = floor_ste.apply(std_int)
        
        # std_int = std_int * (2 ** 10)  # Scale up to avoid too small values

        # k = 2 ** 16
        # for _ in range(10):
        #     k_1 = floor_ste.apply((k + floor_ste.apply(var_int/k))/2)
        #     k = k_1
        # std_int = k

        # print("std_int:", std_int)
        # Continue with the original quantization process
        factor = floor_ste.apply((2 ** 31-1) / std_int)
        y_int = floor_ste.apply(y_int * factor / 2)
        # scaling_factor = self.dim_sqrt / 2 ** 30
        scaling_factor = 1 / (2 ** 30)
        
        # scaling and shifting
        bias = self.bias.data.detach() / (self.weight.data.detach())
        bias_int = floor_ste.apply(bias / scaling_factor)
        self.bias_integer = bias_int

        # print(self.weight)
        y_int = y_int + bias_int
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor
        self.norm_scaling_factor = scaling_factor
        
        return x, scaling_factor
    
# class IntLayerNorm_LUT(nn.LayerNorm):
#     """
#     Implementation of I-LayerNorm with lookup table for inverse square root
#     Class to quantize given LayerNorm layer using table lookup instead of Newton's method
#     """
#     def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
#         super().__init__(normalized_shape, eps, elementwise_affine)
#         self.dim_sqrt = None
#         self.register_buffer('norm_scaling_factor', torch.zeros(1))
        
#         # Handle bias initialization properly
#         if elementwise_affine and self.bias is not None:
#             self.register_buffer('bias_integer', torch.zeros_like(self.bias))
#         else:
#             self.register_buffer('bias_integer', torch.zeros(normalized_shape))
        
#         # Lookup table parameters for y = ax + b approximation of 1/sqrt(x)
#         self.register_buffer('x_seg', torch.tensor([1, 2, 3, 4, 5, 46, 426, 3935, 36330, 335417, 3096768], dtype=torch.float32))
#         self.register_buffer('a_fp', torch.tensor([-0.29289, -0.12976, -0.07735, -0.05279, -0.00559, -0.00019, -0.00001, -0.00000, -0.00000, -0.00000], dtype=torch.float32))
#         self.register_buffer('b_fp', torch.tensor([1.29289, 0.96662, 0.80940, 0.71115, 0.36622, 0.11890, 0.03905, 0.01285, 0.00423, 0.00139], dtype=torch.float32))
        
#     def fix(self):
#         pass
        
#     def unfix(self):
#         pass
    
#     def lookup_inv_sqrt(self, var_int):
#         """
#         Lookup table implementation of 1/sqrt(x) using linear approximation y = ax + b
#         """
#         result = torch.zeros_like(var_int)
        
#         for i in range(len(self.x_seg) - 1):
#             # Create mask for values in current segment
#             mask = (var_int >= self.x_seg[i]) & (var_int < self.x_seg[i + 1])
            
#             # Apply linear approximation y = ax + b for this segment
#             if torch.any(mask):
#                 a_val = self.a_fp[i].to(var_int.device)
#                 b_val = self.b_fp[i].to(var_int.device)
#                 result[mask] = a_val * var_int[mask] + b_val
        
#         # Handle values >= last segment boundary
#         mask = var_int >= self.x_seg[-1]
#         if torch.any(mask):
#             a_val = self.a_fp[-1].to(var_int.device)
#             b_val = self.b_fp[-1].to(var_int.device)
#             result[mask] = a_val * var_int[mask] + b_val
            
#         # Handle values < first segment boundary (edge case)
#         mask = var_int < self.x_seg[0]
#         if torch.any(mask):
#             a_val = self.a_fp[0].to(var_int.device)
#             b_val = self.b_fp[0].to(var_int.device)
#             result[mask] = a_val * var_int[mask] + b_val
            
#         return result
        
#     def forward(self, x, scaling_factor=None):
#         if self.dim_sqrt is None:
#             n = torch.tensor(x.shape[2], dtype=torch.float)
#             self.dim_sqrt = torch.sqrt(n).cuda()
            
#         # Integer conversion
#         x_int = x / scaling_factor
        
#         # Following the paper's approach: Var(x) = E[x²] - E[x]²
#         # But using summation instead of mean for integer computation
        
#         # Calculate ∑x (sum of x)
#         sum_x = torch.sum(x_int, axis=2, keepdim=True)
        
#         # Calculate ∑x² (sum of x squared)
#         x_sq_int = x_int ** 2
#         sum_x_sq = torch.sum(x_sq_int, axis=2, keepdim=True)
        
#         # Calculate mean using integer division (can be approximated)
#         mean_int = round_ste.apply(sum_x / x.shape[2])
        
#         # Calculate variance using: Var = (∑x²)/N - (∑x/N)²
#         # But for integer computation, we use: N*Var = ∑x² - (∑x)²/N
#         # To avoid division, we compute: N²*Var = N*∑x² - (∑x)²
#         N = x.shape[2]
#         # var_numerator = N * sum_x_sq - sum_x ** 2 
#         #  # This is N²*Var
#         var_numerator =  sum_x_sq / N - (sum_x / N) ** 2

#         # Ensure variance is positive
#         var_numerator = torch.clamp(var_numerator, min=1.0)
        
#         # Use lookup table to get 1/sqrt(N²*Var) 
#         # inv_sqrt_var_scaled = self.lookup_inv_sqrt(var_numerator)
#         inv_sqrt_var_scaled = 1 / (var_numerator ** 0.5)

#         # To get 1/std = 1/sqrt(Var) = N/sqrt(N²*Var)
#         inv_std_int = inv_sqrt_var_scaled
        
#         # Apply normalization: (x - mean) * (1/std)
#         # y_int = (x_int - mean_int) * inv_std_int
#         y_int = (x_int - mean_int)
        
        
        
#         # y_int = floor_ste.apply(y_int * inv_std_int)

#         # Scaling factor calculation (prevent overflow)
#         # factor = floor_ste.apply((2 ** 31-1) / torch.max(torch.abs()))
        
#         # factor = floor_ste.apply((2 ** 31-1) / torch.max(torch.abs(inv_std_int)))
#         # y_int = floor_ste.apply(y_int * factor / 2)
#         # scaling_factor = 1 / 2 ** 30

#         factor = floor_ste.apply((2 ** 16-1) * torch.max(torch.abs(inv_std_int)))
#         y_int = floor_ste.apply(y_int * inv_std_int / 2)
#         scaling_factor = 1 / 2 ** 15


#         # Apply bias and weight (if elementwise_affine is True)
#         # if self.elementwise_affine:
#         bias = self.bias.data.detach() / (self.weight.data.detach())
#         bias_int = floor_ste.apply(bias / scaling_factor)
#         self.bias_integer = bias_int
#         y_int = y_int + bias_int
#         scaling_factor = scaling_factor * self.weight
        
#         x = y_int * scaling_factor
#         self.norm_scaling_factor = scaling_factor
        
#         return x, scaling_factor

# class IntLayerNorm_LUT(nn.LayerNorm):
#     """
#     Implementation of I-LayerNorm with lookup table for inverse square root
#     Class to quantize given LayerNorm layer using table lookup instead of Newton's method
#     """
#     def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
#         super().__init__(normalized_shape, eps, elementwise_affine)
#         self.dim_sqrt = None
#         self.register_buffer('norm_scaling_factor', torch.zeros(1))
        
#         # Handle bias initialization properly
#         if elementwise_affine and self.bias is not None:
#             self.register_buffer('bias_integer', torch.zeros_like(self.bias))
#         else:
#             self.register_buffer('bias_integer', torch.zeros(normalized_shape))
        
#         # Lookup table parameters for y = ax + b approximation of 1/sqrt(x)
#         self.register_buffer('x_seg', torch.tensor([1, 2, 3, 4, 5, 46, 426, 3935, 36330, 335417, 3096768], dtype=torch.float32))
#         self.register_buffer('a_fp', torch.tensor([-0.29289, -0.12976, -0.07735, -0.05279, -0.00559, -0.00019, -0.00001, -0.00000, -0.00000, -0.00000], dtype=torch.float32))
#         self.register_buffer('b_fp', torch.tensor([1.29289, 0.96662, 0.80940, 0.71115, 0.36622, 0.11890, 0.03905, 0.01285, 0.00423, 0.00139], dtype=torch.float32))
        
#     def fix(self):
#         pass
        
#     def unfix(self):
#         pass
    
#     def lookup_inv_sqrt(self, var_int):
#         """
#         Lookup table implementation of 1/sqrt(x) using linear approximation y = ax + b
#         """
#         # Find the appropriate segment for each variance value
#         batch_size, seq_len, _ = var_int.shape
#         result = torch.zeros_like(var_int)
        
#         for i in range(len(self.x_seg) - 1):
#             # Create mask for values in current segment
#             mask = (var_int >= self.x_seg[i]) & (var_int < self.x_seg[i + 1])
            
#             # Apply linear approximation y = ax + b for this segment
#             if torch.any(mask):
#                 a_val = self.a_fp[i].to(var_int.device)
#                 b_val = self.b_fp[i].to(var_int.device)
#                 result[mask] = a_val * var_int[mask] + b_val
        
#         # Handle values >= last segment boundary
#         mask = var_int >= self.x_seg[-1]
#         if torch.any(mask):
#             a_val = self.a_fp[-1].to(var_int.device)
#             b_val = self.b_fp[-1].to(var_int.device)
#             result[mask] = a_val * var_int[mask] + b_val
            
#         # Handle values < first segment boundary (edge case)
#         mask = var_int < self.x_seg[0]
#         if torch.any(mask):
#             a_val = self.a_fp[0].to(var_int.device)
#             b_val = self.b_fp[0].to(var_int.device)
#             result[mask] = a_val * var_int[mask] + b_val
            
#         return result
        
#     def forward(self, x, scaling_factor=None):
#         if self.dim_sqrt is None:
#             n = torch.tensor(x.shape[2], dtype=torch.float)
#             self.dim_sqrt = torch.sqrt(n).cuda()
            
#         # Normalization: computes mean and variance using E[x^2] - E[x]^2 formula
#         x_int = x / scaling_factor
        
#         # Calculate mean: E[x]
#         mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
        
#         # Calculate E[x^2]
#         x_sq_int = x_int ** 2
#         mean_sq_int = x_sq_int.mean(axis=2, keepdim=True)
        
#         # Calculate variance: E[x^2] - E[x]^2
#         var_int = mean_sq_int - mean_int ** 2
        
#         # Ensure variance is positive (add small epsilon if needed)
#         var_int = torch.clamp(var_int, min=1.0)  # Minimum value to avoid division issues
        
#         # Use lookup table to get 1/sqrt(variance)
#         # inv_std_int = self.lookup_inv_sqrt(var_int)
#         inv_std_int = 1/(var_int) ** 0.5
        
#         # Apply normalization: (x - mean) * (1/std)
#         y_int = (x_int - mean_int) * inv_std_int
        
        
#         # # Scaling factor calculation (similar to original)
#         # factor = floor_ste.apply((2 ** 31-1) / torch.max(torch.abs(y_int)))
#         # y_int = floor_ste.apply(y_int * factor / 2)
#         # # scaling_factor = self.dim_sqrt / 2 ** 30
#         # scaling_factor = 1 / 2 ** 30
        
#         # # Scaling and shifting (same as original)
#         # if self.elementwise_affine:
#         #     bias = self.bias.data.detach() / (self.weight.data.detach())
#         #     bias_int = floor_ste.apply(bias / scaling_factor)
#         #     self.bias_integer = bias_int
#         #     y_int = y_int + bias_int
#         #     scaling_factor = scaling_factor * self.weight
        
#         # x = y_int * scaling_factor
#         # self.norm_scaling_factor = scaling_factor
        
#         return x, scaling_factor





class IntGELU_HWF(nn.Module):
    """
    Class to quantize given GELU layer

    Parameters:
    ----------
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    force_dequant : str, default 'none'
        Force dequantize GELU if either 'gelu' or 'nonlinear' is given.
    """
    def __init__(self,
                 quant_mode='none',
                 force_dequant='none'):
        super(IntGELU_HWF, self).__init__()
        self.register_buffer('input_scaling_factor', torch.ones(1))
        self.quant_mode = quant_mode
        if force_dequant in ['nonlinear', 'gelu']:
            logger.info("Force dequantize gelu")
            self.quant_mode = 'none'


        if self.quant_mode == 'none':
            self.activation_fn = nn.GELU()
        elif self.quant_mode == 'symmetric':
            pass
        elif quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

        self.k = 1.4142
        self.n = 14 # sufficiently large integer
        self.coeff = [-0.2888, -1.769, 1] # a(x+b)**2 + c
        self.coeff[2] /= self.coeff[0]

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_erf(self, x_int, scaling_factor):
        with torch.no_grad():
            b_int = torch.floor(self.coeff[1] / scaling_factor)
            c_int = torch.floor(self.coeff[2] / scaling_factor ** 2)

        with torch.no_grad():
            sign = torch.sign(x_int)
        abs_int = torch.abs(x_int)
        abs_int = torch.min(abs_int, -b_int)
        y_int = (abs_int + b_int) ** 2 + c_int
        y_int = sign * y_int
        scaling_factor = scaling_factor ** 2 * self.coeff[0]
        y_int = floor_ste.apply(y_int / 2 ** self.n)
        scaling_factor = scaling_factor * 2 ** self.n
        
        return y_int, scaling_factor

    def forward(self, x, scaling_factor=None):
        if self.quant_mode == 'none':
            return self.activation_fn(x), None

        assert self.quant_mode == 'symmetric', \
                "unsupported quant mode: {}".format(quant_mode)

        x_int = x / scaling_factor
        
        sigmoid_int, sigmoid_scaling_factor = self.int_erf(x_int, scaling_factor / self.k)

        shift_int = torch.floor(1. / sigmoid_scaling_factor)

        x_int = x_int * (sigmoid_int + shift_int)
        scaling_factor = scaling_factor * sigmoid_scaling_factor / 2

        return x_int * scaling_factor, scaling_factor



class IntGELU(nn.Module):
    """
    Implementation of ShiftGELU
    Class to quantize given GELU layer
    """

    def __init__(self, output_bit=8):
        super(IntGELU, self).__init__()
        self.output_bit = output_bit

        self.n = 23  # sufficiently large integer
        #The minimum value for ensuring accuracy (varies depending on models)

        self.register_buffer('act_scaling_factor', torch.zeros(1))

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2 ** 4)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r/2 - x0_int
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2 ** self.n

        return exp_int, scaling_factor

    def forward(self, x, scaling_factor=None):
        pre_x_int = x / scaling_factor
        scaling_factor_sig = scaling_factor * 1.702

        x_int_max, _ = pre_x_int.max(dim=-1, keepdim=True)
        x_int = pre_x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor_sig) # e^(x-x_max)

        exp_int_max, _ = self.int_exp_shift(-x_int_max, scaling_factor_sig)  # e^(-x_max)
        exp_int_sum = exp_int + exp_int_max

        exp_int_sum.clamp_max_(2**31-1)
        factor = floor_ste.apply((2 ** 31-1) / exp_int_sum)
        sigmoid_int = floor_ste.apply(exp_int * factor / 2 ** (31-self.output_bit+1))
        sigmoid_scaling_factor = torch.Tensor([1 / 2 ** (self.output_bit-1)]).cuda()

        x_int = pre_x_int * sigmoid_int
        scaling_factor = scaling_factor * sigmoid_scaling_factor
        self.act_scaling_factor = scaling_factor
        return x_int * scaling_factor, scaling_factor


class IntSoftmax(nn.Module):
    """
    Implementation of Shiftmax
    Class to quantize given Softmax layer
    """

    def __init__(self, output_bit=8):
        super(IntSoftmax, self).__init__()
        self.output_bit = output_bit

        self.n = 15  # sufficiently large integer
        #The minimum value for ensuring accuracy (varies depending on models)

        self.register_buffer('act_scaling_factor', torch.zeros(1))

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2 ** 4)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r/2 - x0_int
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2 ** self.n
        return exp_int, scaling_factor

    def forward(self, x, scaling_factor):
        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

        exp_int_sum.clamp_max_(2**31-1)
        factor = floor_ste.apply((2**31-1) / exp_int_sum)
        exp_int = floor_ste.apply(exp_int * factor / 2 ** (31-self.output_bit+1))
        scaling_factor = torch.Tensor([1 / 2 ** (self.output_bit-1)]).cuda()

        self.act_scaling_factor = scaling_factor
        return exp_int * scaling_factor, scaling_factor
