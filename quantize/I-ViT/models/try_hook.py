import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import os
import math
import re
import warnings
from itertools import repeat
import collections.abc
from collections import OrderedDict
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from .layers_quant import PatchEmbed, Mlp, DropPath, trunc_normal_
from .quantization_utils import QuantLinear, QuantAct, QuantConv2d, IntLayerNorm, IntSoftmax, IntGELU, QuantMatMul, IntLayerNorm_LUT, IntGELU_HWF
from .utils import load_weights_from_npz
from .vit_quant import *

from models import *
from utils import *  
# 假設你已經有了所有的量化模組定義
# 這裡我們需要創建一個 hook 系統來捕獲中間結果

class DebugHook:
    def __init__(self, name):
        self.name = name
        self.inputs = []
        self.outputs = []
        self.weights = []
        self.scaling_factors = []
    
    def __call__(self, module, input, output):
        # 保存輸入
        if isinstance(input, tuple):
            self.inputs.append([x.clone() if isinstance(x, torch.Tensor) else x for x in input])
        else:
            self.inputs.append(input.clone() if isinstance(input, torch.Tensor) else input)
        
        # 保存輸出
        if isinstance(output, tuple):
            self.outputs.append([x.clone() if isinstance(x, torch.Tensor) else x for x in output])
        else:
            self.outputs.append(output.clone() if isinstance(output, torch.Tensor) else output)
        
        # 嘗試獲取權重（如果是線性層）
        if hasattr(module, 'weight') and module.weight is not None:
            self.weights.append(module.weight.clone())
        
        # 嘗試獲取 scaling factor（如果是量化模組）
        if hasattr(module, 'act_scaling_factor'):
            self.scaling_factors.append(module.act_scaling_factor)

def create_debug_model_and_get_layer0_info(model, input_tensor):
    """
    創建帶有 debug hooks 的模型並獲取 layer0 的詳細資訊
    
    Args:
        model: 你的 VisionTransformer 模型
        input_tensor: 輸入張量 (1, 198, 192)
    
    Returns:
        dict: 包含所有中間結果的字典
    """
    
    # 設置模型為評估模式
    model.eval()
    
    # 創建 hooks 字典
    hooks = {}
    handles = []
    
    # 為第一個 block（layer0）的所有子模組註冊 hooks
    first_block = model.blocks[0]
    
    # 註冊 hooks
    def register_hooks_recursively(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            hook = DebugHook(full_name)
            hooks[full_name] = hook
            handle = child.register_forward_hook(hook)
            handles.append(handle)
            
            # 遞歸註冊子模組
            if len(list(child.children())) > 0:
                register_hooks_recursively(child, full_name)
    
    # 為整個模型註冊 hooks（關注前面的層）
    hook_model = DebugHook("model_input")
    hooks["model_input"] = hook_model
    handle_model = model.register_forward_hook(hook_model)
    handles.append(handle_model)
    
    # 為 patch_embed 註冊 hook
    hook_patch = DebugHook("patch_embed")
    hooks["patch_embed"] = hook_patch
    handle_patch = model.patch_embed.register_forward_hook(hook_patch)
    handles.append(handle_patch)
    
    # 為第一個 block 註冊詳細的 hooks
    register_hooks_recursively(first_block, "layer0")
    
    try:
        # 前向傳播
        with torch.no_grad():
            # 如果輸入是 (1, 198, 192)，我們需要確保它經過正確的預處理
            if input_tensor.shape == (1, 198, 192):
                # 假設這是已經經過 patch embedding 的輸入
                # 我們需要模擬完整的前向傳播過程
                
                # 創建一個虛擬的圖像輸入來獲取正確的流程
                dummy_img = torch.randn(1, 3, 224, 224)
                
                # 獲取 patch embedding 的輸出
                B = dummy_img.shape[0]
                x, act_scaling_factor = model.qact_input(dummy_img)
                x, act_scaling_factor = model.patch_embed(x, act_scaling_factor)
                
                # 添加 cls token
                cls_tokens = model.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                
                # 添加位置編碼
                x_pos, act_scaling_factor_pos = model.qact_pos(model.pos_embed)
                x, act_scaling_factor = model.qact1(x, act_scaling_factor, x_pos, act_scaling_factor_pos)
                x = model.pos_drop(x)
                
                # 現在 x 應該是 (1, 198, 192) 的形狀，這是 layer0 的輸入
                layer0_input = x.clone()
                layer0_input_scaling_factor = act_scaling_factor
                
                # 通過第一個 block
                x, act_scaling_factor = model.blocks[0](x, act_scaling_factor)
                
            else:
                # 完整前向傳播
                output = model(input_tensor)
                layer0_input = None
                layer0_input_scaling_factor = None
        
        # 收集結果
        results = {
            'layer0_input': layer0_input,
            'layer0_input_scaling_factor': layer0_input_scaling_factor,
            'hooks_data': {}
        }
        
        # 整理 hooks 數據
        for name, hook in hooks.items():
            results['hooks_data'][name] = {
                'inputs': hook.inputs,
                'outputs': hook.outputs,
                'weights': hook.weights,
                'scaling_factors': hook.scaling_factors
            }
        
        return results
        
    finally:
        # 清理 hooks
        for handle in handles:
            handle.remove()

def analyze_layer0_operations(results):
    """
    分析 layer0 的操作詳情
    """
    print("=== Layer0 Input Information ===")
    if results['layer0_input'] is not None:
        print(f"Layer0 Input Shape: {results['layer0_input'].shape}")
        print(f"Layer0 Input Scaling Factor: {results['layer0_input_scaling_factor']}")
        print(f"Layer0 Input Stats: mean={results['layer0_input'].mean():.4f}, std={results['layer0_input'].std():.4f}")
    
    print("\n=== Layer0 Operations ===")
    for name, data in results['hooks_data'].items():
        if name.startswith('layer0'):
            print(f"\n--- {name} ---")
            
            # 輸入資訊
            if data['inputs']:
                for i, inp in enumerate(data['inputs']):
                    if isinstance(inp, list):
                        for j, sub_inp in enumerate(inp):
                            if isinstance(sub_inp, torch.Tensor):
                                print(f"  Input[{i}][{j}] Shape: {sub_inp.shape}, Stats: mean={sub_inp.mean():.4f}, std={sub_inp.std():.4f}")
                    elif isinstance(inp, torch.Tensor):
                        print(f"  Input[{i}] Shape: {inp.shape}, Stats: mean={inp.mean():.4f}, std={inp.std():.4f}")
            
            # 輸出資訊
            if data['outputs']:
                for i, out in enumerate(data['outputs']):
                    if isinstance(out, list):
                        for j, sub_out in enumerate(out):
                            if isinstance(sub_out, torch.Tensor):
                                print(f"  Output[{i}][{j}] Shape: {sub_out.shape}, Stats: mean={sub_out.mean():.4f}, std={sub_out.std():.4f}")
                    elif isinstance(out, torch.Tensor):
                        print(f"  Output[{i}] Shape: {out.shape}, Stats: mean={out.mean():.4f}, std={out.std():.4f}")
            
            # 權重資訊
            if data['weights']:
                for i, weight in enumerate(data['weights']):
                    print(f"  Weight[{i}] Shape: {weight.shape}, Stats: mean={weight.mean():.4f}, std={weight.std():.4f}")
            
            # Scaling factor 資訊
            if data['scaling_factors']:
                for i, sf in enumerate(data['scaling_factors']):
                    print(f"  Scaling Factor[{i}]: {sf}")

# 使用示例
def main():
    # 創建模型（假設你已經有了 deit_tiny_patch16_224 函數）
    model = deit_tiny_patch16_224(pretrained=False)
    
    # 創建輸入
    # 方法1: 使用圖像輸入
    img_input = torch.randn(1, 3, 224, 224)
    
    # 方法2: 如果你想直接使用 (1, 198, 192) 的輸入
    # direct_input = torch.randn(1, 198, 192)
    
    # 獲取調試資訊
    results = create_debug_model_and_get_layer0_info(model, img_input)
    
    # 分析結果
    analyze_layer0_operations(results)
    
    return results

# 額外的工具函數
def save_tensors_to_file(results, filename="layer0_debug.npz"):
    """
    將結果保存到文件
    """
    save_dict = {}
    
    if results['layer0_input'] is not None:
        save_dict['layer0_input'] = results['layer0_input'].numpy()
        save_dict['layer0_input_scaling_factor'] = results['layer0_input_scaling_factor']
    
    # 保存其他張量
    for name, data in results['hooks_data'].items():
        if name.startswith('layer0'):
            for i, inp in enumerate(data['inputs']):
                if isinstance(inp, torch.Tensor):
                    save_dict[f'{name}_input_{i}'] = inp.numpy()
            
            for i, out in enumerate(data['outputs']):
                if isinstance(out, torch.Tensor):
                    save_dict[f'{name}_output_{i}'] = out.numpy()
            
            for i, weight in enumerate(data['weights']):
                if isinstance(weight, torch.Tensor):
                    save_dict[f'{name}_weight_{i}'] = weight.numpy()
    
    np.savez_compressed(filename, **save_dict)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    results = main()
    save_tensors_to_file(results)