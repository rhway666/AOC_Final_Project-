from models import *
from utils import *  
import hook  
# import torch
# import torch.nn as nn
# import os
# import numpy as np

# 你的量化相關自定義層會被 import 進來
#            # 這樣你可直接 vit_quant.deit_tiny_patch16_224(...)

# def save_tensor_txt(tensor, path, mode="int32"):
#     """
#     Save tensor as txt (int), default int32.
#     mode: "int32" (四捨五入直接存 int32)
#           "int8"  (scale/clip 成 int8)
#     """
#     arr = tensor.detach().cpu().numpy().flatten()
#     if mode == "int32":
#         arr = np.rint(arr).astype(np.int32)
#     elif mode == "int8":
#         max_abs = np.max(np.abs(arr))
#         scale = 1 if max_abs == 0 else 127 / max_abs
#         arr = np.clip(np.rint(arr * scale), -128, 127).astype(np.int8)
#     np.savetxt(path, arr[None], fmt="%d")

# def ensure_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def get_encoder0_activations(model: nn.Module, x: torch.Tensor, save_dir="activations", txt_mode="int32"):
#     """
#     Hook VisionTransformer.blocks[0] and all its submodules, 
#     recording all input/output activations as integer txt.

#     Args:
#         model: VisionTransformer instance
#         x: input tensor
#         save_dir: where to save activations
#         txt_mode: "int32" (default) or "int8"
#     Returns:
#         inputs: dict {name: input tensor}
#         outputs: dict {name: output tensor}
#     """
#     hook_layers = (
#         nn.Conv2d, nn.Linear, nn.LayerNorm, nn.Softmax, nn.GELU,
#         IntLayerNorm, QuantLinear, QuantAct, IntGELU
#     )
    
#     inputs = {}
#     outputs = {}
#     ensure_dir(save_dir)

#     def _get_act(name):
#         def hook(module, input, output):
#             # handle input
#             in_tensor = input[0] if isinstance(input, tuple) else input
#             if isinstance(in_tensor, torch.Tensor):
#                 inputs[name] = in_tensor.detach().cpu()
#                 path = os.path.join(save_dir, f"{name}.input.txt")
#                 save_tensor_txt(in_tensor, path, mode=txt_mode)
#             # handle output
#             out_tensor = output[0] if isinstance(output, tuple) else output
#             if isinstance(out_tensor, torch.Tensor):
#                 outputs[name] = out_tensor.detach().cpu()
#                 path = os.path.join(save_dir, f"{name}.output.txt")
#                 save_tensor_txt(out_tensor, path, mode=txt_mode)
#         return hook

#     # 只 hook blocks[0] (encoder0)
#     encoder0 = model.blocks[0]

#     # hook encoder0 input（block入口）
#     def encoder0_input_hook(module, input, output):
#         in_tensor = input[0] if isinstance(input, tuple) else input
#         if isinstance(in_tensor, torch.Tensor):
#             inputs['blocks.0.input'] = in_tensor.detach().cpu()
#             path = os.path.join(save_dir, "blocks.0.input.txt")
#             save_tensor_txt(in_tensor, path, mode=txt_mode)
#     encoder0.register_forward_hook(encoder0_input_hook)

#     # hook encoder0 所有內層 module
#     for name, module in encoder0.named_modules():
#         if isinstance(module, hook_layers):
#             full_name = f'blocks.0.{name}' if name else 'blocks.0'
#             module.register_forward_hook(_get_act(full_name))

#     # Run forward
#     model(x)
#     return inputs, outputs

# # 範例用法
# if __name__ == "__main__":
#     import hook      # 確保在 __main__ 也有 import


#     # 初始化模型
#     model = vit_quant.deit_tiny_patch16_224()
#     model.eval()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)  # 把 model 放到 GPU
#     x = torch.randn(1, 3, 224, 224, device=device)  # input 也要在同一個 device

#     inputs, outputs = get_encoder0_activations(model, x, save_dir="activations", txt_mode="int32")
#     print("All encoder0 activations are saved in ./activations/ as int txt.")

    # 如果想改 int8：
    # inputs, outputs = get_encoder0_activations(model, x, save_dir="activations", txt_mode="int8")



# import torch
# import torch.nn as nn
# import os
# import numpy as np


# def save_tensor_txt_fp(tensor, path):
#     """
#     Save tensor as txt (float32), one row, space separated, 6 decimals.
#     """
#     arr = tensor.detach().cpu().numpy().flatten().astype(np.float32)
#     np.savetxt(path, arr[None], fmt="%.6f")

# def ensure_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def get_encoder0_activations_fp(model: nn.Module, x: torch.Tensor, save_dir="activations_fp"):
#     """
#     Hook VisionTransformer.blocks[0] and all its submodules,
#     recording all input/output activations as float txt.
#     Returns: inputs, outputs (dict)
#     """
#     hook_layers = (
#         nn.Conv2d, nn.Linear, nn.LayerNorm, nn.Softmax, nn.GELU,
#         IntLayerNorm, QuantLinear, QuantAct, IntGELU
#     )
#     inputs = {}
#     outputs = {}
#     ensure_dir(save_dir)

#     def _get_act(name):
#         def hook(module, input, output):
#             in_tensor = input[0] if isinstance(input, tuple) else input
#             if isinstance(in_tensor, torch.Tensor):
#                 inputs[name] = in_tensor.detach().cpu()
#                 path = os.path.join(save_dir, f"{name}.input.txt")
#                 save_tensor_txt_fp(in_tensor, path)
#             out_tensor = output[0] if isinstance(output, tuple) else output
#             if isinstance(out_tensor, torch.Tensor):
#                 outputs[name] = out_tensor.detach().cpu()
#                 path = os.path.join(save_dir, f"{name}.output.txt")
#                 save_tensor_txt_fp(out_tensor, path)
#         return hook

#     encoder0 = model.blocks[0]
#     def encoder0_input_hook(module, input, output):
#         in_tensor = input[0] if isinstance(input, tuple) else input
#         if isinstance(in_tensor, torch.Tensor):
#             inputs['blocks.0.input'] = in_tensor.detach().cpu()
#             path = os.path.join(save_dir, "blocks.0.input.txt")
#             save_tensor_txt_fp(in_tensor, path)
#     encoder0.register_forward_hook(encoder0_input_hook)

#     for name, module in encoder0.named_modules():
#         if isinstance(module, hook_layers):
#             full_name = f'blocks.0.{name}' if name else 'blocks.0'
#             module.register_forward_hook(_get_act(full_name))

#     model(x)
#     return inputs, outputs

# if __name__ == "__main__":
#     model = vit_quant.deit_tiny_patch16_224()
#     model.eval()
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     x = torch.randn(1, 3, 224, 224, device=device)

#     inputs, outputs = get_encoder0_activations_fp(model, x, save_dir="activations_fp")
#     # print(model)
#     print("All encoder0 activations are saved in ./activations_fp/ as float txt.")



import os
import torch
import numpy as np
import re

from models import *
from utils import *
import hook

def safe_name(name):
    # 把 / . 轉成 _ 以免檔名有問題
    return re.sub(r'[^A-Za-z0-9_]', '_', name)

def shape_to_str(shape):
    return "x".join(str(s) for s in shape)

def save_tensor_txt_fp(tensor, path):
    arr = tensor.detach().cpu().numpy().flatten().astype(np.float32)
    np.savetxt(path, arr[None], fmt="%.6f")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_activations(model: nn.Module, x: torch.Tensor, save_dir="activations_fp", hook_layers=None):
    """
    Hook model所有子層，存 activation input/output (float)，檔名帶 shape。
    """
    if hook_layers is None:
        # 你可自定義或直接傳入
        hook_layers = (
            nn.Conv2d, nn.Linear, nn.LayerNorm, nn.Softmax, nn.GELU,
            IntLayerNorm, QuantLinear, QuantAct, IntGELU
        )
    ensure_dir(save_dir)
    inputs, outputs = {}, {}

    def _get_act(name):
        def hook(module, input, output):
            in_tensor = input[0] if isinstance(input, tuple) else input
            if isinstance(in_tensor, torch.Tensor):
                shape_str = shape_to_str(in_tensor.shape)
                fname = f"{safe_name(name)}.input.shape_{shape_str}.txt"
                save_tensor_txt_fp(in_tensor, os.path.join(save_dir, fname))
                inputs[name] = in_tensor.detach().cpu()
            out_tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(out_tensor, torch.Tensor):
                shape_str = shape_to_str(out_tensor.shape)
                fname = f"{safe_name(name)}.output.shape_{shape_str}.txt"
                save_tensor_txt_fp(out_tensor, os.path.join(save_dir, fname))
                outputs[name] = out_tensor.detach().cpu()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, hook_layers):
            module.register_forward_hook(_get_act(name))

    model.eval()
    with torch.no_grad():
        model(x)
    return inputs, outputs

def save_weights(model: nn.Module, save_dir="weights_fp", layers=None, quantized_layers=None):
    """
    存 weight 進 weights_fp 目錄，檔名帶 shape。
    """
    if layers is None:
        layers = (nn.Conv2d, nn.Linear)
    if quantized_layers is None:
        quantized_layers = (QuantLinear, QuantConv2d)
    ensure_dir(save_dir)
    weights = {}

    for name, module in model.named_modules():
        if isinstance(module, layers):
            w = module.weight.data
            shape_str = shape_to_str(w.shape)
            fname = f"{safe_name(name)}.weight.shape_{shape_str}.txt"
            save_tensor_txt_fp(w, os.path.join(save_dir, fname))
            weights[name] = w.detach().cpu()
        elif isinstance(module, quantized_layers):
            # 假設 quantized weight 也是 weight() 這樣存法
            try:
                w = module.weight() if callable(module.weight) else module.weight
                shape_str = shape_to_str(w.shape)
                fname = f"{safe_name(name)}.weight.shape_{shape_str}.txt"
                save_tensor_txt_fp(w, os.path.join(save_dir, fname))
                weights[name] = w.detach().cpu()
            except Exception as e:
                print(f"Warning: cannot save weight for {name}: {e}")
    return weights

if __name__ == "__main__":
    model = vit_quant.deit_tiny_patch16_224()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = torch.randn(1, 3, 224, 224, device=device)

    # 存 activation
    act_dir = "activations_fp"
    inputs, outputs = save_activations(model, x, save_dir=act_dir)
    print(f"All encoder activations are saved in ./{act_dir}/ as float txt.")

    # 存 weight
    w_dir = "weights_fp"
    weights = save_weights(model, save_dir=w_dir)
    print(f"All weights are saved in ./{w_dir}/ as float txt.")
