import torch
import torch.nn.functional as F
import numpy as np
import math

def float_to_dyadic(value, max_bits=8):
    """
    將浮點數轉換為 dyadic 格式: a ≈ b / 2^c
    其中 b 和 c 都是 int8 範圍內的整數
    
    Args:
        value: 浮點數值
        max_bits: b 和 c 的最大位數 (默認8位)
    
    Returns:
        tuple: (original_float, b, c, approximated_float)
    """
    if value == 0:
        return (0.0, 0, 0, 0.0)
    
    # 找到最佳的 c 值
    max_val = 2**(max_bits-1) - 1  # 127 for int8
    min_val = -2**(max_bits-1)     # -128 for int8
    
    best_error = float('inf')
    best_b, best_c = 0, 0
    
    # 嘗試不同的 c 值
    for c in range(max_bits):  # c 從 0 到 7
        scale = 2**c
        b_float = value * scale
        
        # b 必須在 int8 範圍內
        if min_val <= b_float <= max_val:
            b = int(round(b_float))
            approx_value = b / scale
            error = abs(value - approx_value)
            
            if error < best_error:
                best_error = error
                best_b, best_c = b, c
    
    # 如果沒找到合適的值，使用最接近的
    if best_error == float('inf'):
        c = max_bits - 1
        scale = 2**c
        b = int(np.clip(round(value * scale), min_val, max_val))
        best_b, best_c = b, c
    
    approximated = best_b / (2**best_c)
    return (float(value), int(best_b), int(best_c), approximated)

def generate_residual_int32(shape, bit_width=16):
    """
    生成模擬的 residual 數據 (16-bit 範圍但用 int32 存儲)
    
    Args:
        shape: 張量形狀
        bit_width: 位寬 (默認16位)
    
    Returns:
        torch.Tensor: int32 張量，數值範圍在 16-bit 內
    """
    max_val = 2**(bit_width-1) - 1  # 32767 for 16-bit
    min_val = -2**(bit_width-1)     # -32768 for 16-bit
    
    # 生成 16-bit 範圍的隨機整數，用 int32 存儲
    residual = torch.randint(min_val, max_val + 1, shape, dtype=torch.int32)
    return residual

def quantized_inference_with_residual(checkpoint=None):
    """
    完整的量化推理實現，包含 residual 連接
    """
    print("=== Quantized Inference with Residual Connection ===")
    
    # 模擬參數
    B, N, C = 1, 198, 192
    
    # 1. 生成隨機輸入 (FP32)
    print("\n=== Step 1: Generate Random Input (FP32) ===")
    input_fp32 = torch.randn(B, N, C, dtype=torch.float32)
    print(f"Input FP32 shape: {input_fp32.shape}")
    print(f"Input FP32 range: [{input_fp32.min():.6f}, {input_fp32.max():.6f}]")
    
    # 2. 計算量化縮放因子並轉換為 dyadic 格式 (Per-tensor Symmetric)
    print("\n=== Step 2: Quantization Scaling Factor (Per-tensor Symmetric) ===")
    n_bits = 8
    max_val = 2**(n_bits-1) - 1  # 127 for 8-bit
    
    # Per-tensor scaling factor (整個tensor只有一個scaling factor)
    alpha_float_scalar = torch.max(torch.abs(input_fp32)) / max_val
    
    # 轉換為 dyadic 格式
    orig_float, b, c, approx_float = float_to_dyadic(alpha_float_scalar.item())
    dyadic_results = [(orig_float, b, c, approx_float)]
    alpha_dyadic_scalar = torch.tensor(approx_float, dtype=torch.float32)
    
    print(f"Per-tensor scaling factor:")
    print(f"  Original: {orig_float:.8f}")
    print(f"  Dyadic: {b}/2^{c} = {approx_float:.8f}")
    print(f"  Error: {abs(orig_float-approx_float):.8f}")
    print(f"  b (int8): {b}, c (int8): {c}")
    
    # 3. 輸入量化 (使用 per-tensor dyadic scaling factor)
    print("\n=== Step 3: Input Quantization (Per-tensor Symmetric INT8) ===")
    input_quantized = (input_fp32 / alpha_dyadic_scalar).round().clamp(-max_val, max_val)
    input_int8 = input_quantized.to(torch.int8)
    print(f"Quantized input range: [{input_int8.min()}, {input_int8.max()}]")
    print(f"Per-tensor scaling applied: {alpha_dyadic_scalar:.8f}")
    
    # 4. 生成量化權重和偏置
    print("\n=== Step 4: Generate Quantized Weights and Bias ===")
    if checkpoint:
        # 從 checkpoint 加載
        weight_int8 = checkpoint["blocks.0.attn.proj.weight_integer"].to(torch.int8)
        bias_int32 = checkpoint["blocks.0.attn.proj.bias_integer"].to(torch.int32)
        weight_scaling = checkpoint["blocks.0.attn.proj.fc_scaling_factor"]
    else:
        # 生成模擬數據
        weight_int8 = torch.randint(-127, 128, (C, C), dtype=torch.int8)
        bias_int32 = torch.randint(-1000, 1000, (C,), dtype=torch.int32)
        weight_scaling = torch.rand(C) * 0.01
    
    print(f"Weight INT8 shape: {weight_int8.shape}")
    print(f"Bias INT32 shape: {bias_int32.shape}")
    print(f"Weight range: [{weight_int8.min()}, {weight_int8.max()}]")
    print(f"Bias range: [{bias_int32.min()}, {bias_int32.max()}]")
    
    # 5. 線性運算 (INT32 輸出)
    print("\n=== Step 5: Linear Operation (INT32) ===")
    linear_output_int32 = F.linear(
        input_int8.float(), 
        weight_int8.float(), 
        bias_int32.float()
    ).to(torch.int32)
    
    print(f"Linear output INT32 shape: {linear_output_int32.shape}")
    print(f"Linear output range: [{linear_output_int32.min()}, {linear_output_int32.max()}]")
    
    # 6. 生成 Residual (INT32, 16-bit 範圍)
    print("\n=== Step 6: Generate Residual (INT32, 16-bit range) ===")
    residual_int32 = generate_residual_int32((B, N, C), bit_width=16)
    print(f"Residual INT32 shape: {residual_int32.shape}")
    print(f"Residual range: [{residual_int32.min()}, {residual_int32.max()}]")
    
    # 7. Residual 加法 (INT32)
    print("\n=== Step 7: Residual Addition (INT32) ===")
    final_output_int32 = linear_output_int32 + residual_int32
    print(f"Final output INT32 shape: {final_output_int32.shape}")
    print(f"Final output range: [{final_output_int32.min()}, {final_output_int32.max()}]")
    
    # 8. 保存所有結果為 TXT 文件
    print("\n=== Step 8: Save All Results to TXT Files ===")
    save_all_results_to_txt(
        input_fp32, alpha_float_scalar, dyadic_results, alpha_dyadic_scalar,
        input_int8, weight_int8, bias_int32, linear_output_int32, 
        residual_int32, final_output_int32
    )
    
    # 9. 驗證結果
    print("\n=== Step 9: Verification ===")
    print(f"Input FP32 sample: {input_fp32[0, 0, :3]}")
    print(f"Alpha dyadic (per-tensor): {alpha_dyadic_scalar}")
    print(f"Input INT8 sample: {input_int8[0, 0, :3]}")
    print(f"Linear output sample: {linear_output_int32[0, 0, :3]}")
    print(f"Residual sample: {residual_int32[0, 0, :3]}")
    print(f"Final output sample: {final_output_int32[0, 0, :3]}")
    
    return {
        'input_fp32': input_fp32,
        'scaling_factor_float': alpha_float_scalar,
        'scaling_factor_dyadic': alpha_dyadic_scalar,
        'dyadic_parameters': dyadic_results,
        'input_int8': input_int8,
        'weight_int8': weight_int8,
        'bias_int32': bias_int32,
        'linear_output_int32': linear_output_int32,
        'residual_int32': residual_int32,
        'final_output_int32': final_output_int32
    }

def save_all_results_to_txt(input_fp32, alpha_float_scalar, dyadic_results, alpha_dyadic_scalar,
                           input_int8, weight_int8, bias_int32, linear_output_int32, 
                           residual_int32, final_output_int32):
    """
    保存所有結果到 TXT 文件，供硬體 testbench 使用
    """
    print("Saving all results to TXT files...")
    
    B, N, C = input_fp32.shape
    
    # 1. 輸入 FP32 (198x192)
    input_fp32_np = input_fp32.squeeze(0).detach().cpu().numpy()  # Remove batch dimension
    np.savetxt('input_fp32.txt', input_fp32_np, fmt='%.8f', delimiter=',')
    print(f"✓ Saved input_fp32.txt: shape {input_fp32_np.shape}")
    
    # 2. Scaling factor (per-tensor, 只有一個值)
    orig_float, b, c, approx_float = dyadic_results[0]  # per-tensor 只有一組值
    scaling_data = np.array([[orig_float, b, c, approx_float]])
    
    np.savetxt('scaling_factor_per_tensor.txt', scaling_data, 
               fmt=['%.8f', '%d', '%d', '%.8f'], delimiter=',',
               header='original_float,b_int8,c_int8,approximated_float')
    print(f"✓ Saved scaling_factor_per_tensor.txt: per-tensor scaling factor")
    
    # 3. 權重 INT8 (192x192)
    weight_int8_np = weight_int8.detach().cpu().numpy()
    np.savetxt('weight_int8.txt', weight_int8_np, fmt='%d', delimiter=',')
    print(f"✓ Saved weight_int8.txt: shape {weight_int8_np.shape}")
    
    # 4. 偏置 INT32 (192,)
    bias_int32_np = bias_int32.detach().cpu().numpy()
    np.savetxt('bias_int32.txt', bias_int32_np.reshape(1, -1), fmt='%d', delimiter=',')
    print(f"✓ Saved bias_int32.txt: shape {bias_int32_np.shape}")
    
    # 5. 量化輸入 INT8 (198x192)
    input_int8_np = input_int8.squeeze(0).detach().cpu().numpy()
    np.savetxt('input_quantized_int8.txt', input_int8_np, fmt='%d', delimiter=',')
    print(f"✓ Saved input_quantized_int8.txt: shape {input_int8_np.shape}")
    
    # 6. 線性運算結果 INT32 (198x192)
    linear_output_np = linear_output_int32.squeeze(0).detach().cpu().numpy()
    np.savetxt('linear_output_int32.txt', linear_output_np, fmt='%d', delimiter=',')
    print(f"✓ Saved linear_output_int32.txt: shape {linear_output_np.shape}")
    
    # 7. Residual INT32 (198x192)
    residual_np = residual_int32.squeeze(0).detach().cpu().numpy()
    np.savetxt('residual_int32.txt', residual_np, fmt='%d', delimiter=',')
    print(f"✓ Saved residual_int32.txt: shape {residual_np.shape}")
    
    # 8. 最終結果 INT32 (198x192)
    final_output_np = final_output_int32.squeeze(0).detach().cpu().numpy()
    np.savetxt('final_output_int32.txt', final_output_np, fmt='%d', delimiter=',')
    print(f"✓ Saved final_output_int32.txt: shape {final_output_np.shape}")
    
    # 9. 創建 README 文件
    readme_content = """# Quantized Inference Test Data (Per-tensor Symmetric)

This directory contains test data for quantized inference with residual connection.
**Activation quantization: Per-tensor Symmetric**

## File Descriptions:

1. **input_fp32.txt** (198x192)
   - Original floating-point input data
   - Format: float32, comma-separated

2. **scaling_factor_per_tensor.txt** (1x4)
   - Single row: original_float, b_int8, c_int8, approximated_float
   - Per-tensor symmetric quantization scaling factor
   - Dyadic approximation: original ≈ b/2^c
   - b and c are int8 values

3. **weight_int8.txt** (192x192)
   - Quantized weights in int8 format
   - Range: [-127, 127]

4. **bias_int32.txt** (1x192)
   - Quantized bias in int32 format

5. **input_quantized_int8.txt** (198x192)
   - Quantized input in int8 format using per-tensor scaling
   - Range: [-127, 127]

6. **linear_output_int32.txt** (198x192)
   - Result of linear operation: input @ weight^T + bias
   - Format: int32

7. **residual_int32.txt** (198x192)
   - Residual data in int32 format
   - Simulates 16-bit range stored in int32

8. **final_output_int32.txt** (198x192)
   - Final result: linear_output + residual
   - Format: int32

## Quantization Details:
- **Activation**: Per-tensor symmetric quantization
- **Input scaling**: Single scaling factor for entire tensor
- **Formula**: quantized_value = round(float_value / scaling_factor)
- **Range**: [-127, 127] for int8

## Hardware Testbench Usage:
Load these files in your testbench to verify the quantized inference implementation.
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    print("✓ Saved README.md")

# 使用示例
if __name__ == "__main__":
    print("Running quantized inference with residual connection...")
    import torch

    checkpoint = torch.load("checkpoint.pth.tar", map_location='cpu')
    # 不使用 checkpoint，生成模擬數據
    results = quantized_inference_with_residual(checkpoint=checkpoint)

    print("\n" + "="*80)
    print("✅ Quantized inference with residual completed successfully!")
    print("📁 All test data saved to TXT files for hardware testbench.")
    print("📋 Check README.md for detailed file descriptions.")
    
    # 顯示一些統計信息
    print(f"\n📊 Data Statistics:")
    print(f"   • Input range: [{results['input_fp32'].min():.6f}, {results['input_fp32'].max():.6f}]")
    print(f"   • Quantized input range: [{results['input_int8'].min()}, {results['input_int8'].max()}]")
    print(f"   • Linear output range: [{results['linear_output_int32'].min()}, {results['linear_output_int32'].max()}]")
    print(f"   • Residual range: [{results['residual_int32'].min()}, {results['residual_int32'].max()}]")
    print(f"   • Final output range: [{results['final_output_int32'].min()}, {results['final_output_int32'].max()}]")