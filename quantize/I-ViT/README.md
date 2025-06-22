# Quantized Inference Test Data (Per-tensor Symmetric)

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
