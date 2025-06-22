import torch
from transformers import DeiTModel, DeiTConfig
from collections import defaultdict

# === 初始化模型與輸入 ===
model_path = './deit_tiny_distilled-patch16-224'  # 確保這個路徑指向您保存的模型
model = DeiTModel.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dummy_input = torch.randn(1, 3, 224, 224).to(device)  # 一張標準 224x224 的圖片

# === hook 註冊 ===
activation_info = defaultdict(list)

def hook_fn(name):
    def forward_hook(module, input, output):
        input_shape = tuple(input[0].shape) if input else None
        output_shape = tuple(output.shape) if isinstance(output, torch.Tensor) else None
        weight_shape = tuple(module.weight.shape) if hasattr(module, 'weight') else None
        activation_info[name].append({
            'input': input_shape,
            'output': output_shape,
            'weight': weight_shape
        })
    return forward_hook

# 只 hook encoder 中的 Linear / Conv2d / LayerNorm 層
for name, module in model.named_modules():
    if name.startswith("encoder.layer") and isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm)):
        module.register_forward_hook(hook_fn(name))

# === 前向傳遞觸發 hook ===
with torch.no_grad():
    model(dummy_input)

# === 印出所有紀錄的形狀資訊 ===
for name, infos in activation_info.items():
    for i, info in enumerate(infos):
        print(f"{name} | in: {info['input']}, out: {info['output']}, weight: {info['weight']}")
