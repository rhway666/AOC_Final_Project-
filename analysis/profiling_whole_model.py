import torch
from transformers import DeiTModel
from torch.profiler import profile, record_function, ProfilerActivity

# 加載您之前存儲的 DeiT 模型
model_path = './deit_tiny_distilled-patch16-224'  # 確保這個路徑指向您保存的模型
model = DeiTModel.from_pretrained(model_path)

# 設置模型為評估模式
model.eval()

# 創建模擬的圖像數據 (batch_size=16, 每張圖片 224x224)
input_data = torch.randn(16, 3, 224, 224)  # 假設 batch_size = 16，圖像尺寸為 224x224

# 開始 profiling
with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
    with record_function("model_inference"):
        # 在不需要計算梯度的情況下進行推理
        with torch.no_grad():
            outputs = model(input_data)

# 顯示 profiling 結果，根據總 CPU 時間排序
print(prof.key_averages().table(sort_by="cpu_time_total"))
