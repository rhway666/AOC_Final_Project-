
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import DeiTModel
import time
import pandas as pd

def profile_deit_model():
    # 加載DeiT模型
    
    model_path = './deit_tiny_distilled-patch16-224'  # 確保這個路徑指向您保存的模型
    model = DeiTModel.from_pretrained(model_path)
    model.eval()  # 設置為評估模式
    
       # 創建隨機輸入數據
    batch_size = 8
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # 將模型和輸入移至GPU（如果可用）
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)
    dummy_input = dummy_input.to(device)
    
    # 預熱模型
    print("預熱模型...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # 使用PyTorch Profiler進行性能分析
    print("開始分析...")
    with profile(
        # activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # 記錄整個模型的前向傳播
        with record_function("Model: DeiT Complete"):
            with torch.no_grad():
                _ = model(dummy_input)
                
        # 專門記錄embeddings的性能
        with record_function("Module: Embeddings"):
            with torch.no_grad():
                _ = model.embeddings(dummy_input)
        
        # 專門記錄encoder的性能
        with record_function("Module: Encoder"):
            with torch.no_grad():
                hidden_states = model.embeddings(dummy_input)
                # 檢查模型文檔確認正確的參數
                _ = model.encoder(hidden_states)
        
        # 單獨分析每個encoder層
        hidden_states = model.embeddings(dummy_input)
        
        for i in range(len(model.encoder.layer)):
            with record_function(f"Encoder Layer {i}"):
                with torch.no_grad():
                    # 使用正確的參數呼叫 encoder layer
                    layer_outputs = model.encoder.layer[i](hidden_states)
                    # 如果返回的是元組，則獲取第一個元素
                    if isinstance(layer_outputs, tuple):
                        hidden_states = layer_outputs[0]
                    else:
                        hidden_states = layer_outputs
        
        # 專門記錄layernorm的性能
        with record_function("Module: LayerNorm"):
            with torch.no_grad():
                _ = model.layernorm(hidden_states)
        
        # 專門記錄pooler的性能
        with record_function("Module: Pooler"):
            with torch.no_grad():
                _ = model.pooler(hidden_states)
    
    # 分析結果
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    # 導出詳細分析結果為Chrome trace格式
    prof.export_chrome_trace("deit_profile_trace.json")
    
    # 將結果轉換為pandas DataFrame便於分析
    events = []
    for func in prof.key_averages():
        events.append({
            "Name": func.key,
            "CPU Time (ms)": func.cpu_time_total / 1000,  # 轉換為毫秒
            # "CUDA Time (ms)": func.cuda_time_total / 1000 if hasattr(func, "cuda_time_total") else 0,
            "Memory Usage (KB)": func.self_cpu_memory_usage / 1024,  # 轉換為KB
            "CPU %": func.cpu_time_total / prof.total_average().cpu_time_total * 100
        })
    
    df = pd.DataFrame(events)
    df.sort_values(by="CPU Time (ms)", ascending=False, inplace=True)
    
    # 顯示按模塊分組的統計數據
    print("\n各模塊性能摘要:")
    modules = df[df["Name"].str.contains("Module:|Encoder Layer")]
    print(modules)
    
    # 保存分析結果為CSV
    df.to_csv("deit_profiling_results.csv", index=False)
    print("\n分析結果已保存為 deit_profiling_results.csv 和 deit_profile_trace.json")
    
    # 檢查是否有bottleneck
    print("\nBottleneck分析:")
    encoder_time = modules[modules["Name"] == "Module: Encoder"]["CPU Time (ms)"].values[0]
    total_time = df[df["Name"] == "Model: DeiT Complete"]["CPU Time (ms)"].values[0]
    
    print(f"Encoder佔總推理時間的 {encoder_time/total_time*100:.2f}%")
    
    # 分析各個層的性能
    encoder_layers = modules[modules["Name"].str.contains("Encoder Layer")]
    if not encoder_layers.empty:
        avg_layer_time = encoder_layers["CPU Time (ms)"].mean()
        max_layer_time = encoder_layers["CPU Time (ms)"].max()
        min_layer_time = encoder_layers["CPU Time (ms)"].min()
        print(f"Encoder層平均時間: {avg_layer_time:.2f} ms")
        print(f"Encoder層最長時間: {max_layer_time:.2f} ms")
        print(f"Encoder層最短時間: {min_layer_time:.2f} ms")
    
    return df

def analyze_encoder_components(model):
    """分析encoder中的各子組件性能"""
    print("\n分析DeiTEncoder內部組件...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    seq_length = 197  # 對於224x224的圖像，使用16x16的patch size
    hidden_size = 192
    
    # 創建encoder的輸入
    hidden_states = torch.randn(batch_size, seq_length, hidden_size).to(device)
    
    # 對encoder的內部組件進行分離分析
    results = {}
    
    # 獲取第一個encoder層作為示例
    layer = model.encoder.layer[0]
    
    # 預熱
    for _ in range(5):
        with torch.no_grad():
            _ = layer(hidden_states)
    
    # 分析層的性能
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof:
        # 完整層
        with record_function("Complete Layer"):
            with torch.no_grad():
                layer_output = layer(hidden_states)
        
        # Self-Attention
        with record_function("Self-Attention"):
            with torch.no_grad():
                residual = hidden_states
                hidden_states = layer.layernorm_before(hidden_states)
                hidden_states = layer.attention(hidden_states)[0]
                hidden_states = residual + hidden_states
                
        # FFN (Feed Forward Network)
        with record_function("FFN"):
            with torch.no_grad():
                residual = hidden_states
                hidden_states = layer.layernorm_after(hidden_states)
                hidden_states = layer.intermediate(hidden_states)
                hidden_states = layer.output(hidden_states)
                hidden_states = residual + hidden_states
    
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    # 將結果轉換為pandas DataFrame
    events = []
    for func in prof.key_averages():
        events.append({
            "Name": func.key,
            "CPU Time (ms)": func.cpu_time_total / 1000,
            "CUDA Time (ms)": func.cuda_time_total / 1000 if hasattr(func, "cuda_time_total") else 0,
        })
    
    df = pd.DataFrame(events)
    df.sort_values(by="CPU Time (ms)", ascending=False, inplace=True)
    
    # 顯示內部組件的統計數據
    print("\nEncoder層內部組件性能:")
    print(df)
    
    return df

if __name__ == "__main__":
    try:
        # 主要性能分析
        df = profile_deit_model()
        
        # 獲取模型用於更深入分析
        model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        
        # 深入分析encoder組件
        encoder_components_df = analyze_encoder_components(model)
        
        print("\n性能分析完成!")
        
    except Exception as e:
        print(f"分析時發生錯誤: {e}")