import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import DeiTModel
import pandas as pd
import matplotlib.pyplot as plt

def analyze_single_encoder_layer(model_name="facebook/deit-base-distilled-patch16-224", layer_idx=0, batch_size=8):
    """深入分析單個DeiT Encoder層內部的計算負擔分布"""
    print(f"\n正在分析DeiT Encoder第{layer_idx}層內部組件...")
    
    # 加載模型
    model_path = './deit_tiny_distilled-patch16-224'  # 確保這個路徑指向您保存的模型
    model = DeiTModel.from_pretrained(model_path)
    model.eval()
    
    # 獲取指定的encoder層
    encoder_layer = model.encoder.layer[layer_idx]
    
    # 設置設備
    device = torch.device("cpu")
    model.to(device)
    
    # 創建輸入數據 - 對於DeiT-base模型，使用標準配置
    seq_length = 197  # 對於224x224圖像，使用16x16的patch size (14x14 + class token + distillation token)
    hidden_size = 768  # DeiT-base的隱藏層大小
    
    # 創建模擬輸入
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # 獲取embeddings輸出作為encoder層的輸入
    with torch.no_grad():
        hidden_states = model.embeddings(dummy_input)
    
    # 預熱
    print("預熱模型...")
    for _ in range(10):
        with torch.no_grad():
            _ = encoder_layer(hidden_states)
    
    # 各個組件列表
    components = {
        "Self-Attention - Query": encoder_layer.attention.attention.query,
        "Self-Attention - Key": encoder_layer.attention.attention.key,
        "Self-Attention - Value": encoder_layer.attention.attention.value,
        "Self-Attention - Output": encoder_layer.attention.output.dense,
        "FFN - Intermediate": encoder_layer.intermediate.dense,
        "FFN - Output": encoder_layer.output.dense,
        "LayerNorm - Before": encoder_layer.layernorm_before,
        "LayerNorm - After": encoder_layer.layernorm_after,
        "GELU Activation": encoder_layer.intermediate.intermediate_act_fn
    }
    
    # 定義每個組件的測量函數
    def measure_component(name, component_fn, input_tensor):
        with record_function(name):
            with torch.no_grad():
                output = component_fn(input_tensor)
        return output
    
    # 為output.dense方法定義一個特殊的包裝器，因為它需要兩個參數
    def measure_output_component(name, input_tensor, hidden_states):
        with record_function(name):
            with torch.no_grad():
                # 這裡我們直接訪問DeiTOutput.forward方法，並傳入兩個必要的參數
                output = encoder_layer.output(input_tensor, hidden_states)
        return output
    
    # 使用profiler分析
    print("開始分析單個Encoder層內部組件...")
    try:
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            # 整個層的前向傳播
            with record_function("Complete Encoder Layer"):
                with torch.no_grad():
                    _ = encoder_layer(hidden_states)
            
            # 分開測量自注意力機制的部分
            with record_function("Self-Attention Mechanism (Complete)"):
                with torch.no_grad():
                    residual = hidden_states
                    attention_input = encoder_layer.layernorm_before(hidden_states)
                    attention_output = encoder_layer.attention(attention_input)
                    hidden_states = residual + attention_output[0]
            
            # 分開測量前饋網絡的部分
            with record_function("Feed-Forward Network (Complete)"):
                with torch.no_grad():
                    residual = hidden_states
                    ffn_input = encoder_layer.layernorm_after(hidden_states)
                    intermediate_output = encoder_layer.intermediate(ffn_input)
                    ffn_output = encoder_layer.output(intermediate_output, ffn_input)  # 修正：傳入兩個參數
                    hidden_states = residual + ffn_output
            
            # 測量各個細分組件
            # 1. 自注意力的Query、Key、Value投影
            h = encoder_layer.layernorm_before(hidden_states)
            q = measure_component("Self-Attention - Query Projection", encoder_layer.attention.attention.query, h)
            k = measure_component("Self-Attention - Key Projection", encoder_layer.attention.attention.key, h)
            v = measure_component("Self-Attention - Value Projection", encoder_layer.attention.attention.value, h)
            
            # 2. 注意力分數計算和加權
            with record_function("Self-Attention - Scores & Context"):
                with torch.no_grad():
                    # 模擬注意力機制的核心操作
                    batch_size, seq_len, hidden_size = h.shape
                    head_size = hidden_size // encoder_layer.attention.attention.num_attention_heads
                    q = q.view(batch_size, seq_len, encoder_layer.attention.attention.num_attention_heads, head_size).transpose(1, 2)
                    k = k.view(batch_size, seq_len, encoder_layer.attention.attention.num_attention_heads, head_size).transpose(1, 2)
                    v = v.view(batch_size, seq_len, encoder_layer.attention.attention.num_attention_heads, head_size).transpose(1, 2)
                    
                    # 計算注意力分數
                    scores = torch.matmul(q, k.transpose(-1, -2))
                    scores = scores / (head_size ** 0.5)
                    
                    # softmax
                    attention_weights = torch.nn.functional.softmax(scores, dim=-1)
                    
                    # 加權上下文
                    context = torch.matmul(attention_weights, v)
                    context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
            
            # 3. 自注意力的輸出投影
            attention_output = measure_component("Self-Attention - Output Projection", 
                                            encoder_layer.attention.output.dense, context)
            
            # 4. 前饋網絡的中間層
            h = encoder_layer.layernorm_after(hidden_states)
            intermediate = measure_component("FFN - Intermediate Layer", encoder_layer.intermediate.dense, h)
            
            # 5. GELU激活
            activated = measure_component("GELU Activation", encoder_layer.intermediate.intermediate_act_fn, intermediate)
            
            # 6. 前饋網絡的輸出投影 - 使用特殊的測量方法，因為需要兩個參數
            with record_function("FFN - Output Layer"):
                with torch.no_grad():
                    ffn_output = encoder_layer.output(activated, h)  # 注意：傳入兩個參數
            
            # 7. LayerNorm操作
            _ = measure_component("LayerNorm - Before", encoder_layer.layernorm_before, hidden_states)
            _ = measure_component("LayerNorm - After", encoder_layer.layernorm_after, hidden_states)

    except Exception as e:
        print(f"分析過程中出現錯誤: {e}")
        return None

    # 分析結果
    print("\n單個Encoder層內部組件性能分析:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
    
    # 處理結果為DataFrame
    events = []
    for func in prof.key_averages():
        events.append({
            "Component": func.key,
            "CPU Time (ms)": func.cpu_time_total / 1000,
            "CPU %": func.cpu_time_total / prof.total_average().cpu_time_total * 100,
            "Memory Usage (KB)": func.self_cpu_memory_usage / 1024 if hasattr(func, "self_cpu_memory_usage") else 0
        })
    
    df = pd.DataFrame(events)
    df.sort_values(by="CPU Time (ms)", ascending=False, inplace=True)
    
    # 顯示主要組件
    main_components = df[df["Component"].isin([
        "Complete Encoder Layer", 
        "Self-Attention Mechanism (Complete)", 
        "Feed-Forward Network (Complete)"
    ])]
    
    print("\n主要組件比較:")
    print(main_components)
    
    # 顯示詳細組件
    detail_components = df[~df["Component"].isin([
        "Complete Encoder Layer", 
        "Self-Attention Mechanism (Complete)", 
        "Feed-Forward Network (Complete)"
    ])]
    
    print("\n詳細組件分析:")
    print(detail_components)
    
    # 保存結果
    df.to_csv(f"deit_encoder_layer_{layer_idx}_profiling.csv", index=False)
    
    # 創建可視化圖表
    create_visualizations(df, layer_idx)
    
    return df

def create_visualizations(df, layer_idx):
    """創建可視化圖表"""
    plt.figure(figsize=(12, 6))
    
    # 過濾出詳細組件並排序
    detail_df = df[~df["Component"].isin([
        "Complete Encoder Layer", 
        "Self-Attention Mechanism (Complete)", 
        "Feed-Forward Network (Complete)"
    ])].sort_values(by="CPU Time (ms)", ascending=False).head(10)
    
    # 繪製條形圖
    plt.barh(detail_df["Component"], detail_df["CPU Time (ms)"], color="skyblue")
    plt.xlabel("CPU Time (ms)")
    plt.ylabel("Component")
    plt.title(f"DeiT Encoder Layer {layer_idx} - Component CPU Time")
    plt.tight_layout()
    plt.savefig(f"deit_encoder_layer_{layer_idx}_components.png")
    print(f"圖表已保存為 deit_encoder_layer_{layer_idx}_components.png")
    
    # 創建餅圖
    plt.figure(figsize=(10, 10))
    
    # 合併小組件為"其他"
    threshold = 5.0  # 小於5%的組件合併為"其他"
    pie_df = detail_df.copy()
    pie_df.loc[pie_df["CPU %"] < threshold, "Component"] = "Other Components"
    pie_df = pie_df.groupby("Component").sum().reset_index()
    
    plt.pie(pie_df["CPU %"], labels=pie_df["Component"], autopct="%1.1f%%", 
            shadow=True, startangle=90)
    plt.axis("equal")
    plt.title(f"DeiT Encoder Layer {layer_idx} - CPU Time Distribution")
    plt.savefig(f"deit_encoder_layer_{layer_idx}_distribution.png")
    print(f"分佈圖已保存為 deit_encoder_layer_{layer_idx}_distribution.png")

def analyze_all_encoder_layers():
    """分析所有encoder層並比較結果"""
    model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
    num_layers = len(model.encoder.layer)
    
    all_results = []
    
    # 分析每一層
    for i in range(num_layers):
        print(f"\n===== 分析 Encoder Layer {i} =====")
        df = analyze_single_encoder_layer(layer_idx=i)
        
        # 提取主要組件數據
        layer_data = {
            "Layer": i,
            "Total Time (ms)": df[df["Component"] == "Complete Encoder Layer"]["CPU Time (ms)"].values[0],
            "Self-Attention (ms)": df[df["Component"] == "Self-Attention Mechanism (Complete)"]["CPU Time (ms)"].values[0],
            "Feed-Forward (ms)": df[df["Component"] == "Feed-Forward Network (Complete)"]["CPU Time (ms)"].values[0]
        }
        
        # 提取細分組件
        for component in df["Component"]:
            if component not in ["Complete Encoder Layer", "Self-Attention Mechanism (Complete)", "Feed-Forward Network (Complete)"]:
                layer_data[component] = df[df["Component"] == component]["CPU Time (ms)"].values[0]
        
        all_results.append(layer_data)
    
    # 創建整體比較DataFrame
    all_layers_df = pd.DataFrame(all_results)
    
    # 保存結果
    all_layers_df.to_csv("deit_all_encoder_layers_comparison.csv", index=False)
    
    # 創建可視化比較
    plt.figure(figsize=(12, 8))
    plt.plot(all_layers_df["Layer"], all_layers_df["Self-Attention (ms)"], marker="o", label="Self-Attention")
    plt.plot(all_layers_df["Layer"], all_layers_df["Feed-Forward (ms)"], marker="s", label="Feed-Forward")
    plt.plot(all_layers_df["Layer"], all_layers_df["Total Time (ms)"], marker="^", label="Total Layer Time")
    
    plt.xlabel("Encoder Layer")
    plt.ylabel("CPU Time (ms)")
    plt.title("Performance Comparison Across DeiT Encoder Layers")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig("deit_all_layers_comparison.png")
    
    print("\n所有層的比較已保存為 deit_all_encoder_layers_comparison.csv")
    print("比較圖表已保存為 deit_all_layers_comparison.png")
    
    return all_layers_df

if __name__ == "__main__":
    try:
        # 分析單個層
        layer_to_analyze = 0  # 分析第一個encoder層，可以根據需要更改
        analyze_single_encoder_layer(layer_idx=layer_to_analyze)
        
        # 如需分析所有層並比較，取消下面的註釋
        # analyze_all_encoder_layers()
        
    except Exception as e:
        print(f"分析時發生錯誤: {e}")