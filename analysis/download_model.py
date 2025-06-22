import os
from transformers import DeiTForImageClassification

def download_deit_model(model_name="facebook/deit-tiny-distilled-patch16-224", save_path="./deit_tiny_distilled-patch16-224"):
    """
    下載 DeiT 模型並保存到指定目錄。
    
    參數:
        model_name (str): Hugging Face Model Hub 上的模型名稱。
        save_path (str): 模型保存的本地目錄路徑。
    """
    # 檢查目錄是否存在，若不存在則創建
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"已創建目錄: {save_path}")
    else:
        print(f"目錄已存在: {save_path}")

    # 檢查目錄是否已包含模型檔案
    config_file = os.path.join(save_path, "config.json")
    model_file = os.path.join(save_path, "model.safetensors")
    if os.path.exists(config_file) and os.path.exists(model_file):
        print("模型檔案已存在，跳過下載。")
        return

    try:
        # 從 Hugging Face 下載模型
        print(f"正在下載模型: {model_name}")
        model = DeiTForImageClassification.from_pretrained(model_name)
        # 保存模型到指定目錄
        model.save_pretrained(save_path)
        print(f"模型已成功下載並保存至: {save_path}")
    except Exception as e:
        print(f"下載模型時發生錯誤: {e}")
        print("請檢查網路連接或模型名稱是否正確。")

if __name__ == "__main__":
    download_deit_model()
