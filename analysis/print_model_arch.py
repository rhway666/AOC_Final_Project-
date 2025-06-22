from transformers import DeiTModel
import torch

# 加載預訓練的 DeiT 模型
model = DeiTModel.from_pretrained("facebook/deit-tiny-distilled-patch16-224")
print(model)
model.save_pretrained("./deit_tiny_distilled-patch16-224")
