# import torch
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torch.utils.data import DataLoader
# from transformers import DeiTModel
# import os


# def load_model(device='cpu', model_path='./deit_tiny_distilled-patch16-224'):
#     """載入DeiT tiny模型到指定設備"""
#     print(f"載入DeiT tiny模型到{device}設備...")
#     # 從指定路徑加載已保存的模型
#     model = DeiTModel.from_pretrained(model_path)
#     model.eval()  # 設置為評估模式
#     model = model.to(device)
#     return model

# def load_imagenet_data(imagenet_path, batch_size=16, num_workers=4, device='cpu'):
#     """載入ImageNet數據集
    
#     參數:
#         imagenet_path: ImageNet數據集的根目錄路徑
#         batch_size: 批處理大小
#         num_workers: 數據載入的工作進程數
#         device: 計算設備 ('cpu' 或 'cuda')
    
#     返回:
#         data_loader: ImageNet驗證集的數據加載器
#     """
#     print(f"載入ImageNet數據，batch_size={batch_size}...")
    
#     # 數據預處理和轉換
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                              std=[0.229, 0.224, 0.225])
#     ])
    
#     # 檢查指定路徑是否存在
#     if not os.path.exists(imagenet_path):
#         raise FileNotFoundError(f"找不到ImageNet數據路徑: {imagenet_path}")
    
#     # 檢查驗證集路徑是否存在
#     val_dir = os.path.join(imagenet_path, 'val')
#     if not os.path.exists(val_dir):
#         raise FileNotFoundError(f"找不到ImageNet驗證集路徑: {val_dir}")
    
#     # 載入ImageNet驗證集
#     val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    
#     # 創建數據加載器
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=(device == 'cuda')  # 如果使用GPU，啟用pin_memory加速數據傳輸
#     )
    
#     print(f"已載入{len(val_dataset)}張驗證圖像")
#     return val_loader

# def load_single_batch(data_loader, device='cpu'):
#     """從數據加載器中獲取單個批次"""
#     data_iter = iter(data_loader)
#     images, labels = next(data_iter)
#     images = images.to(device)
#     labels = labels.to(device)
#     return images, labels

# def create_dummy_batch(batch_size=16, device='cpu'):
#     """創建一個模擬的ImageNet批次（當無法訪問實際數據時使用）"""
#     print(f"創建模擬批次，batch_size={batch_size}...")
#     dummy_images = torch.randn(batch_size, 3, 224, 224, device=device)
#     return dummy_images

# def load_data_and_model(config):
#     """根據配置載入模型和數據
    
#     參數:
#         config: 包含以下鍵的字典
#             - use_real_data: 是否使用真實數據
#             - imagenet_path: ImageNet數據路徑（僅當use_real_data=True時需要）
#             - batch_size: 批處理大小
#             - device: 計算設備 ('cpu' 或 'cuda')
    
#     返回:
#         model: 載入的DeiT模型
#         images: 一個批次的圖像數據
#     """
#     # 設置默認值
#     config.setdefault('use_real_data', False)
#     config.setdefault('batch_size', 16)
#     config.setdefault('device', 'cpu')
#     config.setdefault('num_workers', 4)
    
#     # 檢查是否可用CUDA
#     if config['device'] == 'cuda' and not torch.cuda.is_available():
#         print("警告: CUDA不可用，改用CPU")
#         config['device'] = 'cpu'
    
#     # 載入模型
#     model = load_model(config['device'])
    
#     # 載入數據
#     if config['use_real_data']:
#         if 'imagenet_path' not in config:
#             raise ValueError("使用真實數據時必須提供ImageNet路徑")
        
#         data_loader = load_imagenet_data(
#             config['imagenet_path'],
#             config['batch_size'],
#             config['num_workers'],
#             config['device']
#         )
#         images, _ = load_single_batch(data_loader, config['device'])
#     else:
#         images = create_dummy_batch(config['batch_size'], config['device'])
    
#     return model, images

# # 使用示例
# if __name__ == "__main__":
#     # 配置
#     config = {
#         'use_real_data': False,  # 改為True以使用真實ImageNet數據
#         'imagenet_path': '../../../../storage/share/imagenet-1000-mini',  # 指定ImageNet數據集路徑
#         'batch_size': 16,
#         'device': 'cpu',  # 或 'cuda' 使用GPU
#         'num_workers': 4
#     }
    
#     # 載入模型和數據
#     model, images = load_data_and_model(config)
    
#     # 顯示信息
#     print(f"模型類型: {type(model).__name__}")
#     print(f"輸入數據形狀: {images.shape}")
    
#     # 進行一次前向傳播測試
#     with torch.no_grad():
#         outputs = model(images)
    
#     print(f"輸出形狀: {outputs.shape}")

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from transformers import DeiTModel
import os


def load_model(device='cpu', model_path='./deit_tiny_distilled-patch16-224'):
    """載入DeiT tiny模型到指定設備"""
    print(f"載入DeiT tiny模型到{device}設備...")
    # 從指定路徑加載已保存的模型
    model = DeiTModel.from_pretrained(model_path)
    model.eval()  # 設置為評估模式
    model = model.to(device)
    return model

def load_imagenet_data(imagenet_path, batch_size=16, num_workers=4, device='cpu'):
    """載入ImageNet數據集
    
    參數:
        imagenet_path: ImageNet數據集的根目錄路徑
        batch_size: 批處理大小
        num_workers: 數據載入的工作進程數
        device: 計算設備 ('cpu' 或 'cuda')
    
    返回:
        data_loader: ImageNet驗證集的數據加載器
    """
    print(f"載入ImageNet數據，batch_size={batch_size}...")
    
    # 數據預處理和轉換
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 檢查指定路徑是否存在
    if not os.path.exists(imagenet_path):
        raise FileNotFoundError(f"找不到ImageNet數據路徑: {imagenet_path}")
    
    # 檢查驗證集路徑是否存在
    val_dir = os.path.join(imagenet_path, 'val')
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"找不到ImageNet驗證集路徑: {val_dir}")
    
    # 載入ImageNet驗證集
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    
    # 創建數據加載器
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == 'cuda')  # 如果使用GPU，啟用pin_memory加速數據傳輸
    )
    
    print(f"已載入{len(val_dataset)}張驗證圖像")
    return val_loader

def load_single_batch(data_loader, device='cpu'):
    """從數據加載器中獲取單個批次"""
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    images = images.to(device)
    labels = labels.to(device)
    return images, labels

def create_dummy_batch(batch_size=16, device='cpu'):
    """創建一個模擬的ImageNet批次（當無法訪問實際數據時使用）"""
    print(f"創建模擬批次，batch_size={batch_size}...")
    dummy_images = torch.randn(batch_size, 3, 224, 224, device=device)
    return dummy_images

def load_data_and_model(config):
    """根據配置載入模型和數據
    
    參數:
        config: 包含以下鍵的字典
            - use_real_data: 是否使用真實數據
            - imagenet_path: ImageNet數據路徑（僅當use_real_data=True時需要）
            - batch_size: 批處理大小
            - device: 計算設備 ('cpu' 或 'cuda')
    
    返回:
        model: 載入的DeiT模型
        images: 一個批次的圖像數據
    """
    # 設置默認值
    config.setdefault('use_real_data', False)
    config.setdefault('batch_size', 16)
    config.setdefault('device', 'cpu')
    config.setdefault('num_workers', 4)
    
    # 檢查是否可用CUDA
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，改用CPU")
        config['device'] = 'cpu'
    
    # 載入模型
    model = load_model(config['device'])
    
    # 載入數據
    if config['use_real_data']:
        if 'imagenet_path' not in config:
            raise ValueError("使用真實數據時必須提供ImageNet路徑")
        
        data_loader = load_imagenet_data(
            config['imagenet_path'],
            config['batch_size'],
            config['num_workers'],
            config['device']
        )
        images, _ = load_single_batch(data_loader, config['device'])
    else:
        images = create_dummy_batch(config['batch_size'], config['device'])
    
    return model, images

# 使用示例
if __name__ == "__main__":
    # 配置
    config = {
        'use_real_data': False,  # 改為True以使用真實ImageNet數據
        'imagenet_path': '../../../../storage/share/imagenet-1000-mini',  # 指定ImageNet數據集路徑
        'batch_size': 16,
        'device': 'cpu',  # 或 'cuda' 使用GPU
        'num_workers': 4
    }
    
    # 載入模型和數據
    model, images = load_data_and_model(config)
    
    # 顯示信息
    print(f"模型類型: {type(model).__name__}")
    print(f"輸入數據形狀: {images.shape}")
    
    # 進行一次前向傳播測試
    with torch.no_grad():
        outputs = model(images)
    
    # 顯示 'last_hidden_state' 的形狀
    print(f"輸出形狀: {outputs.last_hidden_state.shape}")
