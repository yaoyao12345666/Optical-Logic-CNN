"""
通用工具函数
"""

import torch
import random
import numpy as np


def set_seed(seed=42):
    """
    设置随机种子，保证实验可复现
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        total_params: 总参数数
        trainable_params: 可训练参数数
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_summary(model):
    """
    打印模型摘要信息
    
    Args:
        model: PyTorch模型
    """
    total_params, trainable_params = count_parameters(model)
    
    print("=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)
    
    # 打印每层的参数信息
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape} = {param.numel():,} parameters")


def get_device():
    """
    获取计算设备
    
    Returns:
        device: torch.device对象
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_time(seconds):
    """
    格式化时间
    
    Args:
        seconds: 秒数
        
    Returns:
        formatted_time: 格式化后的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"