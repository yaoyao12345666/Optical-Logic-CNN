"""
数据加载器模块
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_loaders(batch_size=64, num_classes=4, data_dir='./data'):
    """
    加载 MNIST 数据集，并执行论文要求的预处理：
    1. Resize 到 9x9
    2. 二值化 (0/1)
    3. 仅选取前 num_classes 个类别 (默认 0,1,2,3)
    
    Args:
        batch_size: 批次大小
        num_classes: 使用的类别数量
        data_dir: 数据存储目录
        
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    transform = transforms.Compose([
        transforms.Resize((9, 9)),       # 关键：9x9
        transforms.ToTensor(),           # 转 Tensor [0, 1]
        transforms.Lambda(lambda x: (x > 0.5).float())  # 关键：二值化
    ])
    
    # 下载训练集
    train_dataset_full = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset_full = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    # 过滤类别 (例如只取 0, 1, 2, 3)
    def filter_dataset(dataset, classes):
        indices = [i for i, label in enumerate(dataset.targets) if label in classes]
        subset = torch.utils.data.Subset(dataset, indices)
        # 修改 target 使其从 0 开始连续 (0,1,2,3 而不是 0,1,2,3 的原标签)
        new_targets = [classes.index(int(dataset.targets[i])) for i in indices]
        
        # 创建一个包装类来处理标签映射
        class SubsetWithRemap(torch.utils.data.Dataset):
            def __init__(self, original_subset, new_targets_list):
                self.subset = original_subset
                self.new_targets = new_targets_list
            def __len__(self): return len(self.subset)
            def __getitem__(self, idx):
                img, _ = self.subset[idx]
                return img, self.new_targets[idx]
        
        return SubsetWithRemap(subset, new_targets)

    target_classes = list(range(num_classes))
    train_dataset = filter_dataset(train_dataset_full, target_classes)
    test_dataset = filter_dataset(test_dataset_full, target_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader


def get_mnist_loaders_full(batch_size=64, data_dir='./data'):
    """
    加载完整的 MNIST 数据集（10分类）
    
    Args:
        batch_size: 批次大小
        data_dir: 数据存储目录
        
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    transform = transforms.Compose([
        transforms.Resize((9, 9)),       # 关键：9x9
        transforms.ToTensor(),           # 转 Tensor [0, 1]
        transforms.Lambda(lambda x: (x > 0.5).float())  # 关键：二值化
    ])
    
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader