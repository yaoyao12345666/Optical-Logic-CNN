"""
评估脚本
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import OLCNN
from src.utils import get_mnist_loaders, get_device, print_model_summary


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估光学逻辑卷积神经网络')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--device', type=str, default='auto', help='计算设备 (auto/cpu/cuda)')
    parser.add_argument('--save_cm', action='store_true', help='保存混淆矩阵')
    parser.add_argument('--output_dir', type=str, default='results', help='结果保存目录')
    
    return parser.parse_args()


def evaluate_model(model, test_loader, criterion, device):
    """
    评估模型
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备
        
    Returns:
        results: 包含评估结果的字典
    """
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 统计
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 保存预测结果
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(torch.softmax(output, dim=1).cpu().numpy())
    
    # 计算平均指标
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    results = {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }
    
    return results


def plot_confusion_matrix(cm, classes, save_path):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        classes: 类别名称
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存到: {save_path}")


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    train_loader, test_loader = get_mnist_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 创建模型
    print("\n创建模型...")
    model = OLCNN().to(device)
    
    # 加载检查点
    print(f"加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 'Unknown')
    best_acc = checkpoint.get('best_acc', 'Unknown')
    print(f"检查点信息: Epoch {epoch}, 最佳准确率: {best_acc:.2f}%")
    
    print_model_summary(model)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 评估模型
    print("\n开始评估...")
    print("=" * 80)
    
    results = evaluate_model(model, test_loader, criterion, device)
    
    # 打印结果
    print(f"\n测试损失: {results['test_loss']:.4f}")
    print(f"测试准确率: {results['accuracy']:.2f}%")
    print(f"正确分类: {int(results['accuracy'] * len(test_loader.dataset) / 100)} / {len(test_loader.dataset)}")
    
    # 分类报告
    print("\n分类报告:")
    print("=" * 80)
    target_names = [f'Class {i}' for i in range(4)]
    print(classification_report(results['labels'], results['predictions'], 
                                target_names=target_names, digits=4))
    
    # 混淆矩阵
    cm = confusion_matrix(results['labels'], results['predictions'])
    print("\n混淆矩阵:")
    print("=" * 80)
    print(cm)
    
    # 保存混淆矩阵
    if args.save_cm:
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(cm, target_names, cm_path)
    
    # 保存结果
    results_path = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("模型评估结果\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"检查点: {args.checkpoint}\n")
        f.write(f"测试集大小: {len(test_loader.dataset)}\n")
        f.write(f"测试损失: {results['test_loss']:.4f}\n")
        f.write(f"测试准确率: {results['accuracy']:.2f}%\n")
        f.write(f"正确分类: {int(results['accuracy'] * len(test_loader.dataset) / 100)} / {len(test_loader.dataset)}\n\n")
        
        f.write("分类报告:\n")
        f.write("-" * 80 + "\n")
        f.write(classification_report(results['labels'], results['predictions'], 
                                     target_names=target_names, digits=4))
        
        f.write("\n混淆矩阵:\n")
        f.write("-" * 80 + "\n")
        np.savetxt(f, cm, fmt='%d')
    
    print(f"\n评估结果已保存到: {results_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()