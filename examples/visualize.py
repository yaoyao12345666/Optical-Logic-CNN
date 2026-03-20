"""
可视化脚本
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import OLCNN
from src.utils import get_mnist_loaders, get_device


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='可视化光学逻辑卷积神经网络')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--num_samples', type=int, default=10, help='可视化的样本数量')
    parser.add_argument('--device', type=str, default='auto', help='计算设备 (auto/cpu/cuda)')
    parser.add_argument('--output_dir', type=str, default='results', help='结果保存目录')
    
    return parser.parse_args()


def visualize_samples(model, test_loader, device, num_samples, output_dir):
    """
    可视化样本和预测结果
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        num_samples: 可视化样本数量
        output_dir: 输出目录
    """
    model.eval()
    
    # 获取样本
    samples = []
    labels = []
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, predicted = output.max(1)
            
            samples.extend(data.cpu().numpy())
            labels.extend(target.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            
            if len(samples) >= num_samples:
                break
    
    samples = np.array(samples[:num_samples])
    labels = np.array(labels[:num_samples])
    predictions = np.array(predictions[:num_samples])
    probabilities = np.array(probabilities[:num_samples])
    
    # 绘制图像
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 6))
    
    for i in range(num_samples):
        # 原始图像
        ax = axes[0, i]
        ax.imshow(samples[i, 0], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'True: {labels[i]}\nPred: {predictions[i]}', fontsize=10)
        ax.axis('off')
        
        # 预测概率
        ax = axes[1, i]
        bars = ax.bar(range(4), probabilities[i], color='skyblue')
        bars[predictions[i]].set_color('red')
        ax.set_ylim([0, 1])
        ax.set_xlabel('Class', fontsize=8)
        ax.set_ylabel('Probability', fontsize=8)
        ax.set_xticks(range(4))
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # 添加概率值
        for j, prob in enumerate(probabilities[i]):
            ax.text(j, prob + 0.02, f'{prob:.2f}', 
                   ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'sample_predictions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"样本预测可视化已保存到: {save_path}")


def visualize_lut_weights(model, output_dir):
    """
    可视化LUT权重
    
    Args:
        model: 模型
        output_dir: 输出目录
    """
    # 获取第一层逻辑卷积层的LUT权重
    logic_conv = model.logic_conv
    lut_params = logic_conv.logic_units[0].lut_params.data.cpu().numpy()
    
    # 计算每个逻辑核的平均激活概率
    num_kernels = len(logic_conv.logic_units)
    lut_probs = torch.sigmoid(logic_conv.logic_units[0].lut_params).data.cpu().numpy()
    
    # 绘制LUT权重热图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 原始参数
    ax = axes[0]
    im = ax.imshow(lut_params, cmap='RdBu_r', aspect='auto')
    ax.set_title('LUT Parameters (Raw)', fontsize=14)
    ax.set_xlabel('Input Combination', fontsize=12)
    ax.set_ylabel('Logic Kernel Index', fontsize=12)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    
    # 激活概率
    ax = axes[1]
    im = ax.imshow(lut_probs, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax.set_title('LUT Activation Probabilities', fontsize=14)
    ax.set_xlabel('Input Combination', fontsize=12)
    ax.set_ylabel('Logic Kernel Index', fontsize=12)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'lut_weights.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"LUT权重可视化已保存到: {save_path}")


def visualize_feature_maps(model, test_loader, device, output_dir):
    """
    可视化特征图
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        output_dir: 输出目录
    """
    model.eval()
    
    # 获取一个样本
    data, target = next(iter(test_loader))
    sample = data[0:1].to(device)
    label = target[0].item()
    
    # 获取第一层输出
    with torch.no_grad():
        features = model.logic_conv(sample)
    
    features = features.cpu().numpy().squeeze()
    
    # 绘制特征图 (9x9 grid)
    fig, axes = plt.subplots(9, 9, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(81):
        ax = axes[i]
        ax.imshow(features[i].reshape(1, 1), cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'F{i}', fontsize=6)
        ax.axis('off')
    
    plt.suptitle(f'Feature Maps (Label: {label})', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'feature_maps.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特征图可视化已保存到: {save_path}")


def visualize_network_architecture(model, output_dir):
    """
    可视化网络架构
    
    Args:
        model: 模型
        output_dir: 输出目录
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 定义层信息
    layers = [
        ('Input\n(9x9)', 81, 'lightblue'),
        ('LogicConv2d\n(81 kernels)', 81, 'lightgreen'),
        ('Hidden\n(27 neurons)', 27, 'lightcoral'),
        ('Middle\n(12 neurons)', 12, 'lightyellow'),
        ('Output\n(4 neurons)', 4, 'lightpink')
    ]
    
    # 绘制网络结构
    layer_names = [layer[0] for layer in layers]
    layer_sizes = [layer[1] for layer in layers]
    colors = [layer[2] for layer in layers]
    
    x = np.arange(len(layers))
    bars = ax.bar(x, layer_sizes, color=colors, edgecolor='black', linewidth=2)
    
    # 添加数值标签
    for i, (bar, size) in enumerate(zip(bars, layer_sizes)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{size}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Neurons/Features', fontsize=14, fontweight='bold')
    ax.set_title('OLCNN Network Architecture', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'network_architecture.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"网络架构可视化已保存到: {save_path}")


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
    _, test_loader = get_mnist_loaders(batch_size=256)
    
    # 创建模型
    print("\n创建模型...")
    model = OLCNN().to(device)
    
    # 加载检查点
    print(f"加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("模型加载成功!")
    
    # 可视化网络架构
    print("\n可视化网络架构...")
    visualize_network_architecture(model, args.output_dir)
    
    # 可视化LUT权重
    print("可视化LUT权重...")
    visualize_lut_weights(model, args.output_dir)
    
    # 可视化样本预测
    print(f"可视化样本预测 ({args.num_samples}个样本)...")
    visualize_samples(model, test_loader, device, args.num_samples, args.output_dir)
    
    # 可视化特征图
    print("可视化特征图...")
    visualize_feature_maps(model, test_loader, device, args.output_dir)
    
    print("\n所有可视化结果已保存到:", args.output_dir)
    print("=" * 80)


if __name__ == '__main__':
    main()