"""
训练脚本
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import OLCNN
from src.utils import get_mnist_loaders, Trainer, set_seed, get_device, print_model_summary


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练光学逻辑卷积神经网络')
    
    # 数据参数
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    # 模型参数
    parser.add_argument('--num_classes', type=int, default=4, help='分类数量')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2正则化系数')
    parser.add_argument('--lr_step_size', type=int, default=30, help='学习率衰减步长')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='学习率衰减因子')
    
    # 早停参数
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    parser.add_argument('--target_acc', type=float, default=95.0, help='目标准确率')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='auto', help='计算设备 (auto/cpu/cuda)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志目录')
    parser.add_argument('--print_freq', type=int, default=10, help='打印频率')
    parser.add_argument('--save_freq', type=int, default=50, help='保存频率')
    parser.add_argument('--use_tensorboard', action='store_true', default=True, help='是否使用TensorBoard')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    train_loader, test_loader = get_mnist_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 创建模型
    print("\n创建模型...")
    model = OLCNN(num_classes=args.num_classes).to(device)
    print_model_summary(model)
    model.print_architecture()
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )
    
    # TensorBoard
    writer = None
    if args.use_tensorboard:
        writer = SummaryWriter(args.log_dir)
        print(f"\nTensorBoard日志目录: {args.log_dir}")
        print(f"启动TensorBoard: tensorboard --logdir {args.log_dir}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        writer=writer,
        save_dir=args.save_dir,
        print_freq=args.print_freq,
        save_freq=args.save_freq
    )
    
    # 恢复训练
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        print(f"\n从检查点恢复训练: {args.resume}")
        start_epoch, best_acc = trainer.load_checkpoint(args.resume)
        print(f"恢复到epoch {start_epoch}, 最佳准确率: {best_acc:.2f}%")
    
    # 训练模型
    print("\n开始训练...")
    print("=" * 80)
    
    trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        start_epoch=start_epoch,
        best_acc=best_acc,
        patience=args.patience,
        target_acc=args.target_acc
    )
    
    print("\n训练完成!")
    print(f"最佳模型保存在: {os.path.join(args.save_dir, 'best_model.pth')}")
    
    # 关闭TensorBoard
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()