"""
训练器模块
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime


class Trainer:
    """
    训练器类，封装训练和评估逻辑
    """
    
    def __init__(self, model, train_loader, test_loader, config):
        """
        初始化训练器
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            config: 配置字典
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('lr', 0.01),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get('lr_step_size', 30),
            gamma=config.get('lr_gamma', 0.5)
        )
        
        # TensorBoard
        self.writer = None
        if config.get('use_tensorboard', False):
            log_dir = os.path.join(
                config.get('log_dir', './logs'),
                datetime.now().strftime('%Y%m%d_%H%M%S')
            )
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard logs will be saved to: {log_dir}")
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_acc': []
        }
        
        # 最佳模型
        self.best_test_acc = 0.0
        self.best_epoch = 0
        
    def train_epoch(self, epoch):
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch
            
        Returns:
            avg_loss: 平均损失
            acc: 训练准确率
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            output_logits = self.model(data)
            
            # 计算损失
            loss = self.criterion(output_logits, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # 计算准确率
            pred = output_logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
        avg_loss = running_loss / len(self.train_loader)
        acc = 100.0 * correct / total
        
        return avg_loss, acc
    
    def evaluate(self):
        """
        评估模型
        
        Returns:
            test_acc: 测试准确率
        """
        self.model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output_logits = self.model(data)
                pred = output_logits.argmax(dim=1)
                test_correct += (pred == target).sum().item()
                test_total += target.size(0)
        
        test_acc = 100.0 * test_correct / test_total
        return test_acc
    
    def train(self, epochs, save_dir='./checkpoints'):
        """
        训练模型
        
        Args:
            epochs: 训练轮数
            save_dir: 模型保存目录
        """
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60)
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 评估
            test_acc = self.evaluate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_acc'].append(test_acc)
            
            # TensorBoard记录
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/test', test_acc, epoch)
                self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 打印进度
            if (epoch + 1) % self.config.get('print_freq', 10) == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Loss: {train_loss:.4f} "
                      f"Train Acc: {train_acc:.2f}% | "
                      f"Test Acc: {test_acc:.2f}%")
            
            # 保存最佳模型
            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                self.best_epoch = epoch + 1
                self.save_checkpoint(save_dir, is_best=True)
            
            # 定期保存
            if (epoch + 1) % self.config.get('save_freq', 50) == 0:
                self.save_checkpoint(save_dir, epoch=epoch+1)
            
            # 早期停止
            if test_acc > self.config.get('target_acc', 95.0):
                print(f">>> Reached target accuracy {self.config.get('target_acc', 95.0)}%! Stopping early.")
                break
        
        print("=" * 60)
        print(f"Training finished!")
        print(f"Best Test Accuracy: {self.best_test_acc:.2f}% at epoch {self.best_epoch}")
        print("=" * 60)
        
        # 关闭TensorBoard
        if self.writer is not None:
            self.writer.close()
        
        return self.history
    
    def save_checkpoint(self, save_dir, is_best=False, epoch=None):
        """
        保存检查点
        
        Args:
            save_dir: 保存目录
            is_best: 是否为最佳模型
            epoch: 当前epoch
        """
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_test_acc': self.best_test_acc,
            'config': self.config
        }
        
        if is_best:
            path = os.path.join(save_dir, 'best_model.pth')
            torch.save(checkpoint, path)
            print(f"Best model saved to {path}")
        elif epoch is not None:
            path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_test_acc = checkpoint.get('best_test_acc', 0.0)
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint