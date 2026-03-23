# scripts/train.py
import sys
import os
import argparse
import yaml
import random
import torch
import numpy as np

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from models.networks import OLCNN
from models.layers import OpticalLogicUnit, SpectralOpticalUnit
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def check_saturation(model):
    """
    检查模型中神经元的饱和情况
    
    功能：检查模型中有多少比例的神经元已经饱和（输出恒为 0 或 1）
    
    参数：
        model: 模型实例
    """
    total_units = 0
    saturated_units = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (OpticalLogicUnit, SpectralOpticalUnit)):
            # 获取当前权重的统计信息
            w_mean = module.weights.mean().item()
            w_std = module.weights.std().item()
            
            # 简单的启发式判断：如果权重均值绝对值过大，可能饱和
            # 更准确的方法是跑一个 batch 看输出方差，这里简化处理
            if abs(w_mean) > 0.1: # 阈值根据初始化调整
                saturated_units += 1
            
            total_units += 1
            
            # 打印第一个单元的详情用于调试
            if name == "conv1.logic_units.0":
                print(f"  [Debug] {name}: Weight Mean={w_mean:.5f}, Std={w_std:.5f}, Bias={module.bias.item():.5f}")

    if total_units > 0:
        print(f"  [Saturation Check] {saturated_units}/{total_units} units potentially saturated.")

def revive_dead_units(model, threshold=0.01):
    """
    复活可能饱和的死亡单元
    
    功能：如果某个单元的权重均值绝对值过大，将其重置为微小随机值
    
    参数：
        model: 模型实例
        threshold: 权重均值阈值，超过此值的单元被认为是饱和的
        
    返回：
        revived_count: 复活的单元数量
    """
    revived_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (OpticalLogicUnit, SpectralOpticalUnit)):
            w_mean = abs(module.weights.mean().item())
            # 如果权重均值过大 (例如 > 0.03)，说明该单元可能已饱和
            if w_mean > 0.03:
                with torch.no_grad():
                    # 重新初始化为极小值
                    torch.nn.init.uniform_(module.weights, -0.005, 0.005)
                    module.bias.fill_(-0.002)
                revived_count += 1
                print(f"  [Revive] Resetting {name} (Mean Weight was {w_mean:.4f})")
    return revived_count

class OLCNNTrainer:
    """
    OLCNN模型训练器
    
    功能：管理模型的训练、验证、保存等过程
    """
    
    def __init__(self, model, config, device):
        """
        初始化训练器
        
        参数：
            model: 模型实例
            config: 配置字典
            device: 训练设备
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = config.get('output_dir', './outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 损失函数
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 学习率调度器 (支持 cosine 和 step)
        scheduler_type = config['training'].get('scheduler', 'step')
        total_epochs = config['training']['epochs']
        
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=total_epochs, 
                eta_min=1e-6
            )
        else:
            # 默认 StepLR
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=10, 
                gamma=0.1
            )
            
        self.best_acc = 0.0

    def train_epoch(self, dataloader, epoch):
        """
        执行单个训练 epoch
        
        参数：
            dataloader: 训练数据加载器
            epoch: 当前 epoch 编号
            
        返回：
            total_loss / len(dataloader): 平均损失
            100. * correct / total: 准确率
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            loss = self.criterion(outputs, targets)
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # 检测梯度是否为0，验证STE是否生效
            # 暂时注释掉警告输出，避免干扰
            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.norm().item()
            #         if grad_norm == 0:
            #             print(f"⚠️ 警告：参数 {name} 的梯度为 0！")
            #         # print(f"{name} grad norm: {grad_norm}")
            self.optimizer.step()
            
            # 🔴【新增】关键步骤：钳制权重，防止饱和
            # 遍历模型所有模块，如果是光逻辑单元，就执行钳制
            for module in self.model.modules():
                if hasattr(module, 'clamp_weights'):
                    module.clamp_weights()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
            
        return total_loss / len(dataloader), 100. * correct / total

    def validate(self, dataloader, epoch):
        """
        执行验证
        
        参数：
            dataloader: 验证数据加载器
            epoch: 当前 epoch 编号
            
        返回：
            total_loss / len(dataloader): 平均损失
            acc: 准确率
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val] ", leave=False)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                total_loss += loss.item()
                
        acc = 100. * correct / total
        return total_loss / len(dataloader), acc

    def save_checkpoint(self, epoch, acc, filename='checkpoint_latest.pth'):
        """
        保存模型检查点
        
        参数：
            epoch: 当前 epoch 编号
            acc: 当前准确率
            filename: 保存文件名
        """
        path = os.path.join(self.output_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': acc,
            'config': self.config
        }, path)
        print(f"💾 Model saved to {path}")

    def train_loop(self, train_loader, val_loader):
        """
        执行完整的训练循环
        
        参数：
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        epochs = self.config['training']['epochs']
        save_freq = self.config.get('save_freq', 5)
        
        for epoch in range(1, epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            print(f"\n[Epoch {epoch}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            
            # 验证
            val_loss, val_acc = self.validate(val_loader, epoch)
            print(f"[Epoch {epoch}] Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # 更新学习率
            self.scheduler.step()
            
            # 保存最佳模型
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(epoch, val_acc, 'best_model.pth')
                print(f"🌟 New Best Accuracy: {self.best_acc:.2f}%")
            
            # 检查神经元饱和情况
            if epoch % 1 == 0: # 每轮打印
                print(f"\n--- Epoch {epoch} Summary ---")
                check_saturation(self.model)
            
            # 复活死亡单元
            if epoch % 5 == 0: # 每 5 轮检查一次
                count = revive_dead_units(self.model)
                if count > 0:
                    print(f"  >>> Revived {count} saturated units!")
            
            # 定期保存
            if epoch % save_freq == 0:
                self.save_checkpoint(epoch, val_acc, f'checkpoint_epoch_{epoch}.pth')
                
        print(f"\n🎉 Training Finished! Best Accuracy: {self.best_acc:.2f}%")

def get_dataloaders(config):
    """
    根据配置构建数据加载器
    
    功能：支持 MNIST，自动处理 9x9 缩放
    
    参数：
        config: 配置字典
        
    返回：
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    data_root = config.get('data_root', './data')
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 0)
    input_size = config['input_size']
    dataset_name = config.get('dataset', 'MNIST')
    
    os.makedirs(data_root, exist_ok=True)

    # 定义变换
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)), # 核心：调整为光学输入尺寸
        transforms.ToTensor(),
        # 注意：这里不做二值化，让模型内部的 STE 处理，或者根据需要开启
        # transforms.Lambda(lambda x: (x > 0.5).float()) 
    ])

    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported yet.")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"📦 Data loaded: {len(train_dataset)} train samples, {len(test_dataset)} test samples.")
    return train_loader, test_loader

def load_config(config_path):
    """
    加载 YAML 配置文件
    
    参数：
        config_path: 配置文件路径
        
    返回：
        config: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed):
    """
    设置随机种子以保证实验可复现
    
    参数：
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 固定卷积算法，进一步提升确定性（可能会稍微降低速度）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set to: {seed}")

def main():
    """
    主函数，协调整个训练过程
    """
    parser = argparse.ArgumentParser(description="Train OLCNN")
    # 默认指向你创建的配置文件，也可以通过命令行参数修改
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()

    config_path = os.path.join(ROOT_DIR, args.config)
    
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found at {config_path}")
        print("Please ensure you created configs/config.yaml")
        return

    # 1. 加载配置
    print(f"📄 Loading config from: {config_path}")
    config = load_config(config_path)
    
    # 2. 设置随机种子
    set_seed(config['seed'])
    
    # 3. 设置设备
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, falling back to CPU.")
        device = 'cpu'
    print(f"🚀 Using Device: {device}")

    # 4. 准备数据
    train_loader, val_loader = get_dataloaders(config)

    # 5. 初始化模型
    # 从 config 中提取模型参数
    model_cfg = config['model']
    model = OLCNN(
        num_classes=model_cfg['num_classes'],
        use_spectral=model_cfg['use_spectral_encoding'],
        num_spectral_channels=model_cfg['num_spectral_channels'],
        use_ste=model_cfg['use_ste']
    )
    
    # 打印模型信息
    print("\n" + "="*30)
    print("Model Architecture:")
    print("="*30)
    # 简单打印结构，或者你可以调用之前的测试函数
    print(model)
    print("="*30 + "\n")

    # 6. 初始化训练器并开始训练
    trainer = OLCNNTrainer(model, config, device)
    trainer.train_loop(train_loader, val_loader)

if __name__ == '__main__':
    main()