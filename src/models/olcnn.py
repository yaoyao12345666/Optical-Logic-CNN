"""
光学逻辑卷积神经网络 (Optical Logic Convolutional Neural Network, OLCNN)
完整的网络架构实现
"""

import torch
import torch.nn as nn
from .logic_conv2d import LogicConv2d
from .logic_unit import LogicUnit


class OLCNN(nn.Module):
    """
    光学逻辑卷积神经网络 (OLCNN)
    
    完整的网络架构，包含：
    - 第一层：光逻辑卷积层 (81个3x3逻辑核)
    - 隐藏层：27个逻辑神经元 (9组×3神经元)
    - 中间层：12个逻辑神经元 (3组×4神经元)
    - 输出层：4个逻辑神经元 (4分类)
    
    参数：
        num_classes: 分类数量（默认为4，对应MNIST的0-3）
    """
    
    def __init__(self, num_classes=4):
        """
        初始化OLCNN网络
        
        Args:
            num_classes: 分类数量
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Layer 1: 光逻辑卷积
        # 输入: (B, 1, 9, 9) -> 输出: (B, 81)
        self.conv1 = LogicConv2d(in_channels=1, kernel_size=3, stride=3, num_kernels=81)
        
        # Layer 2: 隐藏层逻辑神经元
        # 将81个输入分成9组，每组9个输入，每组有3个神经元
        # 总共 9*3 = 27 个神经元
        self.hidden_layer = nn.ModuleList()
        for group_idx in range(9):
            for neuron_idx in range(3):
                # 每个神经元只接收9个输入（从81个输入中选择9个）
                self.hidden_layer.append(LogicUnit(num_inputs=9))
        
        # Layer 3: 中间层
        # 27个输入 -> 12个神经元（每组9个输入，3组×4神经元）
        self.middle_layer = nn.ModuleList()
        for group_idx in range(3):
            # 每组有4个神经元
            for neuron_idx in range(4):
                self.middle_layer.append(LogicUnit(num_inputs=9))
        
        # Layer 4: 输出层
        # 12个输入 -> num_classes个神经元
        # 为了避免2^12=4096的组合爆炸，我们分成3组，每组4个输入
        self.output_layer = nn.ModuleList()
        for class_idx in range(num_classes):
            # 每个输出神经元接收4个输入
            self.output_layer.append(LogicUnit(num_inputs=4))
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (Batch, 1, 9, 9) 输入图像
            
        Returns:
            output_logits: (Batch, num_classes) 输出logits，用于分类
        """
        # 1. 第一层卷积 -> (B, 81)
        x = self.conv1(x)
        
        # 2. 隐藏层 -> (B, 27)
        # 将81个输入分成9组，每组9个输入，每组有3个神经元
        hidden_outputs = []
        hidden_logits = []  # 存储概率logit用于损失计算
        for group_idx in range(9):
            # 获取这一组的9个输入
            group_input = x[:, group_idx*9:(group_idx+1)*9]  # (B, 9)
            # 这一组有3个神经元
            for neuron_idx in range(3):
                neuron = self.hidden_layer[group_idx * 3 + neuron_idx]
                h_out, _, h_logit = neuron(group_input)  # (B,)
                hidden_outputs.append(h_out)
                hidden_logits.append(h_logit)
        
        x = torch.stack(hidden_outputs, dim=1)  # (B, 27) 用于下一层输入
        x_logits = torch.stack(hidden_logits, dim=1)  # (B, 27) 用于损失计算
        
        # 3. 中间层 -> (B, 12)
        # 将27个输入分成3组，每组9个输入
        middle_outputs = []
        middle_logits = []  # 存储概率logit用于损失计算
        for group_idx in range(3):
            # 获取这一组的9个输入
            group_input = x[:, group_idx*9:(group_idx+1)*9]  # (B, 9)
            # 这一组有4个神经元
            for neuron_idx in range(4):
                neuron = self.middle_layer[group_idx * 4 + neuron_idx]
                m_out, _, m_logit = neuron(group_input)  # (B,)
                middle_outputs.append(m_out)
                middle_logits.append(m_logit)
        
        x = torch.stack(middle_outputs, dim=1)  # (B, 12) 用于下一层输入
        x_logits = torch.stack(middle_logits, dim=1)  # (B, 12) 用于损失计算
        
        # 4. 输出层 -> (B, num_classes)
        # 将12个输入分成3组，每组4个输入
        # 每个输出神经元接收一组输入
        logits = []
        output_logits = []  # 存储输出层的概率logit用于损失计算
        for class_idx in range(self.num_classes):
            # 选择一组输入（循环使用3组）
            group_idx = class_idx % 3
            group_input = x[:, group_idx*4:(group_idx+1)*4]  # (B, 4)
            out_neuron = self.output_layer[class_idx]
            o_out, _, o_logit = out_neuron(group_input)
            logits.append(o_out)
            output_logits.append(o_logit)
            
        logits = torch.stack(logits, dim=1)  # (B, 4) 用于下一层输入
        output_logits = torch.stack(output_logits, dim=1)  # (B, 4) 用于损失计算
        
        # 返回概率值的logit形式，用于CrossEntropyLoss
        return output_logits
    
    def get_num_params(self):
        """
        获取网络参数总数
        
        Returns:
            num_params: 参数总数
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_num_neurons(self):
        """
        获取网络神经元总数
        
        Returns:
            num_neurons: 神经元总数
        """
        return 81 + 27 + 12 + self.num_classes
    
    def print_architecture(self):
        """
        打印网络架构信息
        """
        print("=" * 60)
        print("OLCNN Network Architecture")
        print("=" * 60)
        print(f"Number of classes: {self.num_classes}")
        print(f"Total parameters: {self.get_num_params():,}")
        print(f"Total neurons: {self.get_num_neurons()}")
        print("-" * 60)
        print("Layer 1: LogicConv2d (81 kernels, 3x3)")
        print("Layer 2: Hidden Layer (27 neurons)")
        print("Layer 3: Middle Layer (12 neurons)")
        print(f"Layer 4: Output Layer ({self.num_classes} neurons)")
        print("=" * 60)