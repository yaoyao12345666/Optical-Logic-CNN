"""
光逻辑单元 (Logic Unit / LUT)
模拟K-input光逻辑门的核心组件
"""

import torch
import torch.nn as nn
from .ste import StraightThroughEstimator


class LogicUnit(nn.Module):
    """
    模拟一个 K-input 的光逻辑门
    
    本质是一个可学习的查找表 (Look-Up Table, LUT)
    
    输入：N 个二值信号 (0/1)
    输出：1 个二值信号 (0/1)
    
    参数：
        num_inputs: 输入信号的数量
        num_combinations: 可能的输入组合数 (2^num_inputs)
        lut_params: 可学习的查找表参数，每个组合对应一个参数
    """
    
    def __init__(self, num_inputs):
        """
        初始化光逻辑单元
        
        Args:
            num_inputs: 输入信号的数量
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.num_combinations = 2 ** num_inputs
        
        # 初始化查找表权重
        # 每个组合对应一个可学习参数，初始化为随机高斯分布
        # 训练后，这些值会通过 Sigmoid 逼近 0 或 1，代表具体的逻辑功能 (如 AND, OR, XOR)
        self.lut_params = nn.Parameter(torch.randn(self.num_combinations))
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (Batch, num_inputs), 值域应为 0.0 或 1.0
            
        Returns:
            output: (Batch,) 二值化输出，值为 0 或 1
            selected_probs: (Batch,) 查表得到的概率值，范围 [0, 1]
            prob_logits: (Batch,) 概率值的logit形式，用于损失计算
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 1. 将二值输入转换为整数索引 (Binary to Decimal)
        # 例如：输入 [1, 0, 1] -> 1*4 + 0*2 + 1*1 = 5
        # 构造权重向量 [2^(N-1), 2^(N-2), ..., 2^0]
        powers = 2 ** torch.arange(self.num_inputs - 1, -1, -1, device=device)
        indices = (x * powers).sum(dim=1).long()  # Shape: (Batch,)
        
        # 2. 查表 (Gather)
        # 将参数通过 Sigmoid 映射到 (0, 1) 区间，表示输出为 1 的概率
        lut_probs = torch.sigmoid(self.lut_params)
        selected_probs = lut_probs[indices]  # Shape: (Batch,)
        
        # 3. 应用 STE 进行二值化输出
        # 前向是硬 0/1，反向梯度传回给 lut_params
        output = StraightThroughEstimator.apply(selected_probs)
        
        # 4. 返回概率值的logit形式，用于损失计算
        # 将概率(0,1)转换为logit范围(-10, 10)
        prob_logits = torch.logit(selected_probs, eps=1e-7) * 5.0
        
        return output, selected_probs, prob_logits
    
    def get_logic_function(self):
        """
        获取当前学习到的逻辑函数
        
        Returns:
            logic_table: (num_combinations,) 二值化的查找表
        """
        with torch.no_grad():
            probs = torch.sigmoid(self.lut_params)
            return (probs >= 0.5).float()