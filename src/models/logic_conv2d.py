"""
光逻辑卷积层 (Optical Logic Convolution, OLCO)
基于光逻辑门的卷积层实现
"""

import torch
import torch.nn as nn
from .logic_unit import LogicUnit


class LogicConv2d(nn.Module):
    """
    光逻辑卷积层 (OLCO)
    
    论文 Fig 5A 第一层：
    输入：9x9 二值图像
    操作：81 个 3x3 逻辑核，步长 (Stride) = 3
    输出：81 个特征值
    
    参数：
        in_channels: 输入通道数（通常为1）
        kernel_size: 逻辑核大小（通常为3x3）
        stride: 步长（通常为3）
        num_kernels: 逻辑核的数量（通常为81）
    """
    
    def __init__(self, in_channels=1, kernel_size=3, stride=3, num_kernels=81):
        """
        初始化光逻辑卷积层
        
        Args:
            in_channels: 输入通道数
            kernel_size: 逻辑核大小
            stride: 步长
            num_kernels: 逻辑核的数量
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_kernels = num_kernels
        self.num_inputs_per_kernel = kernel_size * kernel_size  # 3*3 = 9
        
        # 创建 num_kernels 个独立的逻辑单元
        # 每个单元学习一种特定的 9-input 逻辑函数
        self.logic_units = nn.ModuleList([
            LogicUnit(self.num_inputs_per_kernel) for _ in range(num_kernels)
        ])
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (Batch, Channels, H, W) -> e.g., (B, 1, 9, 9)
            
        Returns:
            output: (Batch, num_kernels) 卷积输出特征
        """
        B, C, H, W = x.shape
        K = self.kernel_size
        S = self.stride
        
        # 1. 使用 unfold 提取滑动窗口 (Im2Col)
        # 输出形状: (B, C, Num_Patches, K*K)
        # 对于 9x9 输入，3x3 核，stride 3 -> 输出 3x3=9 个 patches
        patches = x.unfold(2, K, S).unfold(3, K, S)
        patches = patches.contiguous().view(B, C, -1, K * K)
        
        # 论文设定：输入通道 C=1
        patches = patches.squeeze(1)  # (B, Num_Patches, 9)
        
        num_patches = patches.shape[1]  # 应该是 9
        
        # 2. 分配逻辑核
        # 论文说 "81 kernels"。
        # 逻辑推断：共有 9 个空间位置 (patches)，每个位置应用 9 个不同的逻辑核。
        # 9 positions * 9 kernels/pos = 81 outputs.
        kernels_per_pos = self.num_kernels // num_patches
        
        all_outputs = []
        
        for pos_idx in range(num_patches):
            # 取出第 pos_idx 个位置的输入 (B, 9)
            patch_input = patches[:, pos_idx, :]
            
            pos_outputs = []
            for k_idx in range(kernels_per_pos):
                unit_idx = pos_idx * kernels_per_pos + k_idx
                if unit_idx < len(self.logic_units):
                    # 应用对应的逻辑单元
                    out, _, _ = self.logic_units[unit_idx](patch_input)
                    pos_outputs.append(out)
            
            # 堆叠该位置的所有输出
            if pos_outputs:
                # (B, kernels_per_pos)
                stacked = torch.stack(pos_outputs, dim=1)
                all_outputs.append(stacked)
        
        # 拼接所有位置的特征 -> (B, Total_Kernels)
        # 最终形状: (B, 81)
        return torch.cat(all_outputs, dim=1)
    
    def get_num_params(self):
        """
        获取参数数量
        
        Returns:
            num_params: 参数总数
        """
        return sum(p.numel() for p in self.parameters())