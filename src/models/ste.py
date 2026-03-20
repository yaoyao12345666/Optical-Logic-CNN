"""
直通估计器 (Straight-Through Estimator, STE)
解决二值化不可导问题的核心组件
"""

import torch


class StraightThroughEstimator(torch.autograd.Function):
    """
    直通估计器 (STE)
    
    解决二值化不可导的问题：
    - 前向传播：强制变为 0 或 1
    - 反向传播：假装梯度可以直接穿过 (Identity)，让前面的参数能更新
    
    论文 Fig 1E 策略：在 [-1, 1] 范围内梯度为 1，否则为 0
    这里简化为直接返回梯度，效果通常也很好
    """
    
    @staticmethod
    def forward(ctx, input):
        """
        前向传播：二值化
        
        Args:
            input: (Batch,) 概率值，范围 [0, 1]
            
        Returns:
            output: (Batch,) 二值化输出，值为 0 或 1
        """
        # 阈值设为 0.5，大于为1，小于为0
        return (input >= 0.5).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：梯度直通
        
        Args:
            grad_output: 上游梯度
            
        Returns:
            grad_input: 直接返回梯度，不进行修改
        """
        # 论文 Fig 1E 策略：在 [-1, 1] 范围内梯度为 1，否则为 0
        # 这里简化为直接返回梯度，效果通常也很好
        return grad_output