import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# ================================================================
# 1. 改进版直通估计器 (Sigmoid STE)
# 核心修改：前向传播使用硬阶跃，但引入一个温度参数 T 控制反向梯度的衰减
# 这能防止梯度在权重稍大时瞬间爆炸或消失
# ================================================================
class StraightThroughEstimator(torch.autograd.Function):
    """
    改进版直通估计器 (Sigmoid STE)
    
    核心修改：前向传播使用硬阶跃，但引入一个温度参数 T 控制反向梯度的衰减
    这能防止梯度在权重稍大时瞬间爆炸或消失
    """
    
    @staticmethod
    def forward(ctx, input, temperature=1.0):
        """
        前向传播
        
        参数:
            ctx: 上下文对象，用于保存中间值
            input: 输入张量
            temperature: 温度参数，控制梯度衰减速率
            
        返回:
            二值化输出张量，形状与输入相同
        """
        ctx.save_for_backward(input)
        ctx.temperature = temperature
        # 前向：硬阶跃 (0/1)
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播
        
        参数:
            ctx: 上下文对象，用于获取保存的中间值
            grad_output: 来自上一层的梯度
            
        返回:
            输入张量的梯度，温度参数的梯度(None)
        """
        input, = ctx.saved_tensors
        temp = ctx.temperature
        
        # 反向：使用 Sigmoid 的导数作为梯度掩码
        # 当 input 接近 0 时，梯度接近 1；当 input 很大时，梯度接近 0
        # 公式：sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        # 为了简化，我们直接用 exp(-|x|/T) 作为衰减因子
        gradient_mask = torch.exp(-torch.abs(input) / temp)
        
        # 应用梯度衰减
        return grad_output * gradient_mask, None

# ================================================================
# 2. 光逻辑单元 (带权重钳制)
# ================================================================
class OpticalLogicUnit(nn.Module):
    """
    光逻辑单元 (带权重钳制)
    
    实现光学系统的基本逻辑计算单元，执行加权和计算并输出二值化结果
    """
    
    def __init__(self, num_inputs, use_ste=True, weight_clip=0.02):
        """
        初始化光逻辑单元
        
        参数:
            num_inputs: 输入维度
            use_ste: 是否使用直通估计器
            weight_clip: 权重钳制阈值
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.use_ste = use_ste
        self.weight_clip = weight_clip # 权重钳制阈值

        # 初始化：极小范围
        self.weights = nn.Parameter(torch.Tensor(num_inputs))
        init.uniform_(self.weights, -0.01, 0.01)

        # 偏置：微小负数，防止初始全 1
        self.bias = nn.Parameter(torch.ones(1) * -0.005)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, num_inputs)
            
        返回:
            二值化输出张量，形状为 (batch_size,)
        """
        # 计算加权和
        z = torch.sum(x * self.weights, dim=1) + self.bias

        if self.training and self.use_ste:
            # 使用改进版 STE，温度参数 T=0.5 (可调节)
            return StraightThroughEstimator.apply(z, 0.5)
        else:
            return (z >= 0).float()
    
    def clamp_weights(self):
        """
        在每次 optimizer.step() 后调用，强制权重不超过阈值
        
        防止权重过大导致神经元饱和，确保计算始终在合理区间内进行
        """
        with torch.no_grad():
            self.weights.clamp_(-self.weight_clip, self.weight_clip)

# ================================================================
# 3. 光谱编码扩展 (带权重钳制)
# ================================================================
class SpectralOpticalUnit(OpticalLogicUnit):
    """
    光谱编码光逻辑单元 (带权重钳制)
    
    通过多通道并行处理提高模型表达能力，实现更丰富的特征提取
    """
    
    def __init__(self, num_inputs, num_spectral_channels=16, use_ste=True, weight_clip=0.02):
        """
        初始化光谱编码光逻辑单元
        
        参数:
            num_inputs: 输入维度
            num_spectral_channels: 光谱通道数
            use_ste: 是否使用直通估计器
            weight_clip: 权重钳制阈值
        """
        nn.Module.__init__(self)
        
        self.num_inputs = num_inputs
        self.num_spectral_channels = num_spectral_channels
        self.use_ste = use_ste
        self.weight_clip = weight_clip
        
        # 矩阵权重初始化
        self.weights = nn.Parameter(torch.Tensor(num_spectral_channels, num_inputs))
        init.uniform_(self.weights, -0.01, 0.01)
        
        self.bias = nn.Parameter(torch.ones(1) * -0.005)
        
        # 通道权重
        self.channel_weights = nn.Parameter(torch.ones(num_spectral_channels) / num_spectral_channels)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, num_inputs)
            
        返回:
            二值化输出张量，形状为 (batch_size,)
        """
        # 多通道并行计算
        z_spectral = torch.matmul(x, self.weights.t())
        z_spectral = z_spectral + self.bias 
        # 通道融合
        z_total = torch.sum(z_spectral * self.channel_weights, dim=1)
        
        if self.training and self.use_ste:
            return StraightThroughEstimator.apply(z_total, 0.5)
        else:
            return (z_total >= 0).float()

    def clamp_weights(self):
        """
        在每次 optimizer.step() 后调用，强制权重不超过阈值
        
        防止权重过大导致神经元饱和，确保计算始终在合理区间内进行
        同时确保通道权重非负
        """
        with torch.no_grad():
            self.weights.clamp_(-self.weight_clip, self.weight_clip)
            self.channel_weights.clamp_(0.0, 1.0) # 通道权重保持正数

if __name__ == "__main__":
    """
    自测脚本
    测试光逻辑单元的基本功能
    """
    # 创建光逻辑单元，输入维度为9
    unit = OpticalLogicUnit(9)
    # 创建随机二值输入，形状为 (2, 9)
    x = (torch.rand(2, 9) > 0.5).float()
    # 前向传播
    y = unit(x)
    # 打印输出结果
    print(f"Test Output: {y}")
    # 打印权重范围
    print(f"Weights Range: [{unit.weights.min():.4f}, {unit.weights.max():.4f}]")
