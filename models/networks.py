import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import OpticalLogicUnit, SpectralOpticalUnit

class LogicConv2d(nn.Module):
    """
    光学逻辑卷积层 (Optical Logic Conv2d)
    
    核心机制：
    1. 使用 Unfold 操作提取滑动窗口 (模拟光斑扫描)。
    2. 每个窗口对应一个独立的 OpticalLogicUnit (模拟空间并行的光神经元阵列)。
    3. 所有单元并行计算，输出特征图。
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 use_spectral=False, num_spectral_channels=16, use_ste=True):
        """
        初始化逻辑卷积层
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            use_spectral: 是否使用光谱编码
            num_spectral_channels: 光谱通道数
            use_ste: 是否使用直通估计器
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_spectral = use_spectral
        
        # 计算单个卷积核覆盖的输入节点数 (Input Dim for each logic unit)
        # 例如：3x3 单通道 -> 9 inputs; 3x3 多通道 -> 9 * in_channels inputs
        self.unit_input_dim = kernel_size * kernel_size * in_channels
        
        # 【核心构建】：创建 out_channels 个独立的光逻辑单元
        # 物理含义：输出特征图的每一个通道，都由一组独立的光路系统处理
        self.logic_units = nn.ModuleList()
        
        for _ in range(out_channels):
            if use_spectral:
                # 进阶版：带光谱编码的单元
                unit = SpectralOpticalUnit(
                    num_inputs=self.unit_input_dim, 
                    num_spectral_channels=num_spectral_channels,
                    use_ste=use_ste
                )
            else:
                # 基础版：标准光逻辑单元 (完全复现你提供的代码逻辑)
                unit = OpticalLogicUnit(
                    num_inputs=self.unit_input_dim, 
                    use_ste=use_ste
                )
            self.logic_units.append(unit)

    def forward(self, x):
        """
        前向传播流程：
        1. Im2Col (Unfold): 将输入图像展开为滑动窗口向量。
           - 输入: (B, C_in, H, W)
           - 输出: (B, C_in * K * K, L)，其中 L 是输出位置的数量
        2. 并行计算: 将每个位置的向量送入对应的 LogicUnit。
        3. 重组: 将结果重塑回特征图形状 (B, C_out, H_out, W_out)。
        """
        batch_size = x.size(0)
        
        # --- 步骤 1: 提取滑动窗口 (光学扫描过程) ---
        # unfold 输出形状: (Batch, Channels * Kernel^2, Num_Patches)
        # Num_Patches = H_out * W_out
        patches = F.unfold(x, kernel_size=self.kernel_size, 
                           stride=self.stride, padding=self.padding)
        
        # 转置以便处理: (Batch, Num_Patches, Channels * Kernel^2)
        # 现在 patches[b, i, :] 代表第 b 个样本、第 i 个空间位置的输入向量
        patches = patches.permute(0, 2, 1)
        
        # --- 步骤 2: 并行光计算 ---
        # 我们需要对每个输出通道 (out_channels) 计算一次
        # 结果列表，每个元素形状: (Batch, Num_Patches)
        output_patches_list = []
        
        for i, unit in enumerate(self.logic_units):
            # unit 输入: (Batch * Num_Patches, Input_Dim) 
            # 为了利用 PyTorch 批处理加速，我们将 Batch 和 Num_Patches 合并
            b, l, d = patches.shape
            
            # 确保输入维度匹配 (虽然理论上应该匹配，但做个检查)
            if d != self.unit_input_dim:
                raise ValueError(f"Input dim mismatch: expected {self.unit_input_dim}, got {d}")
            
            # 重塑为 (B*L, D) 以送入全连接风格的 LogicUnit
            flat_input = patches.reshape(-1, d)
            
            # 【核心计算】：光逻辑单元前向传播
            # 如果是 SpectralOpticalUnit，内部会处理光谱复用
            unit_output = unit(flat_input) # 输出: (B*L,)
            
            # 重塑回 (B, L)
            unit_output = unit_output.view(b, l)
            output_patches_list.append(unit_output)
        
        # 堆叠所有通道: (Batch, Num_Patches, Out_Channels)
        stacked_outputs = torch.stack(output_patches_list, dim=2)
        
        # --- 步骤 3: 重组为图像格式 ---
        # 转置回 (Batch, Out_Channels, Num_Patches)
        stacked_outputs = stacked_outputs.permute(0, 2, 1)
        
        # 计算输出高宽
        # 公式: H_out = (H_in + 2*pad - K) / stride + 1
        h_in = x.size(2)
        w_in = x.size(3)
        h_out = int((h_in + 2 * self.padding - self.kernel_size) / self.stride + 1)
        w_out = int((w_in + 2 * self.padding - self.kernel_size) / self.stride + 1)
        
        # Fold 操作：将 (B, C_out, H_out*W_out) 变回 (B, C_out, H_out, W_out)
        # 注意：fold 需要输入形状为 (B, C*K*K, L)，但我们现在是 (B, C, L)
        # 所以这里不能直接用 fold，直接 reshape 即可，因为我们的输出已经是标量值 (1x1 的核响应)
        # 逻辑卷积的输出本身就是每个位置的一个标量，不需要像传统卷积那样再乘核权重求和
        # 因为求和已经在 LogicUnit 内部完成了！
        
        final_output = stacked_outputs.reshape(batch_size, self.out_channels, h_out, w_out)
        
        return final_output


class OLCNN(nn.Module):
    """
    光学逻辑卷积神经网络 (OLCNN)
    
    架构设计参考典型的光学神经网络论文结构：
    Input (9x9) -> [Optical Conv] -> [Pool] -> [Optical Conv] -> [Pool] -> [FC] -> Output
    """
    
    def __init__(self, num_classes=4, input_size=9, use_spectral=False, num_spectral_channels=16, use_ste=True):
        """
        初始化OLCNN模型
        
        参数:
            num_classes: 类别数
            input_size: 输入图片大小
            use_spectral: 是否使用光谱编码
            num_spectral_channels: 光谱通道数
            use_ste: 是否使用直通估计器
        """
        super().__init__()
        
        self.input_size = input_size
        self.use_spectral = use_spectral
        self.num_spectral_channels = num_spectral_channels
        
        # --- 第一层：光学逻辑卷积 ---
        # 输入: 1通道 (灰度), 9x9
        # 核大小: 3x3 (覆盖 9 个像素) -> 每个神经元处理 9 个输入
        # 输出: 16 个特征通道
        self.conv1 = LogicConv2d(
            in_channels=1, 
            out_channels=16, 
            kernel_size=3, 
            stride=1, 
            padding=0, # 9x9 - 3 + 1 = 7x7
            use_spectral=use_spectral,
            num_spectral_channels=num_spectral_channels,
            use_ste=use_ste
        )
        
        # 池化层 (模拟光学系统中的下采样或电学后处理)
        # 2x2 Max Pooling: 7x7 -> 3x3 (向下取整)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- 第二层：光学逻辑卷积 ---
        # 输入: 16 通道, 3x3
        # 核大小: 3x3 (覆盖整个 3x3 区域) -> 每个神经元处理 16*9 = 144 个输入
        # 输出: 32 个特征通道
        # 注意：这里 padding=0, 3x3 - 3 + 1 = 1x1. 全局感知。
        self.conv2 = LogicConv2d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=3, 
            stride=1, 
            padding=0, 
            use_spectral=use_spectral,
            num_spectral_channels=num_spectral_channels,
            use_ste=use_ste
        )
        
        # 池化 (此时尺寸已经是 1x1，池化可选，或者用于 Dropout 效果)
        # 如果 conv2 输出是 1x1，这里就不需要 pool 了，或者我们可以调整 conv1 的 stride
        # 让我们重新计算一下尺寸以确保稳健：
        # Input: 9x9
        # Conv1 (3x3, s=1): 7x7
        # Pool1 (2x2, s=2): 3x3
        # Conv2 (3x3, s=1): 1x1
        # 此时特征图是 (Batch, 32, 1, 1)
        
        # --- 全连接层 (分类头) ---
        # 输入: 32 * 1 * 1 = 32
        # 输出: num_classes
        self.fc = nn.Linear(32, num_classes)
        
        # 打印架构信息
        self.print_architecture()

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, 1, height, width)
            
        返回:
            输出张量，形状为 (batch_size, num_classes)
        """
        # x shape: (Batch, 1, 9, 9)
        
        # Layer 1
        x = self.conv1(x)   # -> (B, 16, 7, 7)
        x = F.relu(x)       # 虽然 LogicUnit 输出是 0/1，但在中间层加 ReLU 无害且有时有助于梯度流动 (可选)
                            # 严格物理复现的话，LogicUnit 已经是阶跃函数，不需要 ReLU。
                            # 但为了训练稳定性，通常保留或去掉均可。这里为了纯粹性，我们去掉 ReLU，因为 Step 函数已经够硬了。
                            # 修正：Step 函数输出 0/1，本身类似 ReLU 后的二值化。不再额外加 ReLU。
        
        x = self.pool1(x)   # -> (B, 16, 3, 3)
        
        # Layer 2
        x = self.conv2(x)   # -> (B, 32, 1, 1)
        
        # Flatten
        x = x.view(x.size(0), -1) # -> (B, 32)
        
        # Classifier
        x = self.fc(x)      # -> (B, num_classes)
        
        return x

    def print_architecture(self):
        """
        打印网络架构信息
        """
        print("\n" + "="*40)
        print("OLCNN Architecture Summary")
        print("="*40)
        print(f"Input Size: {self.input_size}x{self.input_size}")
        print(f"Use Spectral Encoding: {self.use_spectral}")
        if self.use_spectral:
            print(f"Spectral Channels: {self.num_spectral_channels}")
        print("-" * 40)
        print("Layer 1: LogicConv2d(1->16, K=3) -> Output: 7x7x16")
        print("         MaxPool(2x2) -> Output: 3x3x16")
        print("Layer 2: LogicConv2d(16->32, K=3) -> Output: 1x1x32")
        print("Layer 3: Linear(32 -> Classes)")
        print("="*40 + "\n")

# ================================================================
# 快速测试脚本
# ================================================================
if __name__ == "__main__":
    # 构造假数据 (Batch=2, Channel=1, Height=9, Width=9)
    dummy_input = torch.randn(2, 1, 9, 9)
    # 二值化输入 (模拟真实光学输入)
    dummy_input = (dummy_input > 0).float()
    
    print("Testing OLCNN with Binary Input...")
    
    # 实例化模型 (基础版)
    model = OLCNN(num_classes=4, use_spectral=False)
    model.eval() # 设置为评估模式，使用硬阶跃函数
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    print(f"Output Values (Logits): \n{output}")
    
    # 预测类别
    _, predicted = torch.max(output, 1)
    print(f"Predicted Classes: {predicted}")
    
    print("\nTesting Spectral Version...")
    model_spec = OLCNN(num_classes=4, use_spectral=True, num_spectral_channels=8)
    model_spec.eval()
    with torch.no_grad():
        output_spec = model_spec(dummy_input)
    print(f"Spectral Output Shape: {output_spec.shape}")
