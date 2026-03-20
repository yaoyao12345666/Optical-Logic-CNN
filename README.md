# Optical Logic Convolutional Neural Network (OLCNN)

基于光逻辑门的二值化神经网络实现，使用查找表(LUT)模拟光逻辑门进行图像分类。

## 📖 项目简介

本项目实现了基于光逻辑门的卷积神经网络(OLCNN)，核心创新点包括：

- **直通估计器(STE)**：解决二值化不可导问题
- **光逻辑单元(LUT)**：使用可学习查找表模拟K-input光逻辑门
- **光逻辑卷积层(OLCO)**：基于逻辑门的卷积操作
- **完整网络架构**：适用于MNIST 9x9二值化图像分类

## 🚀 核心特性

### 1. 光逻辑单元 (Logic Unit)
- 使用查找表(LUT)模拟光逻辑门
- 支持任意输入数量的逻辑门
- 通过Sigmoid映射参数到(0,1)区间
- 直通估计器(STE)解决梯度流动问题

### 2. 光逻辑卷积层 (LogicConv2d)
- 81个3x3逻辑核
- 步长为3的卷积操作
- 每个位置应用9个不同的逻辑核
- 输出81个特征值

### 3. 完整网络架构 (OLCNN)
```
输入: (B, 1, 9, 9) 9x9二值化图像
  ↓
Layer 1: LogicConv2d (81 kernels, 3x3)
  ↓ (B, 81)
Layer 2: Hidden Layer (27 neurons)
  ↓ (B, 27)
Layer 3: Middle Layer (12 neurons)
  ↓ (B, 12)
Layer 4: Output Layer (4 neurons)
  ↓ (B, 4)
输出: 4分类logits
```

## 📦 安装

### 环境要求
- Python 3.7+
- PyTorch 1.8+
- CUDA 10.2+ (可选，用于GPU加速)

### 安装依赖

```bash
pip install -r requirements.txt
```

## 🎯 快速开始

### 1. 训练模型

```bash
python examples/train.py --epochs 200 --batch_size 256 --lr 0.01
```

### 2. 评估模型

```bash
python examples/evaluate.py --checkpoint checkpoints/best_model.pth
```

### 3. 可视化结果

```bash
python examples/visualize.py --checkpoint checkpoints/best_model.pth
```

## 📊 实验结果

### MNIST 4分类 (0-3)
- **输入**: 9x9二值化图像
- **网络**: 81 + 27 + 12 + 4 = 124个逻辑神经元
- **参数**: ~10K个可学习参数
- **准确率**: 待更新

### 网络架构详情
```
Layer 1: LogicConv2d
  - 81个3x3逻辑核
  - 每个核512个参数(2^9组合)
  - 总参数: 81 × 512 = 41,472

Layer 2: Hidden Layer
  - 27个9输入逻辑神经元
  - 每个神经元512个参数
  - 总参数: 27 × 512 = 13,824

Layer 3: Middle Layer
  - 12个9输入逻辑神经元
  - 总参数: 12 × 512 = 6,144

Layer 4: Output Layer
  - 4个4输入逻辑神经元
  - 每个神经元16个参数(2^4组合)
  - 总参数: 4 × 16 = 64

总参数: 61,504
```

## 📁 项目结构

```
Optical-Logic-CNN/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ste.py              # 直通估计器
│   │   ├── logic_unit.py       # 光逻辑单元
│   │   ├── logic_conv2d.py     # 光逻辑卷积层
│   │   └── olcnn.py            # 完整网络架构
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py      # 数据加载器
│       ├── trainer.py          # 训练器
│       └── utils.py            # 工具函数
├── examples/
│   ├── train.py                # 训练脚本
│   ├── evaluate.py             # 评估脚本
│   └── visualize.py            # 可视化脚本
├── data/                       # 数据目录
├── checkpoints/                # 模型检查点
├── logs/                       # 训练日志
├── requirements.txt            # 依赖列表
└── README.md                  # 项目说明
```

## 🔧 配置参数

### 训练配置
```python
config = {
    'num_classes': 4,           # 分类数量
    'batch_size': 256,           # 批次大小
    'epochs': 200,               # 训练轮数
    'lr': 0.01,                  # 学习率
    'weight_decay': 1e-5,        # L2正则化
    'lr_step_size': 30,          # 学习率衰减步长
    'lr_gamma': 0.5,             # 学习率衰减因子
    'target_acc': 95.0,          # 目标准确率
    'print_freq': 10,            # 打印频率
    'save_freq': 50,             # 保存频率
    'use_tensorboard': True      # 是否使用TensorBoard
}
```

## 🎓 技术细节

### 直通估计器 (STE)
```python
# 前向传播：二值化
output = (input >= 0.5).float()

# 反向传播：梯度直通
grad_input = grad_output
```

### 光逻辑单元 (LUT)
```python
# 1. 二值输入转索引
indices = (x * powers).sum(dim=1).long()

# 2. 查表获取概率
lut_probs = torch.sigmoid(lut_params)
selected_probs = lut_probs[indices]

# 3. STE二值化
output = STE.apply(selected_probs)
```

### 损失函数
使用CrossEntropyLoss，直接对概率logit计算损失：
```python
loss = criterion(output_logits, target)
```

## 📈 训练监控

使用TensorBoard监控训练过程：
```bash
tensorboard --logdir logs/
```

监控指标：
- 训练损失
- 训练准确率
- 测试准确率
- 学习率变化

## 🔬 实验建议

### 提高准确率的方法
1. **增加输入分辨率**: 从9x9提升到16x16或28x28
2. **保留灰度信息**: 使用多级量化而非二值化
3. **增加网络容量**: 添加更多逻辑神经元
4. **改进学习率策略**: 使用余弦退火或warmup
5. **数据增强**: 添加旋转、平移等增强
6. **混合架构**: 前几层用逻辑门，后几层用传统神经元

### 消融实验
- 对比纯逻辑门网络 vs 传统CNN
- 测试不同输入分辨率的影响
- 分析逻辑门表达力限制
- 研究STE近似误差

## 📝 引用

如果本项目对您的研究有帮助，请引用相关论文。

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 👨‍💻 作者

Your Name

## 📧 联系方式

如有问题，请提交Issue或联系作者。

---

**注意**: 本项目主要用于研究和学习目的。光神经网络仍处于发展阶段，实际应用需要进一步优化。