# OLCNN_Simulation

光学逻辑卷积神经网络（Optical Logic Convolutional Neural Network, OLCNN）的仿真实现。

## 项目简介

OLCNN是一种基于光学计算原理的神经网络模型，通过模拟光学系统的特性，实现高效的二值化计算。本项目实现了OLCNN的核心组件，包括：

- 改进版直通估计器（STE）
- 光逻辑单元（OpticalLogicUnit）
- 光谱编码光逻辑单元（SpectralOpticalUnit）
- 逻辑卷积层（LogicConv2d）
- 完整的训练流程

## 项目结构

```
OLCNN_Simulation/
├── configs/            # 配置文件
│   └── config.yaml     # 主配置文件
├── data/               # 数据集目录
│   └── MNIST/          # MNIST数据集
├── models/             # 模型定义
│   ├── __init__.py
│   ├── layers.py       # 核心层实现
│   └── networks.py     # 网络架构
├── outputs/            # 模型输出目录
├── scripts/            # 脚本文件
│   └── train.py        # 训练脚本
├── trainers/           # 训练器
│   ├── __init__.py
│   └── trainer.py      # 训练器实现
├── utils/              # 工具函数
│   ├── __init__.py
│   ├── config_loader.py # 配置加载
│   └── data_loader.py   # 数据加载
├── README.md           # 项目说明
└── requirements.txt    # 依赖库
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

配置文件位于 `configs/config.yaml`，主要配置项包括：

- **model**: 模型配置
  - `num_classes`: 类别数（默认为10，对应MNIST数据集）
  - `num_spectral_channels`: 光谱通道数（默认为16）
  - `use_spectral_encoding`: 是否使用光谱编码（默认为True）
  - `use_ste`: 是否使用直通估计器（默认为True）

- **training**: 训练配置
  - `epochs`: 训练轮数（默认为50）
  - `learning_rate`: 学习率（默认为0.0001）
  - `weight_decay`: 权重衰减（默认为0.0001）
  - `scheduler`: 学习率调度器（默认为"step"，可选"cosine"）

- **其他配置**
  - `batch_size`: 批次大小（默认为64）
  - `input_size`: 输入尺寸（默认为9，对应光学系统输入）
  - `device`: 训练设备（默认为"cuda"，可选"cpu"）
  - `seed`: 随机种子（默认为42）

## 如何运行

### 训练模型

```bash
python scripts/train.py --config configs/config.yaml
```

### 主要功能

1. **梯度检测**：验证STE是否生效
2. **梯度裁剪**：防止梯度爆炸
3. **权重钳制**：防止神经元饱和
4. **死亡单元复活**：定期检查并重置饱和的神经元
5. **神经元饱和检查**：监控模型训练状态

## 模型架构

OLCNN的架构设计参考典型的光学神经网络论文结构：

```
Input (9x9) -> [Optical Conv] -> [Pool] -> [Optical Conv] -> [Pool] -> [FC] -> Output
```

- **输入层**：9x9的二值化输入
- **第一层逻辑卷积**：16个输出通道，3x3卷积核
- **池化层**：2x2最大池化
- **第二层逻辑卷积**：32个输出通道，3x3卷积核
- **全连接层**：输出类别概率

## 关键技术点

### 1. 改进版直通估计器（STE）

- **前向传播**：使用硬阶跃函数输出0/1
- **反向传播**：使用指数衰减梯度掩码 `torch.exp(-torch.abs(input) / temp)`
- **技术创新**：引入温度参数控制梯度衰减速率

### 2. 光逻辑单元

- **基本原理**：实现光学系统的逻辑计算
- **权重钳制**：防止权重过大导致神经元饱和
- **初始化策略**：使用极小范围的权重初始化

### 3. 光谱编码

- **多通道并行**：通过多个光谱通道提高表达能力
- **通道融合**：使用可学习的通道权重进行融合

### 4. 训练技巧

- **梯度裁剪**：防止梯度爆炸
- **权重钳制**：确保计算在合理区间内进行
- **死亡单元复活**：定期重置饱和的神经元
- **学习率调度**：支持余弦退火和步进衰减

## 结果分析

训练过程中会输出以下信息：

- 训练损失和准确率
- 验证损失和准确率
- 神经元饱和情况
- 死亡单元复活情况
- 最佳模型保存

## 注意事项

1. 本项目使用MNIST数据集作为默认数据集
2. 输入尺寸固定为9x9，模拟光学系统的输入要求
3. 训练过程中会自动下载MNIST数据集到data目录
4. 模型权重和检查点会保存在outputs目录

## 参考资料

- 光学神经网络相关论文
- PyTorch官方文档
- MNIST数据集说明
