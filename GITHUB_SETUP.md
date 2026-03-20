# GitHub仓库创建和推送指南

## 方法一：通过GitHub网站创建仓库（推荐）

### 步骤1：在GitHub上创建新仓库

1. 访问 [GitHub](https://github.com)
2. 点击右上角的 "+" 按钮，选择 "New repository"
3. 填写仓库信息：
   - **Repository name**: `Optical-Logic-CNN`
   - **Description**: `基于光逻辑门的二值化神经网络实现`
   - **Public/Private**: 根据需要选择
   - **不要勾选** "Add a README file"（因为我们已经有了）
   - **不要勾选** "Add .gitignore"（我们已经有了）
   - **不要选择** "Choose a license"（稍后可以添加）
4. 点击 "Create repository"

### 步骤2：推送代码到GitHub

创建仓库后，GitHub会显示推送命令。在项目目录下执行以下命令：

```powershell
cd "e:\github学习\Optical-Logic-CNN"
```

然后执行GitHub显示的命令（类似下面这样，请替换为您的实际URL）：

```powershell
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Optical-Logic-CNN.git
git push -u origin main
```

**注意**：将 `YOUR_USERNAME` 替换为您的GitHub用户名。

### 步骤3：验证推送成功

```powershell
git remote -v
```

应该显示：
```
origin  https://github.com/YOUR_USERNAME/Optical-Logic-CNN.git (fetch)
origin  https://github.com/YOUR_USERNAME/Optical-Logic-CNN.git (push)
```

## 方法二：使用GitHub CLI（需要先安装）

### 安装GitHub CLI

1. 下载GitHub CLI: https://cli.github.com/
2. 安装后，在PowerShell中执行：

```powershell
gh auth login
```

按照提示完成GitHub账户认证。

### 创建并推送仓库

```powershell
cd "e:\github学习\Optical-Logic-CNN"
gh repo create Optical-Logic-CNN --public --source=. --remote=origin --push
```

## 推送后的后续操作

### 1. 添加GitHub Actions（可选）

创建 `.github/workflows/train.yml` 文件：

```yaml
name: Train Model

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  train:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Train model
      run: |
        python examples/train.py --epochs 10 --batch_size 128
```

### 2. 添加License（可选）

在GitHub仓库页面：
1. 点击 "Add file" -> "Create new file"
2. 文件名：`LICENSE`
3. 选择 "MIT License" 模板
4. 点击 "Review and submit" -> "Commit changes"

### 3. 添加项目标签（可选）

在GitHub仓库页面：
1. 点击 "Settings" -> "Labels"
2. 添加有用的标签，如：
   - `bug` - Bug报告
   - `enhancement` - 功能增强
   - `documentation` - 文档改进
   - `good first issue` - 适合新手的任务

### 4. 设置仓库描述和主题

在GitHub仓库页面：
1. 点击 "Settings" -> "General"
2. 设置仓库描述
3. 添加主题标签（如：`machine-learning`, `neural-networks`, `optical-computing`）

## 常见问题

### Q1: 推送时提示 "fatal: remote origin already exists"

**解决方案**：
```powershell
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/Optical-Logic-CNN.git
git push -u origin main
```

### Q2: 推送时提示 "error: failed to push some refs"

**解决方案**：
```powershell
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### Q3: 想要修改commit信息

**解决方案**：
```powershell
git commit --amend -m "新的提交信息"
git push -f origin main
```

**注意**：强制推送会重写历史，仅在必要时使用。

### Q4: 想要添加.gitignore中被忽略的文件

**解决方案**：
```powershell
git add -f 文件名
git commit -m "添加文件"
git push origin main
```

## 项目结构说明

推送成功后，您的GitHub仓库将包含以下结构：

```
Optical-Logic-CNN/
├── .gitignore              # Git忽略文件配置
├── README.md               # 项目说明文档
├── requirements.txt        # Python依赖列表
├── __init__.py            # 包初始化文件
├── src/                   # 源代码目录
│   ├── __init__.py
│   ├── models/           # 模型定义
│   │   ├── __init__.py
│   │   ├── ste.py        # 直通估计器
│   │   ├── logic_unit.py # 光逻辑单元
│   │   ├── logic_conv2d.py # 光逻辑卷积层
│   │   └── olcnn.py      # 完整网络架构
│   └── utils/            # 工具函数
│       ├── __init__.py
│       ├── data_loader.py # 数据加载器
│       ├── trainer.py    # 训练器
│       └── utils.py      # 通用工具
├── examples/             # 示例脚本
│   ├── __init__.py
│   ├── train.py          # 训练脚本
│   ├── evaluate.py       # 评估脚本
│   └── visualize.py      # 可视化脚本
├── data/                 # 数据目录（.gitignore）
├── checkpoints/          # 模型检查点（.gitignore）
└── logs/                 # 训练日志（.gitignore）
```

## 下一步建议

1. **训练模型**：
   ```powershell
   python examples/train.py --epochs 200 --batch_size 256
   ```

2. **评估模型**：
   ```powershell
   python examples/evaluate.py --checkpoint checkpoints/best_model.pth
   ```

3. **可视化结果**：
   ```powershell
   python examples/visualize.py --checkpoint checkpoints/best_model.pth
   ```

4. **监控训练**：
   ```powershell
   tensorboard --logdir logs
   ```

5. **贡献代码**：
   - 创建新分支：`git checkout -b feature/your-feature`
   - 提交更改：`git commit -m "Add your feature"`
   - 推送分支：`git push origin feature/your-feature`
   - 创建Pull Request

## 联系方式

如有问题，请提交Issue或联系项目维护者。

---

**祝您使用愉快！** 🚀