# Project Poseidon v3.0 环境配置指导

## 📋 目录
- [系统要求](#系统要求)
- [环境配置步骤](#环境配置步骤)
- [依赖安装](#依赖安装)
- [数据准备](#数据准备)
- [配置验证](#配置验证)
- [常见问题](#常见问题)
- [快速测试](#快速测试)

## 🖥️ 系统要求

### 最低配置
- **操作系统**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8 - 3.10
- **GPU**: NVIDIA GTX 1080 (8GB VRAM) 或更高
- **内存**: 16GB RAM
- **存储**: 100GB 可用空间
- **CUDA**: 11.8+ (推荐)

### 推荐配置
- **操作系统**: Ubuntu 20.04 LTS
- **Python**: 3.9
- **GPU**: NVIDIA RTX 3080/4080 (12GB+ VRAM)
- **内存**: 32GB RAM
- **存储**: 500GB NVMe SSD
- **CUDA**: 12.1

## 🚀 环境配置步骤

### 步骤 1: 克隆项目

```bash
# 克隆项目到本地
git clone <https://github.com/72402364loong-sketch/project_poseidon>
cd project_poseidon

# 检查项目结构
ls -la
```

### 步骤 2: 创建虚拟环境

#### Windows (PowerShell)
```powershell
# 创建虚拟环境
python -m venv poseidon_env

# 激活虚拟环境
.\poseidon_env\Scripts\Activate.ps1

# 如果遇到执行策略问题，运行：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Linux/macOS
```bash
# 创建虚拟环境
python -m venv poseidon_env

# 激活虚拟环境
source poseidon_env/bin/activate
```

### 步骤 3: 升级基础工具

```bash
# 升级 pip
python -m pip install --upgrade pip

# 安装基础工具
pip install wheel setuptools
```

## 📦 依赖安装

### 步骤 1: 安装 PyTorch

根据您的 CUDA 版本选择对应的 PyTorch 安装命令：

#### CUDA 11.8
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

#### CUDA 12.1
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu121
```

#### CPU 版本 (仅用于测试)
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

### 步骤 2: 安装项目依赖

```bash
# 安装项目依赖
pip install -r requirements.txt
```

### 步骤 3: 安装额外依赖 (可选)

```bash
# 安装开发工具
pip install jupyter ipykernel

# 安装可视化工具
pip install tensorboard wandb

# 安装代码质量工具
pip install black flake8 mypy
```

## 📊 数据准备

### 步骤 1: 创建数据目录结构

```bash
# 创建数据目录
mkdir -p data/{urpc,representation,policy,classification}

# 创建输出目录
mkdir -p outputs/{logs,checkpoints,results}
```

### 步骤 2: 准备数据集

#### URPC 数据集 (阶段 0.5)
```bash
# 下载 URPC 数据集到 data/urpc/
# 数据集结构：
data/urpc/
├── train/
│   ├── holothurian/
│   ├── echinus/
│   ├── scallop/
│   └── starfish/
├── val/
└── test/
```

#### 表征学习数据集 (阶段 1)
```bash
# 创建表征学习数据索引
# 数据格式参考 README.md 中的数据格式说明
data/representation/
├── train_index.json
├── val_index.json
├── test_index.json
└── raw_data/
    ├── images/
    └── tactile/
```

#### 分类数据集 (阶段 1.5)
```bash
# 分类数据集与表征学习数据集结构相同
# 但需要包含类别标签信息
data/classification/
├── train_index.json
├── val_index.json
└── test_index.json
```

#### 策略学习数据集 (阶段 2)
```bash
# 策略学习轨迹数据
data/policy/
├── trajectories/
│   ├── traj_001.json
│   ├── traj_002.json
│   └── ...
└── index.json
```

## ⚙️ 配置验证

### 步骤 1: 验证 PyTorch 安装

```python
# 运行 Python 验证脚本
python -c "
import torch
import torchvision
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
    print(f'当前 GPU: {torch.cuda.get_device_name(0)}')
"
```

### 步骤 2: 验证项目依赖

```python
# 验证主要依赖
python -c "
try:
    import torch
    import torchvision
    import timm
    import numpy as np
    import cv2
    import PIL
    import yaml
    import wandb
    print('✅ 所有依赖安装成功!')
except ImportError as e:
    print(f'❌ 依赖安装失败: {e}')
"
```

### 步骤 3: 检查配置文件

```bash
# 检查配置文件是否存在
ls configs/

# 验证配置文件格式
python -c "
import yaml
import os
config_files = ['stage0_vision_finetune.yaml', 'stage1_representation.yaml', 'stage2_policy.yaml']
for config_file in config_files:
    if os.path.exists(f'configs/{config_file}'):
        with open(f'configs/{config_file}', 'r') as f:
            config = yaml.safe_load(f)
        print(f'✅ {config_file} 配置有效')
    else:
        print(f'❌ {config_file} 配置文件缺失')
"
```

## 🧪 快速测试

### 测试 1: 模型导入测试

```python
# 测试模型导入
python -c "
from models.representation_model import RepresentationModel
from models.classifier import ObjectClassifier
from models.policy_model import PolicyModel
print('✅ 模型导入成功!')
"
```

### 测试 2: 数据加载器测试

```python
# 测试数据加载器
python -c "
from data_loader.dataset import URPCDataset, RepresentationDataset, ClassificationDataset, PolicyDataset
print('✅ 数据加载器导入成功!')
"
```

### 测试 3: 训练引擎测试

```python
# 测试训练引擎
python -c "
from engine.trainer import train_representation_epoch
from engine.evaluator import evaluate_representation_epoch
from engine.losses import InfoNCELoss
print('✅ 训练引擎导入成功!')
"
```

## 🔧 常见问题

### 问题 1: CUDA 版本不匹配

**症状**: `RuntimeError: CUDA runtime error`

**解决方案**:
```bash
# 检查 CUDA 版本
nvidia-smi

# 重新安装对应版本的 PyTorch
pip uninstall torch torchvision torchaudio
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### 问题 2: 内存不足

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```yaml
# 在配置文件中减小批次大小
training_params:
  batch_size: 16  # 从 32 减小到 16
  gradient_accumulation_steps: 2  # 增加梯度累积
```

### 问题 3: 依赖冲突

**症状**: `ImportError: cannot import name`

**解决方案**:
```bash
# 重新创建虚拟环境
deactivate
rm -rf poseidon_env
python -m venv poseidon_env
source poseidon_env/bin/activate  # Linux/macOS
# 或
.\poseidon_env\Scripts\Activate.ps1  # Windows

# 重新安装依赖
pip install -r requirements.txt
```

### 问题 4: 配置文件错误

**症状**: `yaml.scanner.ScannerError`

**解决方案**:
```bash
# 验证 YAML 语法
python -c "
import yaml
with open('configs/stage1_representation.yaml', 'r') as f:
    yaml.safe_load(f)
print('配置文件语法正确')
"
```

## 🎯 下一步

环境配置完成后，您可以：

1. **开始训练阶段 0.5**: 视觉领域适应
   ```bash
   python main_finetune_vision_on_urpc.py --config configs/stage0_vision_finetune.yaml
   ```

2. **开始训练阶段 1**: 多模态表征学习
   ```bash
   python main_train_representation.py --config configs/stage1_representation.yaml
   ```

3. **开始训练阶段 1.5**: 物体分类器
   ```bash
   python main_train_classifier.py --config configs/stage1_classifier.yaml
   ```

4. **开始训练阶段 2**: 动态策略学习
   ```bash
   python main_train_policy.py --config configs/stage2_policy.yaml
   ```

## 📞 获取帮助

如果遇到问题，请：

1. 检查 [常见问题](#常见问题) 部分
2. 查看项目 [README.md](README.md)
3. 查看详细 [项目规范](PROJECT_SPECIFICATION_v3.0.md)
4. 提交 Issue 到项目仓库

---

**祝您使用愉快！** 🚀
