# Project Poseidon v3.0 快速开始指南

## 🚀 一键安装

### Windows 用户
```cmd
# 双击运行或在命令行执行
setup.bat
```

### Linux/macOS 用户
```bash
# 给脚本执行权限并运行
chmod +x setup.sh
./setup.sh
```

### 跨平台 Python 脚本
```bash
# 使用 Python 脚本 (推荐)
python setup.py
```

## 🔍 验证安装

安装完成后，运行验证脚本：

```bash
python verify_setup.py
```

## 📊 准备数据

### 1. 创建数据目录
```bash
mkdir -p data/{urpc,representation,classification,policy}
```

### 2. 下载 URPC 数据集 (阶段 0.5)
```bash
# 下载到 data/urpc/ 目录
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

### 3. 准备表征学习数据 (阶段 1)
```bash
# 创建数据索引文件
# 参考 README.md 中的数据格式说明
```

## 🎯 开始训练

### 阶段 0.5: 视觉领域适应
```bash
python main_finetune_vision_on_urpc.py --config configs/stage0_vision_finetune.yaml
```

### 阶段 1: 多模态表征学习
```bash
python main_train_representation.py --config configs/stage1_representation.yaml
```

### 阶段 1.5: 物体分类器训练
```bash
python main_train_classifier.py --config configs/stage1_classifier.yaml
```

### 阶段 2: 动态策略学习
```bash
python main_train_policy.py --config configs/stage2_policy.yaml
```

## 🔧 常见问题

### 问题 1: CUDA 不可用
```bash
# 检查 CUDA 版本
nvidia-smi

# 重新安装对应版本的 PyTorch
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### 问题 2: 内存不足
```yaml
# 在配置文件中减小批次大小
training_params:
  batch_size: 16  # 从 32 减小到 16
```

### 问题 3: 依赖冲突
```bash
# 重新创建虚拟环境
deactivate
rm -rf poseidon_env
python -m venv poseidon_env
source poseidon_env/bin/activate  # Linux/macOS
# 或
.\poseidon_env\Scripts\Activate.ps1  # Windows
```

## 📚 更多信息

- 详细配置指南: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- 项目说明: [README.md](README.md)
- 技术规范: [PROJECT_SPECIFICATION_v3.0.md](PROJECT_SPECIFICATION_v3.0.md)

---

**开始您的 Project Poseidon 之旅！** 🌊🤖
