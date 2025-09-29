#!/bin/bash
# Project Poseidon v3.0 Linux/macOS 环境配置脚本

set -e  # 遇到错误立即退出

echo "🚀 Project Poseidon v3.0 环境配置"
echo "================================================"

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装"
    echo "请先安装Python 3.8或更高版本"
    exit 1
fi

echo "✅ Python已安装"
python3 --version

# 检查Python版本
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python版本过低: $PYTHON_VERSION"
    echo "需要Python 3.8或更高版本"
    exit 1
fi

# 创建虚拟环境
echo ""
echo "📦 创建虚拟环境..."
if [ -d "poseidon_env" ]; then
    echo "虚拟环境已存在，跳过创建"
else
    python3 -m venv poseidon_env
    echo "✅ 虚拟环境创建成功"
fi

# 激活虚拟环境
echo ""
echo "🔄 激活虚拟环境..."
source poseidon_env/bin/activate

# 升级pip
echo ""
echo "⬆️ 升级pip..."
python -m pip install --upgrade pip

# 安装PyTorch
echo ""
echo "🔧 安装PyTorch..."
echo "尝试安装CUDA 11.8版本..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || {
    echo "⚠️ CUDA 11.8版本安装失败，尝试CUDA 12.1版本..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || {
        echo "⚠️ CUDA版本安装失败，尝试CPU版本..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    }
}

# 安装项目依赖
echo ""
echo "📦 安装项目依赖..."
pip install -r requirements.txt

# 创建目录
echo ""
echo "📁 创建项目目录..."
mkdir -p data/{urpc,representation,classification,policy}
mkdir -p outputs/{logs,checkpoints,results}

# 验证安装
echo ""
echo "🧪 验证安装..."
python -c "
import torch
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU数量:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
"

python -c "
from models.representation_model import RepresentationModel
from data_loader.dataset import URPCDataset
from engine.losses import InfoNCELoss
print('✅ 项目模块导入成功')
"

echo ""
echo "🎉 环境配置完成！"
echo ""
echo "下一步:"
echo "1. 准备数据集 (参考 SETUP_GUIDE.md)"
echo "2. 配置YAML文件"
echo "3. 开始训练:"
echo "   python main_finetune_vision_on_urpc.py --config configs/stage0_vision_finetune.yaml"
echo ""
echo "要激活虚拟环境，请运行:"
echo "   source poseidon_env/bin/activate"
echo ""
