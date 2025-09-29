#!/usr/bin/env python3
"""
Project Poseidon v3.0 快速安装脚本
自动检测系统环境并安装必要的依赖
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, check=True):
    """运行命令并处理错误"""
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python版本不兼容: {version.major}.{version.minor}")
        print("需要Python 3.8或更高版本")
        return False
    print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    return True

def check_cuda():
    """检查CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.version.cuda}")
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠️  CUDA不可用，将使用CPU版本")
            return False
    except ImportError:
        print("⚠️  PyTorch未安装，无法检查CUDA")
        return False

def install_pytorch():
    """安装PyTorch"""
    print("\n🔧 安装PyTorch...")
    
    # 检查CUDA版本
    cuda_available = check_cuda()
    
    if cuda_available:
        # 尝试安装CUDA版本
        commands = [
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        ]
        
        for cmd in commands:
            print(f"尝试安装命令: {cmd}")
            success, stdout, stderr = run_command(cmd, check=False)
            if success:
                print("✅ PyTorch安装成功")
                return True
            else:
                print(f"❌ 安装失败: {stderr}")
        
        print("⚠️  CUDA版本安装失败，尝试CPU版本")
    
    # 安装CPU版本
    cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    success, stdout, stderr = run_command(cmd)
    if success:
        print("✅ PyTorch CPU版本安装成功")
        return True
    else:
        print(f"❌ PyTorch安装失败: {stderr}")
        return False

def install_requirements():
    """安装项目依赖"""
    print("\n📦 安装项目依赖...")
    
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt文件不存在")
        return False
    
    success, stdout, stderr = run_command("pip install -r requirements.txt")
    if success:
        print("✅ 项目依赖安装成功")
        return True
    else:
        print(f"❌ 依赖安装失败: {stderr}")
        return False

def create_directories():
    """创建必要的目录"""
    print("\n📁 创建项目目录...")
    
    directories = [
        "data/urpc",
        "data/representation", 
        "data/classification",
        "data/policy",
        "outputs/logs",
        "outputs/checkpoints",
        "outputs/results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {directory}")

def verify_installation():
    """验证安装"""
    print("\n🧪 验证安装...")
    
    test_imports = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("yaml", "PyYAML"),
        ("tqdm", "TQDM"),
        ("matplotlib", "Matplotlib")
    ]
    
    all_success = True
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name}")
            all_success = False
    
    # 测试项目模块
    try:
        from models.representation_model import RepresentationModel
        from data_loader.dataset import URPCDataset
        from engine.losses import InfoNCELoss
        print("✅ 项目模块")
    except ImportError as e:
        print(f"❌ 项目模块: {e}")
        all_success = False
    
    return all_success

def main():
    """主函数"""
    print("🚀 Project Poseidon v3.0 环境配置")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 升级pip
    print("\n⬆️  升级pip...")
    run_command("python -m pip install --upgrade pip")
    
    # 安装PyTorch
    if not install_pytorch():
        print("❌ PyTorch安装失败，请手动安装")
        sys.exit(1)
    
    # 安装项目依赖
    if not install_requirements():
        print("❌ 项目依赖安装失败，请检查requirements.txt")
        sys.exit(1)
    
    # 创建目录
    create_directories()
    
    # 验证安装
    if verify_installation():
        print("\n🎉 环境配置完成！")
        print("\n下一步:")
        print("1. 准备数据集 (参考 SETUP_GUIDE.md)")
        print("2. 配置YAML文件")
        print("3. 开始训练:")
        print("   python main_finetune_vision_on_urpc.py --config configs/stage0_vision_finetune.yaml")
    else:
        print("\n⚠️  安装验证失败，请检查错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()
