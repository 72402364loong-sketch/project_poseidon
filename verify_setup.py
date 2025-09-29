#!/usr/bin/env python3
"""
Project Poseidon v3.0 环境验证脚本
检查所有依赖是否正确安装和配置
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python版本过低: {version.major}.{version.minor}.{version.micro}")
        print("需要Python 3.8或更高版本")
        return False

def check_cuda():
    """检查CUDA支持"""
    print("\n🔧 检查CUDA支持...")
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA版本: {torch.version.cuda}")
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠️ CUDA不可用，将使用CPU")
            return False
    except ImportError:
        print("❌ PyTorch未安装")
        return False

def check_dependencies():
    """检查依赖包"""
    print("\n📦 检查依赖包...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("yaml", "PyYAML"),
        ("tqdm", "TQDM"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
        ("scikit-learn", "Scikit-learn"),
        ("timm", "PyTorch Image Models"),
    ]
    
    optional_packages = [
        ("wandb", "Weights & Biases"),
        ("tensorboard", "TensorBoard"),
        ("jupyter", "Jupyter"),
    ]
    
    all_required = True
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name}")
            all_required = False
    
    print("\n可选依赖:")
    for package, name in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️ {name} (可选)")
    
    return all_required

def check_project_modules():
    """检查项目模块"""
    print("\n🏗️ 检查项目模块...")
    
    project_modules = [
        ("models.representation_model", "RepresentationModel"),
        ("models.classifier", "ObjectClassifier"),
        ("models.policy_model", "PolicyModel"),
        ("data_loader.dataset", "数据集类"),
        ("engine.trainer", "训练器"),
        ("engine.evaluator", "评估器"),
        ("engine.losses", "损失函数"),
    ]
    
    all_modules = True
    for module, name in project_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            all_modules = False
    
    return all_modules

def check_config_files():
    """检查配置文件"""
    print("\n⚙️ 检查配置文件...")
    
    config_files = [
        "configs/stage0_vision_finetune.yaml",
        "configs/stage1_representation.yaml", 
        "configs/stage2_policy.yaml",
    ]
    
    all_configs = True
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file} 不存在")
            all_configs = False
    
    return all_configs

def check_directories():
    """检查目录结构"""
    print("\n📁 检查目录结构...")
    
    required_dirs = [
        "data",
        "data/urpc",
        "data/representation",
        "data/classification", 
        "data/policy",
        "outputs",
        "outputs/logs",
        "outputs/checkpoints",
        "outputs/results",
        "models",
        "engine",
        "data_loader",
        "configs",
    ]
    
    all_dirs = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}")
        else:
            print(f"❌ {directory} 不存在")
            all_dirs = False
    
    return all_dirs

def check_gpu_memory():
    """检查GPU内存"""
    print("\n💾 检查GPU内存...")
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {total_memory:.1f} GB")
                
                if total_memory < 8:
                    print(f"⚠️ GPU {i} 内存不足8GB，可能影响训练")
                else:
                    print(f"✅ GPU {i} 内存充足")
            return True
        else:
            print("⚠️ 无可用GPU")
            return False
    except Exception as e:
        print(f"❌ GPU检查失败: {e}")
        return False

def run_quick_test():
    """运行快速测试"""
    print("\n🧪 运行快速测试...")
    
    try:
        # 测试模型创建
        from models.representation_model import RepresentationModel
        model = RepresentationModel()
        print("✅ RepresentationModel创建成功")
        
        # 测试数据加载器
        from data_loader.dataset import URPCDataset
        print("✅ 数据加载器导入成功")
        
        # 测试损失函数
        from engine.losses import InfoNCELoss
        loss_fn = InfoNCELoss()
        print("✅ 损失函数创建成功")
        
        return True
    except Exception as e:
        print(f"❌ 快速测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔍 Project Poseidon v3.0 环境验证")
    print("=" * 50)
    
    checks = [
        ("Python版本", check_python_version),
        ("CUDA支持", check_cuda),
        ("依赖包", check_dependencies),
        ("项目模块", check_project_modules),
        ("配置文件", check_config_files),
        ("目录结构", check_directories),
        ("GPU内存", check_gpu_memory),
        ("快速测试", run_quick_test),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name}检查出错: {e}")
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 验证结果总结:")
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print(f"\n通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有检查通过！环境配置正确。")
        print("\n下一步:")
        print("1. 准备数据集")
        print("2. 开始训练:")
        print("   python main_finetune_vision_on_urpc.py --config configs/stage0_vision_finetune.yaml")
    else:
        print(f"\n⚠️ {total-passed}个检查失败，请修复后重新验证。")
        print("\n常见解决方案:")
        print("1. 重新运行安装脚本")
        print("2. 检查CUDA版本兼容性")
        print("3. 确保所有依赖正确安装")
        print("4. 参考 SETUP_GUIDE.md 获取详细指导")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
