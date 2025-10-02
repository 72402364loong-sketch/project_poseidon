#!/usr/bin/env python3
"""
网络配置脚本 - 解决 HuggingFace 和 timm 连接问题
Network Configuration Script - Fix HuggingFace and timm connection issues
"""

import os
import sys
import subprocess
import urllib.request
import socket
from pathlib import Path


def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    """检查网络连接"""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False


def check_huggingface_access():
    """检查 HuggingFace 访问"""
    try:
        urllib.request.urlopen('https://huggingface.co', timeout=10)
        return True
    except:
        return False


def setup_pip_mirror():
    """设置 pip 镜像源"""
    print("🔧 设置 pip 镜像源...")
    
    pip_conf_dir = Path.home() / ".pip"
    pip_conf_dir.mkdir(exist_ok=True)
    
    pip_conf_content = """[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
"""
    
    pip_conf_file = pip_conf_dir / "pip.conf"
    with open(pip_conf_file, 'w', encoding='utf-8') as f:
        f.write(pip_conf_content)
    
    print(f"✅ pip 配置已保存到: {pip_conf_file}")


def setup_huggingface_mirror():
    """设置 HuggingFace 镜像源"""
    print("🔧 设置 HuggingFace 镜像源...")
    
    # 设置环境变量
    mirror_url = "https://hf-mirror.com"
    
    # 创建环境变量设置脚本
    env_script_content = f"""# HuggingFace 镜像源配置
export HF_ENDPOINT={mirror_url}
export HUGGINGFACE_HUB_CACHE=./cache/huggingface
export TRANSFORMERS_CACHE=./cache/transformers
export HF_HOME=./cache/hf_home
"""
    
    with open("set_hf_mirror.sh", 'w', encoding='utf-8') as f:
        f.write(env_script_content)
    
    # Windows 批处理文件
    bat_content = f"""@echo off
set HF_ENDPOINT={mirror_url}
set HUGGINGFACE_HUB_CACHE=./cache/huggingface
set TRANSFORMERS_CACHE=./cache/transformers
set HF_HOME=./cache/hf_home
echo HuggingFace 镜像源已设置
"""
    
    with open("set_hf_mirror.bat", 'w', encoding='utf-8') as f:
        f.write(bat_content)
    
    print("✅ HuggingFace 镜像源配置文件已创建:")
    print("   - Linux/Mac: source set_hf_mirror.sh")
    print("   - Windows: set_hf_mirror.bat")


def setup_timm_offline_cache():
    """设置 timm 离线缓存"""
    print("🔧 设置 timm 离线缓存...")
    
    # 创建缓存目录
    cache_dir = Path("cache/timm")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置 timm 缓存环境变量
    os.environ['TORCH_HOME'] = str(cache_dir.parent / "torch")
    
    print(f"✅ timm 缓存目录已创建: {cache_dir}")
    
    # 创建预下载脚本
    download_script = """#!/usr/bin/env python3
'''
预下载 timm 模型脚本
Pre-download timm models script
'''

import timm
import torch
import os
from pathlib import Path

def download_vit_models():
    '''下载常用的 ViT 模型'''
    models = [
        'vit_base_patch16_224',
        'vit_small_patch16_224',
        'vit_large_patch16_224'
    ]
    
    cache_dir = Path("cache/torch")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置缓存目录
    os.environ['TORCH_HOME'] = str(cache_dir)
    
    for model_name in models:
        try:
            print(f"📥 下载模型: {model_name}")
            model = timm.create_model(model_name, pretrained=True)
            print(f"✅ {model_name} 下载完成")
        except Exception as e:
            print(f"❌ {model_name} 下载失败: {e}")
    
    print("🎉 模型下载完成！")

if __name__ == '__main__':
    download_vit_models()
"""
    
    with open("download_timm_models.py", 'w', encoding='utf-8') as f:
        f.write(download_script)
    
    print("✅ 模型预下载脚本已创建: download_timm_models.py")


def create_offline_training_config():
    """创建离线训练配置"""
    print("🔧 创建离线训练配置...")
    
    offline_config = """# 离线训练配置
# Offline Training Configuration

# 数据参数
data_params:
  urpc_dataset_path: "data/urpc_dataset"
  train_split: 0.8
  val_split: 0.2
  batch_size: 32
  num_workers: 4
  pin_memory: true
  use_all_splits_for_training: true
  ignore_labels: true
  split_mapping:
    val: "valid"
    validation: "valid"

# 模型参数 - 离线模式
model_params:
  model_name: "vit_base_patch16_224"
  pretrained: false  # 离线模式下设为 false
  num_classes: 4
  freeze_layers: 0   # 不使用预训练权重时建议设为 0
  
# 训练参数 - 调整学习率
training_params:
  epochs: 100        # 增加训练轮数
  learning_rate: 1e-3  # 提高学习率
  weight_decay: 0.01
  optimizer: "adamw"
  warmup_epochs: 10    # 增加预热轮数
  
# 学习率调度器参数
scheduler_params:
  type: "cosine"
  T_max: 100
  eta_min: 1e-6

# 数据增强参数 - 增强数据增强
augmentation_params:
  underwater_style: true
  color_jitter:
    brightness: 0.4
    contrast: 0.4
    saturation: 0.4
    hue: 0.2
  gaussian_blur:
    kernel_size: 3
    sigma: [0.1, 2.0]
  random_rotation: 20
  random_horizontal_flip: 0.5
  
# 检查点参数
checkpoint_params:
  output_dir: "checkpoints/stage0_offline"
  save_freq: 10
  save_best: true
  
# 日志参数
logging_params:
  log_freq: 10
  tensorboard_dir: "logs/stage0_offline"
  
# 设备参数
device_params:
  use_cuda: true
  mixed_precision: true
"""
    
    with open("configs/stage0_offline.yaml", 'w', encoding='utf-8') as f:
        f.write(offline_config)
    
    print("✅ 离线训练配置已创建: configs/stage0_offline.yaml")


def main():
    print("🚀 Project Poseidon 网络配置工具")
    print("=" * 50)
    
    # 检查网络连接
    print("🔍 检查网络连接...")
    if check_internet_connection():
        print("✅ 网络连接正常")
    else:
        print("❌ 网络连接异常")
    
    # 检查 HuggingFace 访问
    print("🔍 检查 HuggingFace 访问...")
    if check_huggingface_access():
        print("✅ HuggingFace 访问正常")
    else:
        print("❌ HuggingFace 访问受限")
    
    print("\n🛠️  开始配置...")
    
    # 设置镜像源
    setup_pip_mirror()
    setup_huggingface_mirror()
    setup_timm_offline_cache()
    create_offline_training_config()
    
    print("\n📋 使用说明:")
    print("1. 如果网络正常，直接运行:")
    print("   python main_finetune_vision_on_urpc.py --config configs/stage0_vision_finetune.yaml")
    
    print("\n2. 如果网络受限，使用离线配置:")
    print("   python main_finetune_vision_on_urpc.py --config configs/stage0_offline.yaml")
    
    print("\n3. 预下载模型（可选）:")
    print("   python download_timm_models.py")
    
    print("\n4. 设置镜像源环境变量:")
    print("   Windows: set_hf_mirror.bat")
    print("   Linux/Mac: source set_hf_mirror.sh")
    
    print("\n🎉 配置完成！")


if __name__ == '__main__':
    main()
