@echo off
echo 🚀 Project Poseidon - HuggingFace 连接问题修复工具
echo ================================================

echo.
echo 🔧 步骤 1: 设置镜像源环境变量
set HF_ENDPOINT=https://hf-mirror.com
set HUGGINGFACE_HUB_CACHE=./cache/huggingface
set TRANSFORMERS_CACHE=./cache/transformers
set HF_HOME=./cache/hf_home
set TORCH_HOME=./cache/torch

echo ✅ 环境变量已设置

echo.
echo 🔧 步骤 2: 创建缓存目录
mkdir cache\huggingface 2>nul
mkdir cache\transformers 2>nul
mkdir cache\hf_home 2>nul
mkdir cache\torch 2>nul
mkdir checkpoints\stage0_offline 2>nul
mkdir logs\stage0_offline 2>nul

echo ✅ 缓存目录已创建

echo.
echo 🔧 步骤 3: 运行网络配置脚本
python setup_network_config.py

echo.
echo 📋 修复完成！现在你可以尝试以下选项:
echo.
echo 选项 1 - 使用原始配置 (如果网络正常):
echo python main_finetune_vision_on_urpc.py --config configs/stage0_vision_finetune.yaml
echo.
echo 选项 2 - 使用离线配置 (推荐):
echo python main_finetune_vision_on_urpc.py --config configs/stage0_offline.yaml
echo.
echo 选项 3 - 预下载模型 (可选):
echo python download_timm_models.py
echo.

pause
