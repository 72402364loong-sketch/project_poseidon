#!/bin/bash

echo "🚀 Project Poseidon - HuggingFace 连接问题修复工具"
echo "================================================"

echo ""
echo "🔧 步骤 1: 设置镜像源环境变量"
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_CACHE=./cache/huggingface
export TRANSFORMERS_CACHE=./cache/transformers
export HF_HOME=./cache/hf_home
export TORCH_HOME=./cache/torch

echo "✅ 环境变量已设置"

echo ""
echo "🔧 步骤 2: 创建缓存目录"
mkdir -p cache/huggingface
mkdir -p cache/transformers
mkdir -p cache/hf_home
mkdir -p cache/torch
mkdir -p checkpoints/stage0_offline
mkdir -p logs/stage0_offline

echo "✅ 缓存目录已创建"

echo ""
echo "🔧 步骤 3: 运行网络配置脚本"
python setup_network_config.py

echo ""
echo "📋 修复完成！现在你可以尝试以下选项:"
echo ""
echo "选项 1 - 使用原始配置 (如果网络正常):"
echo "python main_finetune_vision_on_urpc.py --config configs/stage0_vision_finetune.yaml"
echo ""
echo "选项 2 - 使用离线配置 (推荐):"
echo "python main_finetune_vision_on_urpc.py --config configs/stage0_offline.yaml"
echo ""
echo "选项 3 - 预下载模型 (可选):"
echo "python download_timm_models.py"
echo ""
