#!/usr/bin/env python3
"""
URPC数据集准备脚本
用于准备阶段0.5视觉领域适应的数据集
"""

import os
import shutil
import argparse
from pathlib import Path


def prepare_urpc_dataset(source_path: str, target_path: str, use_all_splits: bool = True):
    """
    准备URPC数据集用于视觉领域适应
    
    Args:
        source_path: URPC2020数据集源路径
        target_path: 目标路径 (data/urpc_dataset)
        use_all_splits: 是否使用所有划分的数据
    """
    
    print(f"🚀 准备URPC数据集...")
    print(f"源路径: {source_path}")
    print(f"目标路径: {target_path}")
    
    # 创建目标目录
    os.makedirs(target_path, exist_ok=True)
    
    if use_all_splits:
        # 方案1: 整合所有图像到一个文件夹
        print("📦 整合所有图像用于视觉领域适应...")
        
        all_images_dir = os.path.join(target_path, "all_images")
        os.makedirs(all_images_dir, exist_ok=True)
        
        splits = ['train', 'valid', 'test']
        total_images = 0
        
        for split in splits:
            source_images_dir = os.path.join(source_path, split, 'images')
            if os.path.exists(source_images_dir):
                print(f"处理 {split} 划分...")
                
                # 复制所有图像
                for img_file in os.listdir(source_images_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        src_path = os.path.join(source_images_dir, img_file)
                        # 重命名以避免冲突
                        new_name = f"{split}_{img_file}"
                        dst_path = os.path.join(all_images_dir, new_name)
                        shutil.copy2(src_path, dst_path)
                        total_images += 1
                
                print(f"  - 复制了 {len([f for f in os.listdir(source_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])} 张图像")
        
        print(f"✅ 总共整合了 {total_images} 张图像到 {all_images_dir}")
        
    else:
        # 方案2: 保持原始结构
        print("📁 保持原始URPC数据集结构...")
        
        splits = ['train', 'valid', 'test']
        
        for split in splits:
            source_split_dir = os.path.join(source_path, split)
            target_split_dir = os.path.join(target_path, split)
            
            if os.path.exists(source_split_dir):
                print(f"复制 {split} 划分...")
                shutil.copytree(source_split_dir, target_split_dir, dirs_exist_ok=True)
                
                # 统计图像数量
                images_dir = os.path.join(target_split_dir, 'images')
                if os.path.exists(images_dir):
                    image_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                    print(f"  - {split}: {image_count} 张图像")
    
    # 复制data.yaml文件（如果存在）
    source_yaml = os.path.join(source_path, 'data.yaml')
    if os.path.exists(source_yaml):
        target_yaml = os.path.join(target_path, 'data.yaml')
        shutil.copy2(source_yaml, target_yaml)
        print("✅ 复制了 data.yaml 配置文件")
    
    print("🎉 数据集准备完成！")


def create_vision_adaptation_structure(source_path: str, target_path: str):
    """
    创建专门的视觉适应数据结构 (适配URPC2020 YOLO格式)
    """
    print("🔧 创建视觉适应数据结构...")
    
    # 创建目标目录
    os.makedirs(target_path, exist_ok=True)
    
    # 创建训练和验证目录
    train_dir = os.path.join(target_path, 'train', 'images')
    val_dir = os.path.join(target_path, 'valid', 'images')  # URPC使用valid
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 收集所有图像
    all_images = []
    splits = ['train', 'valid', 'test']  # URPC2020的实际划分
    
    for split in splits:
        images_dir = os.path.join(source_path, split, 'images')
        if os.path.exists(images_dir):
            print(f"处理 {split} 划分...")
            split_images = []
            for img_file in os.listdir(images_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    split_images.append(os.path.join(images_dir, img_file))
            all_images.extend(split_images)
            print(f"  - {split}: {len(split_images)} 张图像")
    
    print(f"总共找到 {len(all_images)} 张图像")
    
    # 按8:2比例分割
    train_count = int(len(all_images) * 0.8)
    train_images = all_images[:train_count]
    val_images = all_images[train_count:]
    
    # 复制训练图像
    print("复制训练图像...")
    for i, img_path in enumerate(train_images):
        img_name = f"train_{i:06d}_{os.path.basename(img_path)}"
        shutil.copy2(img_path, os.path.join(train_dir, img_name))
    
    # 复制验证图像
    print("复制验证图像...")
    for i, img_path in enumerate(val_images):
        img_name = f"val_{i:06d}_{os.path.basename(img_path)}"
        shutil.copy2(img_path, os.path.join(val_dir, img_name))
    
    print(f"✅ 训练集: {len(train_images)} 张图像")
    print(f"✅ 验证集: {len(val_images)} 张图像")


def main():
    parser = argparse.ArgumentParser(description='准备URPC数据集用于视觉领域适应')
    parser.add_argument('--source', type=str, required=True,
                       help='URPC2020数据集源路径')
    parser.add_argument('--target', type=str, default='data/urpc_dataset',
                       help='目标路径 (默认: data/urpc_dataset)')
    parser.add_argument('--mode', type=str, choices=['original', 'integrated', 'vision_adaptation'],
                       default='vision_adaptation',
                       help='准备模式: original(保持原结构), integrated(整合所有图像), vision_adaptation(视觉适应结构)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source):
        print(f"❌ 源路径不存在: {args.source}")
        return
    
    if args.mode == 'original':
        prepare_urpc_dataset(args.source, args.target, use_all_splits=False)
    elif args.mode == 'integrated':
        prepare_urpc_dataset(args.source, args.target, use_all_splits=True)
    elif args.mode == 'vision_adaptation':
        create_vision_adaptation_structure(args.source, args.target)
    
    print("\n📋 下一步:")
    print("1. 检查数据集结构")
    print("2. 运行训练脚本:")
    print("   python main_finetune_vision_on_urpc.py --config configs/stage0_vision_finetune.yaml")


if __name__ == '__main__':
    main()
