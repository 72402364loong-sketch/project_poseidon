#!/usr/bin/env python3
"""
简单的URPC数据集设置脚本
直接使用URPC2020的原始YOLO格式结构
"""

import os
import shutil
import argparse


def setup_urpc_direct(source_path: str, target_path: str = "data/urpc_dataset"):
    """
    直接设置URPC数据集，保持原始YOLO格式结构
    
    Args:
        source_path: URPC2020数据集源路径
        target_path: 目标路径
    """
    
    print(f"🚀 设置URPC数据集...")
    print(f"源路径: {source_path}")
    print(f"目标路径: {target_path}")
    
    # 检查源路径
    if not os.path.exists(source_path):
        print(f"❌ 源路径不存在: {source_path}")
        return False
    
    # 创建目标目录
    os.makedirs(target_path, exist_ok=True)
    
    # 检查URPC2020结构
    expected_splits = ['train', 'valid', 'test']
    found_splits = []
    
    for split in expected_splits:
        split_path = os.path.join(source_path, split)
        if os.path.exists(split_path):
            images_dir = os.path.join(split_path, 'images')
            if os.path.exists(images_dir):
                found_splits.append(split)
                print(f"✅ 找到 {split} 划分")
            else:
                print(f"⚠️  {split} 划分缺少 images 文件夹")
        else:
            print(f"⚠️  未找到 {split} 划分")
    
    if not found_splits:
        print("❌ 未找到有效的URPC数据集结构")
        return False
    
    # 复制数据集结构
    print("\n📁 复制数据集结构...")
    for split in found_splits:
        source_split = os.path.join(source_path, split)
        target_split = os.path.join(target_path, split)
        
        print(f"复制 {split} 划分...")
        shutil.copytree(source_split, target_split, dirs_exist_ok=True)
        
        # 统计图像数量
        images_dir = os.path.join(target_split, 'images')
        if os.path.exists(images_dir):
            image_count = len([f for f in os.listdir(images_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            print(f"  - {split}: {image_count} 张图像")
    
    # 复制data.yaml文件（如果存在）
    source_yaml = os.path.join(source_path, 'data.yaml')
    if os.path.exists(source_yaml):
        target_yaml = os.path.join(target_path, 'data.yaml')
        shutil.copy2(source_yaml, target_yaml)
        print("✅ 复制了 data.yaml 配置文件")
    
    print("\n🎉 URPC数据集设置完成！")
    print(f"数据集位置: {target_path}")
    print(f"包含划分: {', '.join(found_splits)}")
    
    return True


def verify_dataset_structure(dataset_path: str):
    """
    验证数据集结构
    """
    print(f"\n🔍 验证数据集结构: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print("❌ 数据集路径不存在")
        return False
    
    splits = ['train', 'valid', 'test']
    total_images = 0
    
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            images_dir = os.path.join(split_path, 'images')
            if os.path.exists(images_dir):
                image_count = len([f for f in os.listdir(images_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                total_images += image_count
                print(f"✅ {split}: {image_count} 张图像")
            else:
                print(f"⚠️  {split}: 缺少 images 文件夹")
        else:
            print(f"⚠️  {split}: 文件夹不存在")
    
    print(f"📊 总计: {total_images} 张图像")
    
    if total_images > 0:
        print("✅ 数据集结构验证通过")
        return True
    else:
        print("❌ 数据集结构验证失败")
        return False


def main():
    parser = argparse.ArgumentParser(description='设置URPC数据集用于视觉领域适应')
    parser.add_argument('--source', type=str, required=True,
                       help='URPC2020数据集源路径')
    parser.add_argument('--target', type=str, default='data/urpc_dataset',
                       help='目标路径 (默认: data/urpc_dataset)')
    parser.add_argument('--verify-only', action='store_true',
                       help='仅验证现有数据集结构')
    
    args = parser.parse_args()
    
    if args.verify_only:
        # 仅验证现有数据集
        verify_dataset_structure(args.target)
    else:
        # 设置新数据集
        success = setup_urpc_direct(args.source, args.target)
        if success:
            # 验证设置结果
            verify_dataset_structure(args.target)
            
            print("\n📋 下一步:")
            print("1. 检查配置文件 configs/stage0_vision_finetune.yaml")
            print("2. 运行训练脚本:")
            print("   python main_finetune_vision_on_urpc.py --config configs/stage0_vision_finetune.yaml")


if __name__ == '__main__':
    main()
