#!/usr/bin/env python3
"""
使用Kaggle Hub下载URPC2020数据集
"""

import os
import shutil
import kagglehub
from pathlib import Path


def download_urpc_dataset():
    """
    使用Kaggle Hub下载URPC2020数据集
    """
    print("🚀 开始下载URPC2020数据集...")
    
    try:
        # 下载数据集
        print("📥 正在从Kaggle下载数据集...")
        path = kagglehub.dataset_download("lywang777/urpc2020")
        print(f"✅ 数据集下载完成！")
        print(f"📁 数据集路径: {path}")
        
        return path
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("请确保：")
        print("1. 已安装kagglehub: pip install kagglehub")
        print("2. 已配置Kaggle API密钥")
        return None


def setup_dataset_structure(source_path: str, target_path: str = "data/urpc_dataset"):
    """
    设置数据集结构到项目目录
    """
    print(f"\n🔧 设置数据集结构...")
    print(f"源路径: {source_path}")
    print(f"目标路径: {target_path}")
    
    # 创建目标目录
    os.makedirs(target_path, exist_ok=True)
    
    # 查找URPC2020文件夹
    urpc_folder = None
    for item in os.listdir(source_path):
        if "urpc" in item.lower() or "2020" in item:
            urpc_folder = os.path.join(source_path, item)
            break
    
    if not urpc_folder:
        print("❌ 未找到URPC2020文件夹")
        return False
    
    print(f"找到URPC文件夹: {urpc_folder}")
    
    # 检查数据集结构
    expected_splits = ['train', 'valid', 'test']
    found_splits = []
    
    for split in expected_splits:
        split_path = os.path.join(urpc_folder, split)
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
    
    # 复制数据集到项目目录
    print(f"\n📁 复制数据集到项目目录...")
    for split in found_splits:
        source_split = os.path.join(urpc_folder, split)
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
    source_yaml = os.path.join(urpc_folder, 'data.yaml')
    if os.path.exists(source_yaml):
        target_yaml = os.path.join(target_path, 'data.yaml')
        shutil.copy2(source_yaml, target_yaml)
        print("✅ 复制了 data.yaml 配置文件")
    
    print(f"\n🎉 数据集设置完成！")
    print(f"项目数据集位置: {target_path}")
    print(f"包含划分: {', '.join(found_splits)}")
    
    return True


def verify_dataset(dataset_path: str):
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
    """
    主函数
    """
    print("🚀 URPC2020数据集下载和设置")
    print("=" * 50)
    
    # 检查是否已安装kagglehub
    try:
        import kagglehub
        print("✅ kagglehub已安装")
    except ImportError:
        print("❌ kagglehub未安装")
        print("请运行: pip install kagglehub")
        return
    
    # 下载数据集
    dataset_path = download_urpc_dataset()
    if not dataset_path:
        return
    
    # 设置数据集结构
    success = setup_dataset_structure(dataset_path)
    if not success:
        return
    
    # 验证数据集
    verify_dataset("data/urpc_dataset")
    
    print("\n📋 下一步:")
    print("1. 检查配置文件 configs/stage0_vision_finetune.yaml")
    print("2. 运行训练脚本:")
    print("   python main_finetune_vision_on_urpc.py --config configs/stage0_vision_finetune.yaml")


if __name__ == '__main__':
    main()
