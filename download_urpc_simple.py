#!/usr/bin/env python3
"""
简单的URPC2020数据集下载脚本
使用Kaggle Hub
"""

import kagglehub
import os
import shutil


def main():
    print("🚀 下载URPC2020数据集...")
    
    # 下载数据集
    print("📥 正在从Kaggle下载数据集...")
    path = kagglehub.dataset_download("lywang777/urpc2020")
    print(f"✅ 数据集下载完成！")
    print(f"📁 数据集路径: {path}")
    
    # 查找URPC2020文件夹
    urpc_folder = None
    for item in os.listdir(path):
        if "urpc" in item.lower() or "2020" in item:
            urpc_folder = os.path.join(path, item)
            break
    
    if not urpc_folder:
        print("❌ 未找到URPC2020文件夹")
        return
    
    print(f"找到URPC文件夹: {urpc_folder}")
    
    # 创建项目数据目录
    target_path = "data/urpc_dataset"
    os.makedirs(target_path, exist_ok=True)
    
    # 复制整个URPC文件夹到项目目录
    print(f"📁 复制数据集到项目目录...")
    for item in os.listdir(urpc_folder):
        source_item = os.path.join(urpc_folder, item)
        target_item = os.path.join(target_path, item)
        
        if os.path.isdir(source_item):
            print(f"复制文件夹: {item}")
            shutil.copytree(source_item, target_item, dirs_exist_ok=True)
        else:
            print(f"复制文件: {item}")
            shutil.copy2(source_item, target_item)
    
    print(f"✅ 数据集已复制到: {target_path}")
    
    # 验证数据集结构
    print("\n🔍 验证数据集结构...")
    splits = ['train', 'valid', 'test']
    total_images = 0
    
    for split in splits:
        split_path = os.path.join(target_path, split)
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
        print("🎉 数据集准备完成！")
        print("\n📋 下一步:")
        print("1. 激活虚拟环境: poseidon_env\\Scripts\\activate.bat")
        print("2. 开始训练:")
        print("   python main_finetune_vision_on_urpc.py --config configs/stage0_vision_finetune.yaml")
    else:
        print("❌ 数据集结构验证失败")


if __name__ == '__main__':
    main()
