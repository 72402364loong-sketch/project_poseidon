"""
Vision Adaptation Dataset for Stage 0.5
专门用于视觉领域适应的数据集类，不需要标签
"""

import os
import random
from typing import List, Tuple, Optional, Any
from torch.utils.data import Dataset
from PIL import Image


class VisionAdaptationDataset(Dataset):
    """
    视觉领域适应数据集
    专门用于阶段0.5的视觉领域适应，不需要标签
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        transform: Optional[Any] = None,
        use_all_splits: bool = False
    ):
        """
        Args:
            data_path: 数据集根路径
            split: 数据集划分 ('train', 'val', 'test', 'all')
            transform: 数据变换
            use_all_splits: 是否使用所有划分的数据
        """
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.use_all_splits = use_all_splits
        
        # 加载图像路径列表
        self.image_paths = self._load_image_paths()
        
    def _load_image_paths(self) -> List[str]:
        """加载图像路径列表"""
        image_paths = []
        
        if self.use_all_splits or self.split == 'all':
            # 使用所有划分的数据
            splits = ['train', 'val', 'test']
        else:
            splits = [self.split]
        
        for split in splits:
            split_path = os.path.join(self.data_path, split)
            if os.path.exists(split_path):
                # 方法1: 如果存在images子文件夹
                images_dir = os.path.join(split_path, 'images')
                if os.path.exists(images_dir):
                    for img_file in os.listdir(images_dir):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            image_paths.append(os.path.join(images_dir, img_file))
                else:
                    # 方法2: 直接在split文件夹中查找图像
                    for img_file in os.listdir(split_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            image_paths.append(os.path.join(split_path, img_file))
        
        # 随机打乱
        random.shuffle(image_paths)
        
        return image_paths
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """
        返回图像和伪标签（用于兼容现有训练代码）
        
        Returns:
            (image, pseudo_label): 图像和伪标签（始终为0）
        """
        img_path = self.image_paths[idx]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个空白图像作为fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 返回伪标签（始终为0，因为视觉适应不需要真实标签）
        pseudo_label = 0
        
        return image, pseudo_label
    
    def get_dataset_info(self) -> dict:
        """获取数据集信息"""
        return {
            'total_images': len(self.image_paths),
            'split': self.split,
            'use_all_splits': self.use_all_splits,
            'data_path': self.data_path
        }


class URPCVisionAdaptationDataset(VisionAdaptationDataset):
    """
    URPC数据集的视觉适应版本
    专门处理URPC2020数据集结构 (YOLO格式)
    """
    
    def _load_image_paths(self) -> List[str]:
        """加载URPC数据集的图像路径"""
        image_paths = []
        
        if self.use_all_splits or self.split == 'all':
            # 使用所有划分的数据
            splits = ['train', 'valid', 'test']  # URPC使用valid而不是val
        else:
            # 处理split名称映射
            split_mapping = {'val': 'valid', 'validation': 'valid'}
            splits = [split_mapping.get(self.split, self.split)]
        
        for split in splits:
            split_path = os.path.join(self.data_path, split)
            if os.path.exists(split_path):
                # URPC数据集结构: split/images/
                images_dir = os.path.join(split_path, 'images')
                if os.path.exists(images_dir):
                    print(f"Loading images from {images_dir}")
                    for img_file in os.listdir(images_dir):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            image_paths.append(os.path.join(images_dir, img_file))
                    print(f"  - Found {len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])} images")
        
        # 随机打乱
        random.shuffle(image_paths)
        
        print(f"Total images loaded: {len(image_paths)}")
        return image_paths
