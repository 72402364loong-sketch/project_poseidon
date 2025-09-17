"""
Data augmentation functions for Project Poseidon
包含视觉和触觉数据的增强方法，特别是水下风格的数据增强
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
import random
from typing import Optional, Tuple, List, Union


class UnderwaterStyleAugmentation:
    """
    水下风格数据增强
    模拟水下环境的视觉特征：色彩偏移、模糊、噪声等
    """
    
    def __init__(
        self,
        blue_shift_prob: float = 0.7,
        blur_prob: float = 0.5,
        noise_prob: float = 0.3,
        contrast_prob: float = 0.6,
        brightness_prob: float = 0.5
    ):
        self.blue_shift_prob = blue_shift_prob
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.contrast_prob = contrast_prob
        self.brightness_prob = brightness_prob
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """应用水下风格增强"""
        
        # 蓝色偏移 - 模拟水下光线衰减
        if random.random() < self.blue_shift_prob:
            image = self._apply_blue_shift(image)
            
        # 模糊 - 模拟水中的光线散射
        if random.random() < self.blur_prob:
            image = self._apply_underwater_blur(image)
            
        # 降低对比度 - 模拟水下能见度降低
        if random.random() < self.contrast_prob:
            image = self._reduce_contrast(image)
            
        # 调整亮度 - 模拟不同深度的光照条件
        if random.random() < self.brightness_prob:
            image = self._adjust_brightness(image)
            
        # 添加噪声 - 模拟水中的悬浮物
        if random.random() < self.noise_prob:
            image = self._add_underwater_noise(image)
            
        return image
    
    def _apply_blue_shift(self, image: Image.Image) -> Image.Image:
        """应用蓝色偏移"""
        # 转换为numpy数组
        img_array = np.array(image, dtype=np.float32)
        
        # 增强蓝色通道，减弱红色通道
        blue_factor = random.uniform(1.1, 1.4)
        red_factor = random.uniform(0.6, 0.9)
        green_factor = random.uniform(0.8, 1.1)
        
        img_array[:, :, 0] *= red_factor    # Red
        img_array[:, :, 1] *= green_factor  # Green
        img_array[:, :, 2] *= blue_factor   # Blue
        
        # 限制像素值范围
        img_array = np.clip(img_array, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _apply_underwater_blur(self, image: Image.Image) -> Image.Image:
        """应用水下模糊效果"""
        blur_radius = random.uniform(0.5, 2.0)
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    def _reduce_contrast(self, image: Image.Image) -> Image.Image:
        """降低对比度"""
        contrast_factor = random.uniform(0.7, 0.9)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(contrast_factor)
    
    def _adjust_brightness(self, image: Image.Image) -> Image.Image:
        """调整亮度"""
        brightness_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(brightness_factor)
    
    def _add_underwater_noise(self, image: Image.Image) -> Image.Image:
        """添加水下噪声"""
        img_array = np.array(image, dtype=np.float32)
        
        # 添加高斯噪声
        noise_std = random.uniform(5, 15)
        noise = np.random.normal(0, noise_std, img_array.shape)
        
        # 添加"悬浮物"噪声点
        if random.random() < 0.3:
            num_spots = random.randint(10, 50)
            for _ in range(num_spots):
                y = random.randint(0, img_array.shape[0] - 1)
                x = random.randint(0, img_array.shape[1] - 1)
                size = random.randint(1, 3)
                brightness = random.uniform(0.5, 1.5)
                
                y1, y2 = max(0, y - size), min(img_array.shape[0], y + size)
                x1, x2 = max(0, x - size), min(img_array.shape[1], x + size)
                img_array[y1:y2, x1:x2] *= brightness
        
        img_array += noise
        img_array = np.clip(img_array, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))


class TactileAugmentation:
    """
    触觉数据增强
    """
    
    def __init__(
        self,
        noise_std: float = 0.01,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        temporal_shift_range: int = 5,
        dropout_prob: float = 0.1
    ):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.temporal_shift_range = temporal_shift_range
        self.dropout_prob = dropout_prob
    
    def __call__(self, tactile_data: np.ndarray) -> np.ndarray:
        """
        应用触觉数据增强
        
        Args:
            tactile_data: 形状为 (seq_len, feature_dim) 的触觉序列
        
        Returns:
            增强后的触觉数据
        """
        data = tactile_data.copy()
        
        # 添加高斯噪声
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, data.shape)
            data += noise
        
        # 随机缩放
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
            data *= scale_factor
        
        # 时间偏移
        if self.temporal_shift_range > 0:
            shift = np.random.randint(-self.temporal_shift_range, self.temporal_shift_range + 1)
            if shift != 0:
                data = self._apply_temporal_shift(data, shift)
        
        # 随机dropout
        if self.dropout_prob > 0:
            mask = np.random.random(data.shape) > self.dropout_prob
            data *= mask
        
        return data
    
    def _apply_temporal_shift(self, data: np.ndarray, shift: int) -> np.ndarray:
        """应用时间偏移"""
        if shift > 0:
            # 向右偏移，前面填充
            shifted_data = np.zeros_like(data)
            shifted_data[shift:] = data[:-shift]
            shifted_data[:shift] = data[0]  # 用第一个值填充
        elif shift < 0:
            # 向左偏移，后面填充
            shifted_data = np.zeros_like(data)
            shifted_data[:shift] = data[-shift:]
            shifted_data[shift:] = data[-1]  # 用最后一个值填充
        else:
            shifted_data = data
            
        return shifted_data


def get_vision_transforms(
    config: dict,
    is_training: bool = True,
    underwater_augmentation: bool = False
) -> transforms.Compose:
    """
    获取视觉数据变换
    
    Args:
        config: 配置字典
        is_training: 是否为训练模式
        underwater_augmentation: 是否应用水下增强
    
    Returns:
        变换组合
    """
    transform_list = []
    
    # 基础变换
    if 'resize' in config:
        if isinstance(config['resize'], list) and len(config['resize']) == 2:
            transform_list.append(transforms.Resize(config['resize']))
        else:
            transform_list.append(transforms.Resize((224, 224)))
    
    if is_training:
        # 训练时的数据增强
        if 'random_crop' in config:
            transform_list.append(transforms.RandomCrop(config['random_crop']))
        
        if 'random_horizontal_flip' in config:
            transform_list.append(transforms.RandomHorizontalFlip(config['random_horizontal_flip']))
        
        if 'random_rotation' in config:
            transform_list.append(transforms.RandomRotation(config['random_rotation']))
        
        # 水下风格增强
        if underwater_augmentation:
            transform_list.append(UnderwaterStyleAugmentation())
        
        # 颜色抖动
        if 'color_jitter' in config:
            cj_config = config['color_jitter']
            transform_list.append(transforms.ColorJitter(
                brightness=cj_config.get('brightness', 0),
                contrast=cj_config.get('contrast', 0),
                saturation=cj_config.get('saturation', 0),
                hue=cj_config.get('hue', 0)
            ))
        
        # 高斯模糊
        if 'gaussian_blur' in config:
            gb_config = config['gaussian_blur']
            transform_list.append(transforms.GaussianBlur(
                kernel_size=gb_config.get('kernel_size', 3),
                sigma=gb_config.get('sigma', (0.1, 2.0))
            ))
    
    else:
        # 验证时只进行基础变换
        if 'center_crop' in config:
            transform_list.append(transforms.CenterCrop(config['center_crop']))
    
    # 转换为Tensor并标准化
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.get('normalize_mean', [0.485, 0.456, 0.406]),
            std=config.get('normalize_std', [0.229, 0.224, 0.225])
        )
    ])
    
    return transforms.Compose(transform_list)


def get_tactile_transforms(
    config: dict,
    is_training: bool = True
) -> Optional[TactileAugmentation]:
    """
    获取触觉数据变换
    
    Args:
        config: 配置字典
        is_training: 是否为训练模式
    
    Returns:
        触觉增强器或None
    """
    if not is_training or 'tactile' not in config:
        return None
    
    tactile_config = config['tactile']
    
    return TactileAugmentation(
        noise_std=tactile_config.get('gaussian_noise', {}).get('std', 0.01),
        scale_range=tactile_config.get('random_scale', {}).get('scale_range', (0.95, 1.05)),
        temporal_shift_range=tactile_config.get('temporal_shift', {}).get('range', 5),
        dropout_prob=tactile_config.get('dropout', {}).get('prob', 0.1)
    )


def underwater_style_augmentation() -> UnderwaterStyleAugmentation:
    """返回水下风格增强器的默认实例"""
    return UnderwaterStyleAugmentation()
