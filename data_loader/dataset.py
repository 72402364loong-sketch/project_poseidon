"""
Dataset classes for Project Poseidon
包含三个核心数据集类：URPCDataset, RepresentationDataset, PolicyDataset
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from typing import List, Tuple, Dict, Optional, Any
from .utils import calculate_target_3d_coordinates, synchronize_timestamps, load_tactile_sequence


class URPCDataset(Dataset):
    """
    URPC水下数据集，用于阶段0.5的视觉领域适应
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        transform: Optional[Any] = None,
        class_names: List[str] = None
    ):
        """
        Args:
            data_path: URPC数据集路径
            split: 数据集划分 ('train', 'val', 'test')
            transform: 数据变换
            class_names: 类别名称列表，默认为['holothurian', 'echinus', 'scallop', 'starfish']
        """
        self.data_path = data_path
        self.split = split
        self.transform = transform
        
        if class_names is None:
            self.class_names = ['holothurian', 'echinus', 'scallop', 'starfish']
        else:
            self.class_names = class_names
            
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        
        # 加载数据列表
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Tuple[str, int]]:
        """加载样本列表"""
        samples = []
        split_file = os.path.join(self.data_path, f'{self.split}.txt')
        
        if os.path.exists(split_file):
            # 如果存在划分文件，直接加载
            with open(split_file, 'r') as f:
                for line in f:
                    img_path, label = line.strip().split()
                    full_img_path = os.path.join(self.data_path, img_path)
                    samples.append((full_img_path, int(label)))
        else:
            # 否则根据文件夹结构加载
            for class_name in self.class_names:
                class_dir = os.path.join(self.data_path, class_name)
                if os.path.exists(class_dir):
                    label = self.class_to_idx[class_name]
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(class_dir, img_name)
                            samples.append((img_path, label))
                            
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
            
        return image, label


class RepresentationDataset(Dataset):
    """
    表征学习数据集，用于阶段1的多模态对比学习
    包含同步的视觉和触觉数据对
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        vision_transform: Optional[Any] = None,
        tactile_transform: Optional[Any] = None,
        tactile_seq_len: int = 100,
        stereo_mode: bool = True
    ):
        """
        Args:
            data_path: 数据集路径
            split: 数据集划分
            vision_transform: 视觉数据变换
            tactile_transform: 触觉数据变换
            tactile_seq_len: 触觉序列长度
            stereo_mode: 是否使用双目视觉
        """
        self.data_path = data_path
        self.split = split
        self.vision_transform = vision_transform
        self.tactile_transform = tactile_transform
        self.tactile_seq_len = tactile_seq_len
        self.stereo_mode = stereo_mode
        
        # 加载数据索引
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict[str, Any]]:
        """加载样本索引"""
        samples = []
        split_file = os.path.join(self.data_path, f'{self.split}_index.json')
        
        with open(split_file, 'r') as f:
            data_index = json.load(f)
            
        for item in data_index:
            samples.append({
                'object_id': item['object_id'],
                'timestamp': item['timestamp'],
                'vision_path': os.path.join(self.data_path, item['vision_path']),
                'tactile_path': os.path.join(self.data_path, item['tactile_path']),
                'stereo_left_path': os.path.join(self.data_path, item.get('stereo_left_path', '')),
                'stereo_right_path': os.path.join(self.data_path, item.get('stereo_right_path', '')),
                'metadata': item.get('metadata', {})
            })
            
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        sample = self.samples[idx]
        
        # 加载视觉数据
        if self.stereo_mode and os.path.exists(sample['stereo_left_path']) and os.path.exists(sample['stereo_right_path']):
            # 双目视觉模式
            left_img = Image.open(sample['stereo_left_path']).convert('RGB')
            right_img = Image.open(sample['stereo_right_path']).convert('RGB')
            
            # 水平拼接双目图像
            left_array = np.array(left_img)
            right_array = np.array(right_img)
            stereo_image = np.concatenate([left_array, right_array], axis=1)
            vision_data = Image.fromarray(stereo_image)
        else:
            # 单目视觉模式
            vision_data = Image.open(sample['vision_path']).convert('RGB')
        
        # 加载触觉数据
        tactile_data = load_tactile_sequence(
            sample['tactile_path'], 
            seq_len=self.tactile_seq_len
        )
        
        # 应用变换
        if self.vision_transform:
            vision_data = self.vision_transform(vision_data)
            
        if self.tactile_transform:
            tactile_data = self.tactile_transform(tactile_data)
        
        # 转换为tensor
        if not isinstance(vision_data, torch.Tensor):
            vision_data = torch.from_numpy(np.array(vision_data)).float()
            
        if not isinstance(tactile_data, torch.Tensor):
            tactile_data = torch.from_numpy(tactile_data).float()
            
        metadata = {
            'object_id': sample['object_id'],
            'timestamp': sample['timestamp'],
            'metadata': sample['metadata']
        }
        
        return vision_data, tactile_data, metadata


class PolicyDataset(Dataset):
    """
    策略学习数据集，用于阶段2的DAgger迭代学习
    处理轨迹序列数据
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        trajectory_length: int = 50,
        vision_transform: Optional[Any] = None,
        tactile_transform: Optional[Any] = None,
        tactile_seq_len: int = 100,
        include_geometry: bool = True
    ):
        """
        Args:
            data_path: 数据集路径
            split: 数据集划分
            trajectory_length: 轨迹长度
            vision_transform: 视觉数据变换
            tactile_transform: 触觉数据变换
            tactile_seq_len: 触觉序列长度
            include_geometry: 是否包含几何特征
        """
        self.data_path = data_path
        self.split = split
        self.trajectory_length = trajectory_length
        self.vision_transform = vision_transform
        self.tactile_transform = tactile_transform
        self.tactile_seq_len = tactile_seq_len
        self.include_geometry = include_geometry
        
        # 加载轨迹数据
        self.trajectories = self._load_trajectories()
        
    def _load_trajectories(self) -> List[Dict[str, Any]]:
        """加载轨迹数据"""
        trajectories = []
        split_file = os.path.join(self.data_path, f'{self.split}_trajectories.json')
        
        with open(split_file, 'r') as f:
            traj_index = json.load(f)
            
        for traj_info in traj_index:
            traj_data = {
                'trajectory_id': traj_info['trajectory_id'],
                'episode_id': traj_info['episode_id'],
                'task_type': traj_info['task_type'],
                'success': traj_info.get('success', False),
                'length': traj_info['length'],
                'data_path': os.path.join(self.data_path, traj_info['data_path']),
                'metadata': traj_info.get('metadata', {})
            }
            trajectories.append(traj_data)
            
        return trajectories
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        trajectory = self.trajectories[idx]
        
        # 加载轨迹数据文件
        with open(trajectory['data_path'], 'r') as f:
            traj_data = json.load(f)
            
        # 提取状态和动作序列
        states = []
        actions = []
        
        for step in traj_data['steps'][:self.trajectory_length]:
            # 构建状态向量
            state_vector = self._build_state_vector(step)
            states.append(state_vector)
            
            # 提取动作
            action = np.array(step['action'], dtype=np.float32)
            actions.append(action)
            
        # 填充或截断到指定长度
        states = self._pad_or_truncate_sequence(states, self.trajectory_length)
        actions = self._pad_or_truncate_sequence(actions, self.trajectory_length)
        
        return {
            'states': torch.from_numpy(np.array(states)).float(),
            'actions': torch.from_numpy(np.array(actions)).float(),
            'trajectory_id': trajectory['trajectory_id'],
            'success': trajectory['success'],
            'length': min(trajectory['length'], self.trajectory_length)
        }
    
    def _build_state_vector(self, step_data: Dict[str, Any]) -> np.ndarray:
        """构建状态向量"""
        state_components = []
        
        # 视觉特征 (假设已经通过表征模型提取)
        if 'vision_features' in step_data:
            vision_features = np.array(step_data['vision_features'], dtype=np.float32)
            state_components.append(vision_features)
            
        # 触觉特征 (假设已经通过表征模型提取)
        if 'tactile_features' in step_data:
            tactile_features = np.array(step_data['tactile_features'], dtype=np.float32)
            state_components.append(tactile_features)
            
        # 几何特征 (3D坐标)
        if self.include_geometry and 'target_3d_coords' in step_data:
            geometry_features = np.array(step_data['target_3d_coords'], dtype=np.float32)
            state_components.append(geometry_features)
            
        # 拼接所有特征
        state_vector = np.concatenate(state_components, axis=0)
        return state_vector
    
    def _pad_or_truncate_sequence(self, sequence: List[np.ndarray], target_length: int) -> List[np.ndarray]:
        """填充或截断序列到目标长度"""
        current_length = len(sequence)
        
        if current_length > target_length:
            # 截断
            return sequence[:target_length]
        elif current_length < target_length:
            # 填充（使用最后一个元素）
            if current_length > 0:
                last_element = sequence[-1]
                padding = [last_element.copy() for _ in range(target_length - current_length)]
                return sequence + padding
            else:
                # 如果序列为空，返回零向量
                dummy_element = np.zeros_like(sequence[0]) if sequence else np.zeros(1)
                return [dummy_element.copy() for _ in range(target_length)]
        else:
            return sequence


class ClassificationDataset(Dataset):
    """
    分类数据集，用于阶段1.5的多模态对象分类
    基于RepresentationDataset，但增加了类别标签
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        vision_transform: Optional[Any] = None,
        tactile_transform: Optional[Any] = None,
        tactile_seq_len: int = 100,
        stereo_mode: bool = True,
        num_classes: int = 15
    ):
        """
        Args:
            data_path: 数据集路径
            split: 数据集划分
            vision_transform: 视觉数据变换
            tactile_transform: 触觉数据变换
            tactile_seq_len: 触觉序列长度
            stereo_mode: 是否使用双目视觉
            num_classes: 物体类别数量
        """
        self.data_path = data_path
        self.split = split
        self.vision_transform = vision_transform
        self.tactile_transform = tactile_transform
        self.tactile_seq_len = tactile_seq_len
        self.stereo_mode = stereo_mode
        self.num_classes = num_classes
        
        # 加载数据索引（包含类别标签）
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict[str, Any]]:
        """加载样本索引（包含类别标签）"""
        samples = []
        split_file = os.path.join(self.data_path, f'{self.split}_classification_index.json')
        
        if os.path.exists(split_file):
            # 如果存在分类索引文件，直接加载
            with open(split_file, 'r') as f:
                data_index = json.load(f)
        else:
            # 否则从表征学习索引文件加载，并添加默认标签
            # 这里假设object_id可以作为类别标签，实际使用时需要根据具体情况调整
            repr_split_file = os.path.join(self.data_path, f'{self.split}_index.json')
            with open(repr_split_file, 'r') as f:
                data_index = json.load(f)
            
            # 为每个样本添加类别标签（这里使用object_id的哈希值作为类别）
            for item in data_index:
                # 使用object_id的哈希值来确定类别，确保一致性
                class_id = hash(item['object_id']) % self.num_classes
                item['class_id'] = class_id
                
        for item in data_index:
            samples.append({
                'object_id': item['object_id'],
                'timestamp': item['timestamp'],
                'class_id': item.get('class_id', 0),  # 默认类别为0
                'vision_path': os.path.join(self.data_path, item['vision_path']),
                'tactile_path': os.path.join(self.data_path, item['tactile_path']),
                'stereo_left_path': os.path.join(self.data_path, item.get('stereo_left_path', '')),
                'stereo_right_path': os.path.join(self.data_path, item.get('stereo_right_path', '')),
                'metadata': item.get('metadata', {})
            })
            
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        返回视觉数据、触觉数据和类别标签
        
        Returns:
            (vision_data, tactile_data, class_label)
        """
        sample = self.samples[idx]
        
        # 加载视觉数据（与RepresentationDataset相同的逻辑）
        if self.stereo_mode and os.path.exists(sample['stereo_left_path']) and os.path.exists(sample['stereo_right_path']):
            # 双目视觉模式
            left_img = Image.open(sample['stereo_left_path']).convert('RGB')
            right_img = Image.open(sample['stereo_right_path']).convert('RGB')
            
            # 水平拼接双目图像
            left_array = np.array(left_img)
            right_array = np.array(right_img)
            stereo_image = np.concatenate([left_array, right_array], axis=1)
            vision_data = Image.fromarray(stereo_image)
        else:
            # 单目视觉模式
            vision_data = Image.open(sample['vision_path']).convert('RGB')
        
        # 加载触觉数据
        tactile_data = load_tactile_sequence(
            sample['tactile_path'], 
            seq_len=self.tactile_seq_len
        )
        
        # 应用变换
        if self.vision_transform:
            vision_data = self.vision_transform(vision_data)
            
        if self.tactile_transform:
            tactile_data = self.tactile_transform(tactile_data)
        
        # 转换为tensor
        if not isinstance(vision_data, torch.Tensor):
            vision_data = torch.from_numpy(np.array(vision_data)).float()
            
        if not isinstance(tactile_data, torch.Tensor):
            tactile_data = torch.from_numpy(tactile_data).float()
        
        # 获取类别标签
        class_label = sample['class_id']
        
        return vision_data, tactile_data, class_label
    
    def get_class_distribution(self) -> Dict[int, int]:
        """获取类别分布统计"""
        class_counts = {}
        for sample in self.samples:
            class_id = sample['class_id']
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        return class_counts
    
    def print_class_distribution(self) -> None:
        """打印类别分布"""
        class_counts = self.get_class_distribution()
        print(f"Class distribution for {self.split} split:")
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            percentage = count / len(self.samples) * 100
            print(f"  Class {class_id}: {count} samples ({percentage:.1f}%)")
        print(f"Total samples: {len(self.samples)}")