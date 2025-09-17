"""
Custom samplers for Project Poseidon
包含BalancedBatchSampler等自定义采样器
"""

import torch
import numpy as np
from torch.utils.data import Sampler, Dataset
from typing import Iterator, List, Dict, Optional, Any
import random
from collections import defaultdict


class BalancedBatchSampler(Sampler[List[int]]):
    """
    平衡批次采样器
    确保每个批次都包含来自多个不同物体/类别的样本，为对比学习提供丰富的负样本
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        samples_per_class: int = 8,
        drop_last: bool = True,
        shuffle: bool = True,
        class_key: str = 'object_id'
    ):
        """
        Args:
            dataset: 数据集
            batch_size: 批次大小
            samples_per_class: 每个类别在一个批次中的样本数
            drop_last: 是否丢弃最后一个不完整的批次
            shuffle: 是否打乱顺序
            class_key: 用于分类的键名
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.class_key = class_key
        
        # 计算需要的类别数
        self.num_classes_per_batch = max(1, batch_size // samples_per_class)
        
        # 构建类别到样本索引的映射
        self.class_to_indices = self._build_class_mapping()
        self.classes = list(self.class_to_indices.keys())
        
        # 计算批次数量
        self.num_batches = self._calculate_num_batches()
    
    def _build_class_mapping(self) -> Dict[Any, List[int]]:
        """构建类别到样本索引的映射"""
        class_to_indices = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            # 获取样本的类别信息
            if hasattr(self.dataset, 'samples') and isinstance(self.dataset.samples, list):
                # 对于RepresentationDataset
                if idx < len(self.dataset.samples):
                    sample = self.dataset.samples[idx]
                    if isinstance(sample, dict) and self.class_key in sample:
                        class_id = sample[self.class_key]
                    else:
                        class_id = 0  # 默认类别
                else:
                    class_id = 0
            else:
                # 尝试直接从数据集获取
                try:
                    sample_data = self.dataset[idx]
                    if isinstance(sample_data, tuple) and len(sample_data) >= 3:
                        metadata = sample_data[2]
                        if isinstance(metadata, dict) and self.class_key in metadata:
                            class_id = metadata[self.class_key]
                        else:
                            class_id = 0
                    else:
                        class_id = 0
                except:
                    class_id = 0
            
            class_to_indices[class_id].append(idx)
        
        # 过滤掉样本数量不足的类别
        min_samples = max(1, self.samples_per_class)
        filtered_mapping = {
            class_id: indices for class_id, indices in class_to_indices.items()
            if len(indices) >= min_samples
        }
        
        if not filtered_mapping:
            # 如果没有足够的类别，创建一个默认映射
            all_indices = list(range(len(self.dataset)))
            filtered_mapping = {0: all_indices}
        
        return filtered_mapping
    
    def _calculate_num_batches(self) -> int:
        """计算批次数量"""
        if not self.classes:
            return 0
        
        # 计算每个类别可以提供的批次数
        min_batches_per_class = min(
            len(indices) // self.samples_per_class 
            for indices in self.class_to_indices.values()
        )
        
        # 总批次数取决于类别数量和每个类别的批次数
        total_batches = min_batches_per_class * len(self.classes) // self.num_classes_per_batch
        
        return max(1, total_batches)
    
    def __iter__(self) -> Iterator[List[int]]:
        """生成批次"""
        # 为每个类别创建索引的副本并打乱
        class_indices = {}
        for class_id, indices in self.class_to_indices.items():
            class_indices[class_id] = indices.copy()
            if self.shuffle:
                random.shuffle(class_indices[class_id])
        
        # 为每个类别维护当前位置
        class_positions = {class_id: 0 for class_id in self.classes}
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            # 选择这个批次要使用的类别
            if self.shuffle:
                selected_classes = random.sample(
                    self.classes, 
                    min(self.num_classes_per_batch, len(self.classes))
                )
            else:
                selected_classes = self.classes[:self.num_classes_per_batch]
            
            # 从每个选定的类别中采样
            for class_id in selected_classes:
                indices = class_indices[class_id]
                pos = class_positions[class_id]
                
                # 获取这个类别的样本
                class_samples = []
                for _ in range(self.samples_per_class):
                    if pos < len(indices):
                        class_samples.append(indices[pos])
                        pos += 1
                    else:
                        # 如果当前类别的样本用完了，重新开始
                        if self.shuffle:
                            random.shuffle(indices)
                        pos = 0
                        if pos < len(indices):
                            class_samples.append(indices[pos])
                            pos += 1
                
                class_positions[class_id] = pos
                batch_indices.extend(class_samples)
            
            # 如果批次大小不够，随机填充
            while len(batch_indices) < self.batch_size:
                remaining_needed = self.batch_size - len(batch_indices)
                
                # 从所有类别中随机选择样本填充
                all_available_indices = []
                for class_id, indices in class_indices.items():
                    all_available_indices.extend(indices)
                
                if all_available_indices:
                    if self.shuffle:
                        additional_samples = random.choices(
                            all_available_indices, 
                            k=min(remaining_needed, len(all_available_indices))
                        )
                    else:
                        additional_samples = all_available_indices[:remaining_needed]
                    
                    batch_indices.extend(additional_samples)
                else:
                    break
            
            # 确保批次大小正确
            if len(batch_indices) >= self.batch_size:
                batch_indices = batch_indices[:self.batch_size]
            
            if len(batch_indices) == self.batch_size or not self.drop_last:
                if self.shuffle:
                    random.shuffle(batch_indices)
                yield batch_indices
    
    def __len__(self) -> int:
        return self.num_batches


class SequentialBatchSampler(Sampler[List[int]]):
    """
    序列批次采样器
    用于策略学习阶段，确保轨迹数据的时序性
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True
    ):
        """
        Args:
            dataset: 数据集
            batch_size: 批次大小
            drop_last: 是否丢弃最后一个不完整的批次
            shuffle: 是否打乱轨迹顺序（但保持轨迹内部顺序）
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # 获取所有轨迹的索引
        self.trajectory_indices = list(range(len(dataset)))
        
    def __iter__(self) -> Iterator[List[int]]:
        """生成批次"""
        indices = self.trajectory_indices.copy()
        
        if self.shuffle:
            random.shuffle(indices)
        
        # 按批次大小分组
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            if len(batch_indices) == self.batch_size or not self.drop_last:
                yield batch_indices
    
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.trajectory_indices) // self.batch_size
        else:
            return (len(self.trajectory_indices) + self.batch_size - 1) // self.batch_size


class DAggerSampler(Sampler[List[int]]):
    """
    DAgger采样器
    在DAgger迭代过程中平衡专家数据和策略生成数据的比例
    """
    
    def __init__(
        self,
        expert_indices: List[int],
        policy_indices: List[int],
        batch_size: int,
        expert_ratio: float = 0.5,
        shuffle: bool = True,
        drop_last: bool = True
    ):
        """
        Args:
            expert_indices: 专家数据的索引
            policy_indices: 策略生成数据的索引
            batch_size: 批次大小
            expert_ratio: 专家数据在批次中的比例
            shuffle: 是否打乱顺序
            drop_last: 是否丢弃最后一个不完整的批次
        """
        self.expert_indices = expert_indices
        self.policy_indices = policy_indices
        self.batch_size = batch_size
        self.expert_ratio = expert_ratio
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # 计算每个批次中专家数据和策略数据的数量
        self.expert_per_batch = int(batch_size * expert_ratio)
        self.policy_per_batch = batch_size - self.expert_per_batch
        
        # 计算总的批次数
        self.num_batches = self._calculate_num_batches()
    
    def _calculate_num_batches(self) -> int:
        """计算批次数量"""
        if not self.expert_indices and not self.policy_indices:
            return 0
        
        # 基于可用数据计算最大批次数
        expert_batches = len(self.expert_indices) // max(1, self.expert_per_batch) if self.expert_indices else 0
        policy_batches = len(self.policy_indices) // max(1, self.policy_per_batch) if self.policy_indices else 0
        
        if expert_batches == 0 and policy_batches == 0:
            return 0
        elif expert_batches == 0:
            return policy_batches
        elif policy_batches == 0:
            return expert_batches
        else:
            return min(expert_batches, policy_batches)
    
    def __iter__(self) -> Iterator[List[int]]:
        """生成批次"""
        # 复制并打乱索引
        expert_indices = self.expert_indices.copy()
        policy_indices = self.policy_indices.copy()
        
        if self.shuffle:
            random.shuffle(expert_indices)
            random.shuffle(policy_indices)
        
        expert_pos = 0
        policy_pos = 0
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            # 添加专家数据
            for _ in range(self.expert_per_batch):
                if expert_pos < len(expert_indices):
                    batch_indices.append(expert_indices[expert_pos])
                    expert_pos += 1
                else:
                    # 如果专家数据用完，重新开始
                    if self.shuffle:
                        random.shuffle(expert_indices)
                    expert_pos = 0
                    if expert_pos < len(expert_indices):
                        batch_indices.append(expert_indices[expert_pos])
                        expert_pos += 1
            
            # 添加策略数据
            for _ in range(self.policy_per_batch):
                if policy_pos < len(policy_indices):
                    batch_indices.append(policy_indices[policy_pos])
                    policy_pos += 1
                else:
                    # 如果策略数据用完，重新开始
                    if self.shuffle:
                        random.shuffle(policy_indices)
                    policy_pos = 0
                    if policy_pos < len(policy_indices):
                        batch_indices.append(policy_indices[policy_pos])
                        policy_pos += 1
            
            if len(batch_indices) == self.batch_size or not self.drop_last:
                if self.shuffle:
                    random.shuffle(batch_indices)
                yield batch_indices
    
    def __len__(self) -> int:
        return self.num_batches
