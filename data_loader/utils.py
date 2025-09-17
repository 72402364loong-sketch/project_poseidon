"""
Utility functions for data processing in Project Poseidon
包含数据预处理、同步、几何计算等辅助函数
"""

import numpy as np
import cv2
import json
import os
from typing import Tuple, List, Dict, Optional, Any, Union
import torch
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def calculate_target_3d_coordinates(
    left_image: np.ndarray,
    right_image: np.ndarray,
    camera_params: Dict[str, float],
    stereo_params: Dict[str, Any] = None
) -> Tuple[float, float, float]:
    """
    使用立体匹配算法计算目标的精确3D坐标
    
    Args:
        left_image: 左目图像
        right_image: 右目图像
        camera_params: 相机参数字典，包含baseline, focal_length, cx, cy
        stereo_params: 立体匹配参数
    
    Returns:
        目标的3D坐标 (X, Y, Z)
    """
    if stereo_params is None:
        stereo_params = {
            'algorithm': 'sgbm',
            'min_disparity': 0,
            'num_disparities': 64,
            'block_size': 11
        }
    
    # 转换为灰度图像
    if len(left_image.shape) == 3:
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
    else:
        left_gray = left_image
        
    if len(right_image.shape) == 3:
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)
    else:
        right_gray = right_image
    
    # 立体匹配
    if stereo_params['algorithm'].lower() == 'sgbm':
        stereo = cv2.StereoSGBM_create(
            minDisparity=stereo_params['min_disparity'],
            numDisparities=stereo_params['num_disparities'],
            blockSize=stereo_params['block_size'],
            P1=8 * 3 * stereo_params['block_size'] ** 2,
            P2=32 * 3 * stereo_params['block_size'] ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    else:
        # 使用基本的BM算法
        stereo = cv2.StereoBM_create(
            numDisparities=stereo_params['num_disparities'],
            blockSize=stereo_params['block_size']
        )
    
    # 计算视差图
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    
    # 找到有效视差的中心点作为目标位置
    valid_disparity = disparity > stereo_params['min_disparity']
    
    if np.any(valid_disparity):
        # 计算有效区域的质心
        y_coords, x_coords = np.where(valid_disparity)
        disparity_values = disparity[valid_disparity]
        
        # 使用加权平均计算目标中心
        weights = disparity_values / np.sum(disparity_values)
        center_x = np.average(x_coords, weights=weights)
        center_y = np.average(y_coords, weights=weights)
        center_disparity = np.average(disparity_values, weights=weights)
    else:
        # 如果没有有效视差，使用图像中心
        center_x = left_image.shape[1] / 2
        center_y = left_image.shape[0] / 2
        center_disparity = stereo_params['num_disparities'] / 2
    
    # 转换为3D坐标
    baseline = camera_params['baseline']
    focal_length = camera_params['focal_length']
    cx = camera_params.get('cx', left_image.shape[1] / 2)
    cy = camera_params.get('cy', left_image.shape[0] / 2)
    
    # 避免除零
    if center_disparity <= 0:
        center_disparity = 1.0
    
    # 计算3D坐标
    Z = baseline * focal_length / center_disparity
    X = (center_x - cx) * Z / focal_length
    Y = (center_y - cy) * Z / focal_length
    
    return float(X), float(Y), float(Z)


def synchronize_timestamps(
    vision_timestamps: List[float],
    tactile_timestamps: List[float],
    tolerance: float = 1e-3
) -> List[Tuple[int, int]]:
    """
    同步视觉和触觉数据的时间戳
    
    Args:
        vision_timestamps: 视觉数据时间戳列表
        tactile_timestamps: 触觉数据时间戳列表
        tolerance: 同步容差（秒）
    
    Returns:
        同步的索引对列表 [(vision_idx, tactile_idx), ...]
    """
    vision_timestamps = np.array(vision_timestamps)
    tactile_timestamps = np.array(tactile_timestamps)
    
    # 计算时间戳之间的距离矩阵
    distance_matrix = cdist(
        vision_timestamps.reshape(-1, 1),
        tactile_timestamps.reshape(-1, 1),
        metric='euclidean'
    )
    
    # 使用匈牙利算法找到最优匹配
    vision_indices, tactile_indices = linear_sum_assignment(distance_matrix)
    
    # 过滤掉超出容差的匹配
    synchronized_pairs = []
    for v_idx, t_idx in zip(vision_indices, tactile_indices):
        if distance_matrix[v_idx, t_idx] <= tolerance:
            synchronized_pairs.append((int(v_idx), int(t_idx)))
    
    return synchronized_pairs


def load_tactile_sequence(
    tactile_file_path: str,
    seq_len: int = 100,
    feature_dim: int = 54
) -> np.ndarray:
    """
    加载触觉序列数据
    
    Args:
        tactile_file_path: 触觉数据文件路径
        seq_len: 序列长度
        feature_dim: 特征维度
    
    Returns:
        形状为 (seq_len, feature_dim) 的触觉序列
    """
    if not os.path.exists(tactile_file_path):
        # 如果文件不存在，返回零序列
        return np.zeros((seq_len, feature_dim), dtype=np.float32)
    
    # 根据文件扩展名选择加载方法
    file_ext = os.path.splitext(tactile_file_path)[1].lower()
    
    if file_ext == '.json':
        # JSON格式
        with open(tactile_file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'tactile_data' in data:
            tactile_data = np.array(data['tactile_data'], dtype=np.float32)
        elif isinstance(data, list):
            tactile_data = np.array(data, dtype=np.float32)
        else:
            tactile_data = np.zeros((seq_len, feature_dim), dtype=np.float32)
            
    elif file_ext in ['.npy', '.npz']:
        # NumPy格式
        if file_ext == '.npy':
            tactile_data = np.load(tactile_file_path)
        else:
            with np.load(tactile_file_path) as data:
                if 'tactile_data' in data:
                    tactile_data = data['tactile_data']
                else:
                    # 取第一个数组
                    tactile_data = list(data.values())[0]
        
        tactile_data = tactile_data.astype(np.float32)
        
    elif file_ext == '.csv':
        # CSV格式
        tactile_data = np.loadtxt(tactile_file_path, delimiter=',', dtype=np.float32)
        
    else:
        # 不支持的格式，返回零序列
        tactile_data = np.zeros((seq_len, feature_dim), dtype=np.float32)
    
    # 确保数据形状正确
    if tactile_data.ndim == 1:
        # 如果是1D数组，重塑为2D
        if len(tactile_data) == feature_dim:
            # 单个时间步
            tactile_data = tactile_data.reshape(1, -1)
        else:
            # 尝试重塑
            tactile_data = tactile_data.reshape(-1, feature_dim)
    
    current_seq_len = tactile_data.shape[0]
    
    # 调整序列长度
    if current_seq_len > seq_len:
        # 截断
        tactile_data = tactile_data[:seq_len]
    elif current_seq_len < seq_len:
        # 填充
        if current_seq_len > 0:
            # 使用最后一个值填充
            last_frame = tactile_data[-1:].repeat(seq_len - current_seq_len, axis=0)
            tactile_data = np.concatenate([tactile_data, last_frame], axis=0)
        else:
            # 如果没有数据，创建零序列
            tactile_data = np.zeros((seq_len, feature_dim), dtype=np.float32)
    
    # 确保特征维度正确
    if tactile_data.shape[1] != feature_dim:
        if tactile_data.shape[1] < feature_dim:
            # 填充零
            padding = np.zeros((tactile_data.shape[0], feature_dim - tactile_data.shape[1]), dtype=np.float32)
            tactile_data = np.concatenate([tactile_data, padding], axis=1)
        else:
            # 截断
            tactile_data = tactile_data[:, :feature_dim]
    
    return tactile_data


def normalize_tactile_data(
    tactile_data: np.ndarray,
    method: str = 'minmax',
    axis: int = 0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    标准化触觉数据
    
    Args:
        tactile_data: 触觉数据
        method: 标准化方法 ('minmax', 'zscore', 'robust')
        axis: 标准化轴
    
    Returns:
        标准化后的数据和标准化参数
    """
    if method == 'minmax':
        data_min = np.min(tactile_data, axis=axis, keepdims=True)
        data_max = np.max(tactile_data, axis=axis, keepdims=True)
        
        # 避免除零
        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0
        
        normalized_data = (tactile_data - data_min) / data_range
        norm_params = {'method': 'minmax', 'min': data_min, 'max': data_max}
        
    elif method == 'zscore':
        data_mean = np.mean(tactile_data, axis=axis, keepdims=True)
        data_std = np.std(tactile_data, axis=axis, keepdims=True)
        
        # 避免除零
        data_std[data_std == 0] = 1.0
        
        normalized_data = (tactile_data - data_mean) / data_std
        norm_params = {'method': 'zscore', 'mean': data_mean, 'std': data_std}
        
    elif method == 'robust':
        data_median = np.median(tactile_data, axis=axis, keepdims=True)
        data_mad = np.median(np.abs(tactile_data - data_median), axis=axis, keepdims=True)
        
        # 避免除零
        data_mad[data_mad == 0] = 1.0
        
        normalized_data = (tactile_data - data_median) / data_mad
        norm_params = {'method': 'robust', 'median': data_median, 'mad': data_mad}
        
    else:
        # 不支持的方法，返回原数据
        normalized_data = tactile_data
        norm_params = {'method': 'none'}
    
    return normalized_data, norm_params


def create_data_splits(
    data_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    创建数据集划分
    
    Args:
        data_path: 数据路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    
    Returns:
        数据划分字典
    """
    np.random.seed(seed)
    
    # 获取所有数据文件
    all_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(('.json', '.npy', '.npz')):
                rel_path = os.path.relpath(os.path.join(root, file), data_path)
                all_files.append(rel_path)
    
    # 打乱文件列表
    np.random.shuffle(all_files)
    
    # 计算划分点
    total_files = len(all_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    # 创建划分
    splits = {
        'train': all_files[:train_end],
        'val': all_files[train_end:val_end],
        'test': all_files[val_end:] if test_ratio > 0 else []
    }
    
    return splits


def save_data_splits(
    splits: Dict[str, List[str]],
    output_path: str
) -> None:
    """
    保存数据划分到文件
    
    Args:
        splits: 数据划分字典
        output_path: 输出路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)


def load_data_splits(splits_file: str) -> Dict[str, List[str]]:
    """
    从文件加载数据划分
    
    Args:
        splits_file: 划分文件路径
    
    Returns:
        数据划分字典
    """
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    return splits


def validate_data_integrity(
    data_path: str,
    required_files: List[str] = None
) -> Dict[str, Any]:
    """
    验证数据完整性
    
    Args:
        data_path: 数据路径
        required_files: 必需的文件列表
    
    Returns:
        验证结果字典
    """
    result = {
        'valid': True,
        'missing_files': [],
        'corrupted_files': [],
        'statistics': {}
    }
    
    if required_files is None:
        required_files = []
    
    # 检查必需文件
    for file_path in required_files:
        full_path = os.path.join(data_path, file_path)
        if not os.path.exists(full_path):
            result['missing_files'].append(file_path)
            result['valid'] = False
    
    # 检查数据文件完整性
    data_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(('.json', '.npy', '.npz')):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, data_path)
                data_files.append(rel_path)
                
                # 尝试加载文件检查是否损坏
                try:
                    if file.lower().endswith('.json'):
                        with open(full_path, 'r') as f:
                            json.load(f)
                    elif file.lower().endswith('.npy'):
                        np.load(full_path)
                    elif file.lower().endswith('.npz'):
                        with np.load(full_path) as data:
                            pass
                except Exception as e:
                    result['corrupted_files'].append({
                        'file': rel_path,
                        'error': str(e)
                    })
                    result['valid'] = False
    
    # 统计信息
    result['statistics'] = {
        'total_files': len(data_files),
        'missing_files': len(result['missing_files']),
        'corrupted_files': len(result['corrupted_files'])
    }
    
    return result
