#!/usr/bin/env python3
"""
Robot Demo Script for Project Poseidon
在真实机器人上运行完整的视觉-触觉融合操控演示
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import time
import logging
from datetime import datetime
import json
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.representation_model import HybridRepresentationModel
from models.policy_model import PolicyModel
from robot.interface import RobotInterface, VisionSensor, TactileSensor, RobotCommand, SensorData
from data_loader.utils import calculate_target_3d_coordinates


def setup_logging(log_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"robot_demo_{timestamp}.log")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_models(config: dict, device: torch.device) -> tuple:
    """加载预训练模型"""
    
    # 加载表征模型
    rep_model_path = config['model_paths']['representation_model']
    representation_model = HybridRepresentationModel()
    
    if os.path.exists(rep_model_path):
        checkpoint = torch.load(rep_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            representation_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            representation_model.load_state_dict(checkpoint)
        print(f"Loaded representation model from: {rep_model_path}")
    else:
        raise FileNotFoundError(f"Representation model not found: {rep_model_path}")
    
    # 加载策略模型
    policy_model_path = config['model_paths']['policy_model']
    
    # 从配置中获取策略模型参数
    policy_config = config.get('policy_model_params', {})
    policy_model = PolicyModel(
        vision_feature_dim=policy_config.get('vision_feature_dim', 768),
        tactile_feature_dim=policy_config.get('tactile_feature_dim', 768),
        geometry_feature_dim=policy_config.get('geometry_feature_dim', 3),
        lstm_hidden_dim=policy_config.get('lstm_hidden_dim', 512),
        lstm_num_layers=policy_config.get('lstm_num_layers', 2),
        action_dim=policy_config.get('action_dim', 6),
        mlp_hidden_dims=policy_config.get('mlp_hidden_dims', [256, 128])
    )
    
    if os.path.exists(policy_model_path):
        checkpoint = torch.load(policy_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            policy_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            policy_model.load_state_dict(checkpoint)
        print(f"Loaded policy model from: {policy_model_path}")
    else:
        raise FileNotFoundError(f"Policy model not found: {policy_model_path}")
    
    # 设置为评估模式
    representation_model.eval()
    policy_model.eval()
    
    return representation_model.to(device), policy_model.to(device)


def create_robot_interface(config: dict) -> RobotInterface:
    """创建机器人接口"""
    robot_config = config.get('robot_params', {})
    
    # 创建机器人接口
    robot_interface = RobotInterface(robot_config)
    
    # 添加传感器
    vision_config = config.get('vision_sensor_params', {})
    tactile_config = config.get('tactile_sensor_params', {})
    
    # 视觉传感器（双目摄像头）
    left_camera = VisionSensor(
        'left_camera', 
        camera_index=vision_config.get('left_camera_index', 0),
        frequency=vision_config.get('frequency', 30.0)
    )
    
    right_camera = VisionSensor(
        'right_camera',
        camera_index=vision_config.get('right_camera_index', 1), 
        frequency=vision_config.get('frequency', 30.0)
    )
    
    # 触觉传感器
    tactile_sensor = TactileSensor(
        'tactile_array',
        num_sensors=tactile_config.get('num_sensors', 18),
        frequency=tactile_config.get('frequency', 1000.0)
    )
    
    robot_interface.add_sensor(left_camera)
    robot_interface.add_sensor(right_camera)
    robot_interface.add_sensor(tactile_sensor)
    
    return robot_interface


def process_sensor_data(
    sensor_data: Dict[str, SensorData],
    representation_model: HybridRepresentationModel,
    device: torch.device,
    config: dict
) -> Optional[torch.Tensor]:
    """
    处理传感器数据并提取特征
    
    Args:
        sensor_data: 同步的传感器数据
        representation_model: 表征模型
        device: 计算设备
        config: 配置
    
    Returns:
        状态向量张量
    """
    try:
        # 获取视觉数据
        left_camera_data = sensor_data.get('left_camera')
        right_camera_data = sensor_data.get('right_camera')
        tactile_data = sensor_data.get('tactile_array')
        
        if not all([left_camera_data, right_camera_data, tactile_data]):
            return None
        
        # 处理双目视觉数据
        left_image = left_camera_data.data
        right_image = right_camera_data.data
        
        # 水平拼接双目图像
        if left_image.shape == right_image.shape:
            stereo_image = np.concatenate([left_image, right_image], axis=1)
        else:
            print("Warning: Left and right images have different shapes")
            stereo_image = left_image
        
        # 转换为张量并预处理
        vision_tensor = torch.from_numpy(stereo_image).float()
        if vision_tensor.dim() == 3:
            vision_tensor = vision_tensor.permute(2, 0, 1)  # HWC -> CHW
        vision_tensor = vision_tensor.unsqueeze(0).to(device)  # 添加batch维度
        
        # 处理触觉数据
        tactile_array = tactile_data.data
        if tactile_array.ndim == 1:
            # 如果是1D，重塑为序列格式
            tactile_sequence = tactile_array.reshape(1, -1)
            # 复制到100个时间步
            tactile_sequence = np.repeat(tactile_sequence, 100, axis=0)
        else:
            tactile_sequence = tactile_array
        
        # 确保序列长度为100
        if tactile_sequence.shape[0] > 100:
            tactile_sequence = tactile_sequence[:100]
        elif tactile_sequence.shape[0] < 100:
            # 填充
            padding = np.tile(tactile_sequence[-1:], (100 - tactile_sequence.shape[0], 1))
            tactile_sequence = np.vstack([tactile_sequence, padding])
        
        tactile_tensor = torch.from_numpy(tactile_sequence).float().unsqueeze(0).to(device)
        
        # 通过表征模型提取特征
        with torch.no_grad():
            vision_features, _ = representation_model.encode_vision(vision_tensor)
            tactile_features, _ = representation_model.encode_tactile(tactile_tensor)
        
        # 计算几何特征（3D坐标）
        camera_params = config.get('camera_params', {
            'baseline': 0.12,
            'focal_length': 800,
            'cx': 320,
            'cy': 240
        })
        
        try:
            x, y, z = calculate_target_3d_coordinates(
                left_image, right_image, camera_params
            )
            geometry_features = torch.tensor([[x, y, z]], dtype=torch.float32, device=device)
        except Exception as e:
            print(f"Warning: Failed to calculate 3D coordinates: {e}")
            geometry_features = torch.zeros(1, 3, device=device)
        
        # 拼接所有特征
        state_vector = torch.cat([
            vision_features,
            tactile_features, 
            geometry_features
        ], dim=1)
        
        return state_vector
        
    except Exception as e:
        print(f"Error processing sensor data: {e}")
        return None


def execute_task(
    robot_interface: RobotInterface,
    representation_model: HybridRepresentationModel,
    policy_model: PolicyModel,
    config: dict,
    device: torch.device,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    执行完整的机器人任务
    
    Args:
        robot_interface: 机器人接口
        representation_model: 表征模型
        policy_model: 策略模型
        config: 配置
        device: 计算设备
        logger: 日志记录器
    
    Returns:
        任务执行结果
    """
    task_config = config.get('task_params', {})
    max_steps = task_config.get('max_steps', 1000)
    control_frequency = task_config.get('control_frequency', 10)  # Hz
    
    # 任务统计
    stats = {
        'total_steps': 0,
        'successful_steps': 0,
        'failed_steps': 0,
        'average_action_norm': 0.0,
        'execution_time': 0.0,
        'trajectory': []
    }
    
    # 初始化LSTM隐藏状态
    hidden_state = policy_model.init_hidden_state(1, device)
    
    logger.info("Starting task execution...")
    start_time = time.time()
    
    try:
        for step in range(max_steps):
            step_start_time = time.time()
            
            # 获取同步的传感器数据
            sensor_data = robot_interface.get_synchronized_sensor_data()
            
            if sensor_data is None:
                logger.warning(f"Step {step}: No synchronized sensor data available")
                stats['failed_steps'] += 1
                continue
            
            # 处理传感器数据
            state_vector = process_sensor_data(
                sensor_data, representation_model, device, config
            )
            
            if state_vector is None:
                logger.warning(f"Step {step}: Failed to process sensor data")
                stats['failed_steps'] += 1
                continue
            
            # 策略预测
            with torch.no_grad():
                predicted_action, hidden_state = policy_model.predict_step(
                    state_vector, hidden_state
                )
            
            # 应用动作约束
            action_constraints = {
                'velocity_limit': config.get('robot_params', {}).get('velocity_limit', 0.1),
                'force_limit': config.get('robot_params', {}).get('force_limit', 50.0)
            }
            
            constrained_action = policy_model.apply_action_constraints(
                predicted_action, action_constraints
            )
            
            # 创建机器人命令
            robot_command = RobotCommand(
                timestamp=time.time(),
                command_type='velocity',
                target_values=constrained_action.cpu().numpy().flatten(),
                duration=1.0 / control_frequency
            )
            
            # 执行命令
            success = robot_interface.execute_command(robot_command)
            
            if success:
                stats['successful_steps'] += 1
            else:
                stats['failed_steps'] += 1
                logger.warning(f"Step {step}: Command execution failed")
            
            # 记录轨迹
            action_norm = float(torch.norm(constrained_action).item())
            stats['trajectory'].append({
                'step': step,
                'timestamp': time.time(),
                'state_norm': float(torch.norm(state_vector).item()),
                'action': constrained_action.cpu().numpy().tolist(),
                'action_norm': action_norm,
                'success': success
            })
            
            stats['average_action_norm'] += action_norm
            stats['total_steps'] += 1
            
            # 检查任务完成条件
            if check_task_completion(sensor_data, config):
                logger.info(f"Task completed successfully at step {step}")
                break
            
            # 检查紧急停止条件
            if check_emergency_conditions(sensor_data, config):
                logger.warning(f"Emergency condition detected at step {step}")
                robot_interface.emergency_stop()
                break
            
            # 控制频率
            step_duration = time.time() - step_start_time
            sleep_time = max(0, (1.0 / control_frequency) - step_duration)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # 定期打印进度
            if step % 50 == 0:
                logger.info(f"Step {step}/{max_steps}, Success rate: {stats['successful_steps']/(step+1)*100:.1f}%")
    
    except KeyboardInterrupt:
        logger.info("Task interrupted by user")
        robot_interface.emergency_stop()
    
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        robot_interface.emergency_stop()
    
    finally:
        # 停止机器人
        stop_command = RobotCommand(
            timestamp=time.time(),
            command_type='stop',
            target_values=np.zeros(6)
        )
        robot_interface.execute_command(stop_command)
    
    # 计算最终统计
    stats['execution_time'] = time.time() - start_time
    if stats['total_steps'] > 0:
        stats['success_rate'] = stats['successful_steps'] / stats['total_steps']
        stats['average_action_norm'] /= stats['total_steps']
    else:
        stats['success_rate'] = 0.0
    
    logger.info(f"Task execution completed:")
    logger.info(f"  Total steps: {stats['total_steps']}")
    logger.info(f"  Success rate: {stats['success_rate']*100:.1f}%")
    logger.info(f"  Execution time: {stats['execution_time']:.1f}s")
    logger.info(f"  Average action norm: {stats['average_action_norm']:.4f}")
    
    return stats


def check_task_completion(sensor_data: Dict[str, SensorData], config: dict) -> bool:
    """
    检查任务是否完成
    
    Args:
        sensor_data: 传感器数据
        config: 配置
    
    Returns:
        是否完成任务
    """
    # 这里需要根据具体任务实现完成条件检查
    # 例如：检测到目标物体已被成功抓取并移动到指定位置
    
    task_type = config.get('task_params', {}).get('task_type', 'screw_tightening')
    
    if task_type == 'screw_tightening':
        # 检查螺丝是否拧紧（通过力传感器或视觉检测）
        tactile_data = sensor_data.get('tactile_array')
        if tactile_data:
            # 简化的完成检测：如果触觉信号稳定且达到一定阈值
            force_magnitude = np.linalg.norm(tactile_data.data)
            if force_magnitude > config.get('task_params', {}).get('completion_force_threshold', 5.0):
                return True
    
    return False


def check_emergency_conditions(sensor_data: Dict[str, SensorData], config: dict) -> bool:
    """
    检查紧急停止条件
    
    Args:
        sensor_data: 传感器数据
        config: 配置
    
    Returns:
        是否需要紧急停止
    """
    # 检查力传感器是否超过安全阈值
    tactile_data = sensor_data.get('tactile_array')
    if tactile_data:
        max_force = np.max(np.abs(tactile_data.data))
        force_limit = config.get('robot_params', {}).get('force_limit', 50.0)
        if max_force > force_limit:
            return True
    
    # 其他安全检查...
    
    return False


def save_results(stats: Dict[str, Any], config: dict, output_dir: str) -> None:
    """保存执行结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存统计信息
    stats_file = os.path.join(output_dir, f"task_stats_{timestamp}.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # 保存轨迹数据
    trajectory_file = os.path.join(output_dir, f"trajectory_{timestamp}.json")
    with open(trajectory_file, 'w') as f:
        json.dump(stats['trajectory'], f, indent=2, default=str)
    
    print(f"Results saved to {output_dir}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Robot Demo for Project Poseidon')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='demo_results',
                       help='Output directory for results')
    parser.add_argument('--task', type=str, default='screw_tightening',
                       choices=['screw_tightening', 'object_manipulation', 'assembly'],
                       help='Task type to execute')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置任务类型
    if 'task_params' not in config:
        config['task_params'] = {}
    config['task_params']['task_type'] = args.task
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 设置日志
    logger = setup_logging(args.output_dir)
    logger.info(f"Starting robot demo with task: {args.task}")
    
    try:
        # 加载模型
        logger.info("Loading models...")
        representation_model, policy_model = load_models(config, device)
        
        # 创建机器人接口
        logger.info("Creating robot interface...")
        robot_interface = create_robot_interface(config)
        
        # 连接机器人
        logger.info("Connecting to robot...")
        if not robot_interface.connect():
            logger.error("Failed to connect to robot")
            return
        
        # 开始数据记录
        robot_interface.start_data_recording()
        
        # 执行任务
        logger.info(f"Executing {args.task} task...")
        stats = execute_task(
            robot_interface, representation_model, policy_model,
            config, device, logger
        )
        
        # 停止数据记录
        robot_interface.stop_data_recording()
        
        # 保存机器人记录的数据
        robot_data_file = os.path.join(args.output_dir, 'robot_data.json')
        robot_interface.save_recorded_data(robot_data_file)
        
        # 保存结果
        save_results(stats, config, args.output_dir)
        
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    
    finally:
        # 确保断开机器人连接
        if 'robot_interface' in locals():
            robot_interface.disconnect()


if __name__ == '__main__':
    main()
