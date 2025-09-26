#!/usr/bin/env python3
"""
Stage 2: Dynamic Policy Learning with DAgger
阶段2：基于DAgger的动态策略学习主训练脚本
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import logging
from datetime import datetime
import json
from typing import Dict, List, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.policy_model import PolicyModel, DAggerTrainer
from models.representation_model import RepresentationModel
from models.classifier import ObjectClassifier
from data_loader.dataset import PolicyDataset
from data_loader.samplers import DAggerSampler
from engine.trainer import train_policy_epoch
from engine.evaluator import evaluate_policy_epoch, get_action_with_uncertainty, should_request_expert_annotation
from engine.losses import PolicyLoss
from robot.interface import RobotInterface, VisionSensor, TactileSensor
from robot.expert import ExpertFactory


def setup_logging(log_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"stage2_policy_{timestamp}.log")
    
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


def set_seed(seed: int) -> None:
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_representation_model(config: dict, device: torch.device) -> RepresentationModel:
    """创建并加载表征模型"""
    model_config = config['model_params']
    rep_config = model_config['representation_model']
    
    # 创建表征模型（纯CLIP变体）
    rep_model = RepresentationModel(
        vision_encoder_weights_path=None,  # 稍后从检查点加载
        embed_dim=128  # 使用默认值，稍后从检查点加载
    )
    
    # 加载预训练权重
    weights_path = rep_config['weights_path']
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            rep_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            rep_model.load_state_dict(checkpoint)
        print(f"Loaded representation model from: {weights_path}")
    else:
        print(f"Warning: Representation model weights not found at {weights_path}")
    
    # 冻结表征模型
    if rep_config.get('freeze', True):
        rep_model.freeze_encoders()
    
    return rep_model.to(device)


def create_classifier_model(config: dict, device: torch.device) -> ObjectClassifier:
    """创建并加载分类器模型"""
    model_config = config['model_params']
    classifier_config = model_config.get('classifier_model', {})
    
    # 创建分类器
    classifier = ObjectClassifier(
        feature_dim=classifier_config.get('feature_dim', 1536),  # 768 + 768
        hidden_dim=classifier_config.get('hidden_dim', 512),
        num_classes=classifier_config.get('num_classes', 15),
        dropout=classifier_config.get('dropout', 0.2)
    )
    
    # 加载预训练权重
    weights_path = classifier_config.get('weights_path')
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        if 'classifier_state_dict' in checkpoint:
            classifier.load_state_dict(checkpoint['classifier_state_dict'])
        else:
            classifier.load_state_dict(checkpoint)
        print(f"Loaded classifier from: {weights_path}")
    else:
        print(f"Warning: Classifier weights not found at {weights_path}")
    
    # 冻结分类器
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    
    return classifier.to(device)


def create_policy_model(config: dict, device: torch.device) -> PolicyModel:
    """创建策略模型"""
    model_config = config['model_params']['policy_model']
    classifier_config = config['model_params'].get('classifier_model', {})
    
    policy_model = PolicyModel(
        vision_feature_dim=model_config.get('vision_feature_dim', 768),
        tactile_feature_dim=model_config.get('tactile_feature_dim', 768),
        geometry_feature_dim=model_config.get('geometry_feature_dim', 3),
        classification_feature_dim=classifier_config.get('num_classes', 15),  # 新增分类特征维度
        lstm_hidden_dim=model_config['lstm_hidden_dim'],
        lstm_num_layers=model_config['lstm_num_layers'],
        lstm_dropout=model_config['lstm_dropout'],
        action_dim=model_config['action_dim'],
        mlp_hidden_dims=model_config.get('mlp_hidden_dims', [256, 128]),
        mlp_dropout=model_config['mlp_dropout']
    )
    
    policy_model = policy_model.to(device)
    policy_model.print_model_info()
    
    return policy_model


def create_datasets(config: dict) -> tuple:
    """创建数据集"""
    data_config = config['data_params']
    
    # 创建初始专家数据集
    train_dataset = PolicyDataset(
        data_path=data_config['dataset_path'],
        split='train',
        trajectory_length=data_config['trajectory_length'],
        tactile_seq_len=100,  # 与表征模型保持一致
        include_geometry=True
    )
    
    val_dataset = PolicyDataset(
        data_path=data_config['dataset_path'],
        split='val',
        trajectory_length=data_config['trajectory_length'],
        tactile_seq_len=100,
        include_geometry=True
    )
    
    return train_dataset, val_dataset


def create_data_loaders(train_dataset, val_dataset, config: dict, expert_indices: List[int] = None, policy_indices: List[int] = None) -> tuple:
    """创建数据加载器"""
    data_config = config['data_params']
    dagger_config = config.get('dagger_params', {})
    
    # 如果提供了DAgger采样索引，使用DAgger采样器
    if expert_indices is not None and policy_indices is not None:
        train_sampler = DAggerSampler(
            expert_indices=expert_indices,
            policy_indices=policy_indices,
            batch_size=data_config['batch_size'],
            expert_ratio=dagger_config.get('expert_data_ratio', 0.5),
            shuffle=True,
            drop_last=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=data_config['num_workers'],
            pin_memory=data_config.get('pin_memory', True)
        )
    else:
        # 常规数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            pin_memory=data_config.get('pin_memory', True),
            drop_last=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config.get('pin_memory', True),
        drop_last=False
    )
    
    return train_loader, val_loader


def create_optimizer_and_scheduler(model: nn.Module, config: dict) -> tuple:
    """创建优化器和学习率调度器"""
    train_config = config['training_params']
    scheduler_config = config.get('scheduler_params', {})
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    
    # 创建学习率调度器
    scheduler = None
    if scheduler_config.get('type') == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 20),
            gamma=scheduler_config.get('gamma', 0.8)
        )
    
    return optimizer, scheduler


def create_loss_function(config: dict) -> nn.Module:
    """创建损失函数"""
    loss_config = config['loss_params']
    
    loss_fn = PolicyLoss(
        action_loss_type=loss_config.get('type', 'mse'),
        action_weight=loss_config.get('action_weight', 1.0),
        regularization_weight=loss_config.get('regularization_weight', 0.01),
        velocity_weight=0.1,
        smoothness_weight=0.05
    )
    
    return loss_fn


def create_robot_interface(config: dict) -> RobotInterface:
    """创建机器人接口"""
    robot_config = config.get('robot_params', {})
    
    # 创建机器人接口
    robot_interface = RobotInterface(robot_config)
    
    # 添加传感器
    vision_sensor = VisionSensor('stereo_camera', frequency=30.0)
    tactile_sensor = TactileSensor('tactile_array', num_sensors=18, frequency=1000.0)
    
    robot_interface.add_sensor(vision_sensor)
    robot_interface.add_sensor(tactile_sensor)
    
    return robot_interface


def expert_policy_function(state: np.ndarray) -> np.ndarray:
    """
    专家策略函数（占位符实现）
    在实际应用中，这应该是人类专家或高级控制器提供的策略
    """
    # 这是一个简化的专家策略示例
    # 实际应用中需要根据具体任务实现
    
    # 提取状态信息
    # state包含: [vision_features(768), tactile_features(768), geometry_features(3)]
    
    # 简单的比例控制示例
    if len(state) >= 3:
        # 假设最后3个元素是3D坐标
        target_pos = state[-3:]
        # 简单的位置控制
        action = np.clip(target_pos * 0.1, -0.1, 0.1)
        # 补充到6DOF
        if len(action) < 6:
            action = np.pad(action, (0, 6 - len(action)), 'constant')
    else:
        action = np.zeros(6)
    
    return action


def collect_policy_rollouts(
    policy_model: PolicyModel,
    robot_interface: RobotInterface,
    representation_model: RepresentationModel,
    classifier: ObjectClassifier,
    expert: Any,
    config: dict,
    num_episodes: int = 10
) -> List[Dict[str, Any]]:
    """
    收集策略执行的轨迹数据（集成主动学习）
    
    Args:
        policy_model: 策略模型
        robot_interface: 机器人接口
        representation_model: 表征模型
        classifier: 分类器模型
        expert: 专家接口
        config: 配置
        num_episodes: episode数量
    
    Returns:
        轨迹数据列表
    """
    policy_model.eval()
    representation_model.eval()
    
    rollouts = []
    device = next(policy_model.parameters()).device
    
    dagger_config = config.get('dagger_params', {})
    max_episode_length = dagger_config.get('max_episode_length', 100)
    
    # --- 新增/修改 Start ---
    # 获取主动学习配置
    active_learning_config = config.get('active_learning_params', {})
    active_learning_enabled = active_learning_config.get('enabled', False)
    mc_samples = active_learning_config.get('mc_dropout_samples', 25)
    arm_threshold = active_learning_config.get('arm_uncertainty_threshold', 0.1)
    gripper_threshold = active_learning_config.get('gripper_uncertainty_threshold', 0.05)
    
    # 统计信息
    total_steps = 0
    expert_requests = 0
    # --- 新增/修改 End ---
    
    with torch.no_grad():
        for episode in range(num_episodes):
            print(f"Collecting episode {episode + 1}/{num_episodes}")
            
            # 初始化episode
            states = []
            actions = []
            expert_actions = []  # 存储专家动作
            uncertainty_scores = []  # 存储不确定性分数
            
            # 初始化LSTM隐藏状态
            hidden_state = policy_model.init_hidden_state(1, device)
            
            for step in range(max_episode_length):
                # 获取同步的传感器数据
                sensor_data = robot_interface.get_synchronized_sensor_data()
                
                if sensor_data is None:
                    continue
                
                # 提取视觉和触觉数据
                vision_data = sensor_data.get('stereo_camera')
                tactile_data = sensor_data.get('tactile_array')
                
                if vision_data is None or tactile_data is None:
                    continue
                
                # 转换数据格式
                vision_tensor = torch.from_numpy(vision_data.data).float().unsqueeze(0).to(device)
                tactile_tensor = torch.from_numpy(tactile_data.data).float().unsqueeze(0).unsqueeze(0).to(device)
                
                # 通过表征模型提取特征
                vision_features, tactile_features = representation_model(vision_tensor, tactile_tensor)
                
                # 通过分类器获取分类特征
                with torch.no_grad():
                    combined_features = torch.cat([vision_features, tactile_features], dim=1)
                    classification_logits = classifier(combined_features)
                
                # 计算几何特征（3D坐标）
                # 这里需要实现具体的几何计算逻辑
                geometry_features = torch.zeros(1, 3).to(device)  # 占位符
                
                # 构建状态向量（包含分类特征）
                state_vector = torch.cat([
                    vision_features,
                    tactile_features,
                    geometry_features,
                    classification_logits
                ], dim=1)
                
                states.append(state_vector.cpu().numpy())
                
                # --- 新增/修改 Start ---
                # 主动学习：使用MC Dropout计算不确定性和动作
                if active_learning_enabled:
                    # 使用MC Dropout获取动作和不确定性
                    robot_action, arm_uncertainty, gripper_uncertainty = get_action_with_uncertainty(
                        policy_model,
                        state_vector,
                        hidden_state,
                        mc_samples
                    )
                    
                    # 判断是否需要专家标注
                    need_expert = should_request_expert_annotation(
                        arm_uncertainty, gripper_uncertainty, arm_threshold, gripper_threshold
                    )
                    
                    if need_expert:
                        # 请求专家标注
                        print(f"🤖 高不确定性! Arm: {arm_uncertainty:.4f}, Gripper: {gripper_uncertainty:.4f}. 请求专家标注...")
                        
                        # 构建当前状态字典供专家使用
                        current_state = {
                            'position': geometry_features.cpu().numpy().flatten().tolist(),
                            'vision_features': vision_features.cpu().numpy().flatten().tolist(),
                            'tactile_features': tactile_features.cpu().numpy().flatten().tolist(),
                            'classification_logits': classification_logits.cpu().numpy().flatten().tolist()
                        }
                        
                        expert_action = expert.get_label(current_state)
                        expert_actions.append(expert_action.cpu().numpy())
                        expert_requests += 1
                        
                        # 使用专家动作进行训练，但执行机器人动作
                        final_action = robot_action
                    else:
                        # 不确定性较低，机器人自主决策
                        expert_actions.append(None)  # 标记为无专家标注
                        final_action = robot_action
                    
                    # 记录不确定性分数
                    uncertainty_scores.append({
                        'arm_uncertainty': arm_uncertainty,
                        'gripper_uncertainty': gripper_uncertainty,
                        'total_uncertainty': arm_uncertainty + gripper_uncertainty
                    })
                    
                    # 更新隐藏状态（使用机器人动作）
                    _, hidden_state = policy_model.predict_step(state_vector, hidden_state)
                    
                else:
                    # 传统DAgger：每次都请求专家标注
                    predicted_action, hidden_state = policy_model.predict_step(
                        state_vector, hidden_state
                    )
                    
                    # 构建当前状态字典供专家使用
                    current_state = {
                        'position': geometry_features.cpu().numpy().flatten().tolist(),
                        'vision_features': vision_features.cpu().numpy().flatten().tolist(),
                        'tactile_features': tactile_features.cpu().numpy().flatten().tolist(),
                        'classification_logits': classification_logits.cpu().numpy().flatten().tolist()
                    }
                    
                    expert_action = expert.get_label(current_state)
                    expert_actions.append(expert_action.cpu().numpy())
                    expert_requests += 1
                    
                    final_action = predicted_action
                    uncertainty_scores.append(None)  # 传统模式下无不确定性分数
                # --- 新增/修改 End ---
                
                # 应用动作约束
                action_constraints = {
                    'velocity_limit': config.get('robot_params', {}).get('velocity_limit', 0.1)
                }
                constrained_action = policy_model.apply_action_constraints(
                    final_action, action_constraints
                )
                
                actions.append(constrained_action.cpu().numpy())
                total_steps += 1
                
                # 执行动作（在实际机器人上）
                # 这里需要将动作发送给机器人
                
                # 检查任务完成条件
                # 这里需要根据具体任务实现
                
            rollouts.append({
                'episode_id': episode,
                'states': states,
                'actions': actions,
                'expert_actions': expert_actions,  # 新增：专家动作
                'uncertainty_scores': uncertainty_scores,  # 新增：不确定性分数
                'length': len(states)
            })
    
    # --- 新增/修改 Start ---
    # 打印主动学习统计信息
    if active_learning_enabled:
        expert_request_rate = expert_requests / max(total_steps, 1) * 100
        print(f"\n📊 主动学习统计:")
        print(f"   总步数: {total_steps}")
        print(f"   专家请求次数: {expert_requests}")
        print(f"   专家请求率: {expert_request_rate:.2f}%")
        print(f"   节省标注: {total_steps - expert_requests} 步")
    # --- 新增/修改 End ---
    
    return rollouts


def run_dagger_iteration(
    policy_model: PolicyModel,
    representation_model: RepresentationModel,
    classifier: ObjectClassifier,
    expert: Any,
    robot_interface: RobotInterface,
    config: dict,
    iteration: int,
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    运行一次DAgger迭代（集成主动学习）
    
    Args:
        policy_model: 策略模型
        representation_model: 表征模型
        classifier: 分类器模型
        expert: 专家接口
        robot_interface: 机器人接口
        config: 配置
        iteration: 迭代次数
        logger: 日志记录器
    
    Returns:
        新收集的训练数据
    """
    dagger_config = config.get('dagger_params', {})
    episodes_per_iteration = dagger_config.get('episodes_per_iteration', 50)
    
    logger.info(f"Starting DAgger iteration {iteration}")
    
    # 1. 使用当前策略收集轨迹（集成主动学习）
    logger.info("Collecting policy rollouts...")
    rollouts = collect_policy_rollouts(
        policy_model, robot_interface, representation_model, classifier, expert,
        config, episodes_per_iteration
    )
    
    # 2. 专家标注（在实际应用中，这里需要人类专家介入）
    logger.info("Collecting expert corrections...")
    corrected_data = []
    
    for rollout in rollouts:
        states = rollout['states']
        actions = rollout['actions']
        
        # 为每个状态获取专家动作
        expert_actions = []
        for state in states:
            expert_action = expert_policy_function(state.flatten())
            expert_actions.append(expert_action)
        
        corrected_data.append({
            'states': states,
            'actions': expert_actions,
            'episode_id': rollout['episode_id'],
            'iteration': iteration
        })
    
    logger.info(f"Collected {len(corrected_data)} corrected trajectories")
    
    return corrected_data


def save_checkpoint(
    policy_model: PolicyModel,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    iteration: int,
    best_success_rate: float,
    checkpoint_dir: str,
    is_best: bool = False
) -> None:
    """保存检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': policy_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_success_rate': best_success_rate,
    }
    
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # 保存常规检查点
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration}_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # 保存最佳模型
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_policy_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Best policy model saved with success rate: {best_success_rate:.4f}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Stage 2: Dynamic Policy Learning with DAgger')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--skip_robot', action='store_true',
                       help='Skip robot interface (for testing)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and config['device_params']['use_cuda'] else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    checkpoint_dir = config['checkpoint_params']['output_dir']
    log_dir = config['logging_params']['tensorboard_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(log_dir)
    logger.info(f"Starting Stage 2 training with config: {args.config}")
    
    # 创建表征模型（冻结）
    logger.info("Loading representation model...")
    representation_model = create_representation_model(config, device)
    representation_model.eval()
    
    # 创建分类器模型（冻结）
    logger.info("Loading classifier model...")
    classifier = create_classifier_model(config, device)
    
    # --- 新增/修改 Start ---
    # 创建专家接口
    logger.info("Creating expert interface...")
    expert = ExpertFactory.create_expert(config)
    # --- 新增/修改 End ---
    
    # 创建策略模型
    logger.info("Creating policy model...")
    policy_model = create_policy_model(config, device)
    
    # 创建优化器和调度器
    optimizer, scheduler = create_optimizer_and_scheduler(policy_model, config)
    
    # 创建损失函数
    loss_fn = create_loss_function(config)
    
    # 创建机器人接口（如果不跳过）
    robot_interface = None
    if not args.skip_robot:
        logger.info("Creating robot interface...")
        robot_interface = create_robot_interface(config)
        robot_interface.connect()
    
    # 创建初始数据集
    logger.info("Creating initial datasets...")
    train_dataset, val_dataset = create_datasets(config)
    
    logger.info(f"Initial train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(log_dir)
    
    # DAgger参数
    dagger_config = config.get('dagger_params', {})
    max_iterations = dagger_config.get('max_iterations', 10)
    initial_epochs = dagger_config.get('initial_epochs', 50)
    iteration_epochs = dagger_config.get('iteration_epochs', 30)
    
    # 恢复训练（如果指定）
    start_iteration = 0
    start_epoch = 1
    best_success_rate = 0.0
    aggregated_data = []
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        policy_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_iteration = checkpoint.get('iteration', 0)
        start_epoch = checkpoint['epoch'] + 1
        best_success_rate = checkpoint.get('best_success_rate', 0.0)
    
    # DAgger迭代循环
    for iteration in range(start_iteration, max_iterations):
        logger.info(f"\n=== DAgger Iteration {iteration + 1}/{max_iterations} ===")
        
        # 确定当前迭代的训练epoch数
        current_epochs = initial_epochs if iteration == 0 else iteration_epochs
        
        # 如果不是第一次迭代且有机器人接口，收集新数据
        if iteration > 0 and robot_interface is not None:
            new_data = run_dagger_iteration(
                policy_model, representation_model, classifier, expert, robot_interface,
                config, iteration, logger
            )
            aggregated_data.extend(new_data)
            
            # 更新数据集（这里需要实现数据聚合逻辑）
            # 在实际实现中，需要将新数据保存并重新创建数据集
            logger.info(f"Aggregated {len(aggregated_data)} total correction samples")
        
        # 创建数据加载器
        if iteration > 0 and aggregated_data:
            # 使用DAgger采样器
            expert_indices = list(range(len(train_dataset)))
            policy_indices = list(range(len(train_dataset), len(train_dataset) + len(aggregated_data)))
            train_loader, val_loader = create_data_loaders(
                train_dataset, val_dataset, config, expert_indices, policy_indices
            )
        else:
            # 使用初始数据
            train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, config)
        
        # 训练当前迭代
        logger.info(f"Training for {current_epochs} epochs...")
        
        for epoch in range(1, current_epochs + 1):
            global_epoch = iteration * current_epochs + epoch
            
            # 训练阶段
            train_results = train_policy_epoch(
                model=policy_model,
                data_loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                epoch=global_epoch,
                lr_scheduler=scheduler,
                grad_clip_norm=config['training_params'].get('grad_clip_norm', 1.0),
                logger=logger
            )
            
            # 验证阶段（每5个epoch一次）
            if epoch % 5 == 0:
                val_results = evaluate_policy_epoch(
                    model=policy_model,
                    data_loader=val_loader,
                    loss_fn=loss_fn,
                    device=device,
                    epoch=global_epoch,
                    metrics=['loss', 'mse', 'action_smoothness'],
                    logger=logger
                )
                
                # 记录到TensorBoard
                writer.add_scalar('Val/Loss', val_results['average_loss'], global_epoch)
                if 'mse' in val_results:
                    writer.add_scalar('Val/MSE', val_results['mse'], global_epoch)
                
                current_success_rate = 1.0 / (1.0 + val_results['average_loss'])  # 简化的成功率计算
            else:
                current_success_rate = 0.0
            
            # 记录训练指标
            writer.add_scalar('Train/Loss', train_results['average_loss'], global_epoch)
            writer.add_scalar('Train/LearningRate', train_results['learning_rate'], global_epoch)
            writer.add_scalar('DAgger/Iteration', iteration + 1, global_epoch)
            
            # 保存检查点
            is_best = current_success_rate > best_success_rate
            if is_best:
                best_success_rate = current_success_rate
            
            if epoch % 10 == 0 or is_best:
                save_checkpoint(
                    policy_model=policy_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    iteration=iteration,
                    best_success_rate=best_success_rate,
                    checkpoint_dir=checkpoint_dir,
                    is_best=is_best
                )
        
        logger.info(f"Iteration {iteration + 1} completed. Best success rate: {best_success_rate:.4f}")
    
    # 训练完成
    logger.info(f"DAgger training completed! Best success rate: {best_success_rate:.4f}")
    
    # 保存最终模型
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_policy_model.pth')
    torch.save({
        'iteration': max_iterations,
        'model_state_dict': policy_model.state_dict(),
        'best_success_rate': best_success_rate,
        'config': config
    }, final_checkpoint_path)
    
    logger.info(f"Final model saved to: {final_checkpoint_path}")
    
    # 断开机器人连接
    if robot_interface:
        robot_interface.disconnect()
    
    # 关闭TensorBoard写入器
    writer.close()


if __name__ == '__main__':
    main()
