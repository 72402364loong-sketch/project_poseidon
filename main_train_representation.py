#!/usr/bin/env python3
"""
Stage 1: Multimodal Representation Learning
阶段1：多模态表征学习的主训练脚本
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

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.representation_model import RepresentationModel
from data_loader.dataset import RepresentationDataset
from data_loader.augmentations import get_vision_transforms, get_tactile_transforms
from data_loader.samplers import BalancedBatchSampler
from engine.trainer import train_representation_epoch
from engine.evaluator import evaluate_representation_epoch
from engine.losses import InfoNCELoss


def setup_logging(log_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"stage1_representation_{timestamp}.log")
    
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


def create_datasets(config: dict) -> tuple:
    """创建数据集"""
    data_config = config['data_params']
    aug_config = config.get('augmentation_params', {})
    
    # 视觉变换
    train_vision_transforms = get_vision_transforms(
        config=aug_config.get('vision', {}),
        is_training=True,
        underwater_augmentation=False  # 在表征学习阶段不使用水下增强
    )
    
    val_vision_transforms = get_vision_transforms(
        config=aug_config.get('vision', {}),
        is_training=False,
        underwater_augmentation=False
    )
    
    # 触觉变换
    train_tactile_transforms = get_tactile_transforms(
        config=aug_config,
        is_training=True
    )
    
    val_tactile_transforms = get_tactile_transforms(
        config=aug_config,
        is_training=False
    )
    
    # 创建数据集
    train_dataset = RepresentationDataset(
        data_path=data_config['dataset_path'],
        split='train',
        vision_transform=train_vision_transforms,
        tactile_transform=train_tactile_transforms,
        tactile_seq_len=config['model_params']['tactile_encoder']['seq_len'],
        stereo_mode=True
    )
    
    val_dataset = RepresentationDataset(
        data_path=data_config['dataset_path'],
        split='val',
        vision_transform=val_vision_transforms,
        tactile_transform=val_tactile_transforms,
        tactile_seq_len=config['model_params']['tactile_encoder']['seq_len'],
        stereo_mode=True
    )
    
    return train_dataset, val_dataset


def create_data_loaders(train_dataset, val_dataset, config: dict) -> tuple:
    """创建数据加载器"""
    data_config = config['data_params']
    sampler_config = config.get('sampler_params', {})
    
    # 训练集使用平衡采样器
    if sampler_config.get('type') == 'balanced_batch':
        train_sampler = BalancedBatchSampler(
            dataset=train_dataset,
            batch_size=data_config['batch_size'],
            samples_per_class=sampler_config.get('samples_per_class', 8),
            drop_last=True,
            shuffle=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=data_config['num_workers'],
            pin_memory=data_config.get('pin_memory', True)
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            pin_memory=data_config.get('pin_memory', True),
            drop_last=True
        )
    
    # 验证集使用常规采样
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config.get('pin_memory', True),
        drop_last=False
    )
    
    return train_loader, val_loader


def create_model(config: dict, device: torch.device) -> nn.Module:
    """创建模型"""
    model_config = config['model_params']
    
    # 创建表征模型（纯CLIP变体）
    model = RepresentationModel(
        # 视觉编码器参数
        vision_encoder_weights_path=model_config['vision_encoder'].get('pretrained_weights_path'),
        vision_model_name=model_config['vision_encoder']['model_name'],
        freeze_vision_encoder=model_config['vision_encoder'].get('freeze_encoder', False),
        
        # 触觉编码器参数
        tactile_feature_dim=model_config['tactile_encoder']['feature_dim'],
        tactile_seq_len=model_config['tactile_encoder']['seq_len'],
        tactile_d_model=model_config['tactile_encoder']['d_model'],
        tactile_nhead=model_config['tactile_encoder']['nhead'],
        tactile_num_layers=model_config['tactile_encoder']['num_layers'],
        tactile_dim_feedforward=model_config['tactile_encoder']['dim_feedforward'],
        tactile_dropout=model_config['tactile_encoder']['dropout'],
        
        # 投影头参数
        embed_dim=model_config['projection']['embed_dim'],
        projection_hidden_dim=model_config['projection']['projection_hidden_dim']
    )
    
    model = model.to(device)
    
    # 打印模型信息
    model.print_model_info()
    
    return model


def create_optimizer_and_scheduler(model: nn.Module, config: dict) -> tuple:
    """创建优化器和学习率调度器"""
    train_config = config['training_params']
    scheduler_config = config.get('scheduler_params', {})
    
    # 创建优化器
    if train_config['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
    elif train_config['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
            momentum=0.9
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
    
    # 创建学习率调度器
    scheduler = None
    if scheduler_config.get('type') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', 100),
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
    elif scheduler_config.get('type') == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_config.get('type') == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_config.get('milestones', [60, 80]),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    
    return optimizer, scheduler


def create_loss_function(config: dict, device: torch.device) -> nn.Module:
    """创建损失函数（纯CLIP变体）"""
    loss_config = config['loss_params']
    
    # 纯CLIP变体：只使用InfoNCE损失
    loss_fn = InfoNCELoss(
        temperature=loss_config.get('temperature', 0.07),
        reduction='mean'
    )
    
    return loss_fn.to(device)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    best_recall: float,
    checkpoint_dir: str,
    is_best: bool = False
) -> None:
    """保存检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_recall': best_recall,
    }
    
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # 保存常规检查点
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # 保存最佳模型
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model saved with recall@1: {best_recall:.4f}")
    
    # 分别保存编码器（用于后续阶段）
    if is_best:
        model.save_encoders_separately(checkpoint_dir)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Stage 1: Multimodal Representation Learning')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
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
    logger.info(f"Starting Stage 1 training with config: {args.config}")
    
    # 创建数据集和数据加载器
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_datasets(config)
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, config)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # 创建模型
    logger.info("Creating model...")
    model = create_model(config, device)
    
    # 创建优化器和调度器
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    # 创建损失函数
    loss_fn = create_loss_function(config, device)
    
    # 创建混合精度缩放器
    scaler = None
    if config['device_params'].get('mixed_precision', False):
        scaler = torch.cuda.amp.GradScaler()
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(log_dir)
    
    # 恢复训练（如果指定）
    start_epoch = 1
    best_recall = 0.0
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_recall = checkpoint.get('best_recall', 0.0)
    
    # 训练循环
    train_config = config['training_params']
    epochs = train_config['epochs']
    log_freq = config['logging_params']['log_freq']
    save_freq = config['checkpoint_params']['save_freq']
    eval_freq = config['evaluation_params'].get('eval_freq', 5)
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(start_epoch, epochs + 1):
        logger.info(f"\nEpoch {epoch}/{epochs}")
        
        # 训练阶段
        train_results = train_representation_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            lr_scheduler=scheduler,
            scaler=scaler,
            log_freq=log_freq,
            logger=logger
        )
        
        # 验证阶段（每eval_freq个epoch进行一次）
        if epoch % eval_freq == 0:
            val_results = evaluate_representation_epoch(
                model=model,
                data_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                epoch=epoch,
                metrics=['loss', 'retrieval_recall@1', 'retrieval_recall@5', 'accuracy'],
                logger=logger
            )
            
            # 记录到TensorBoard
            writer.add_scalar('Val/Loss', val_results['average_loss'], epoch)
            
            if 'retrieval_recall@1' in val_results:
                writer.add_scalar('Val/Recall@1', val_results['retrieval_recall@1'], epoch)
                current_recall = val_results['retrieval_recall@1']
            else:
                current_recall = 0.0
            
            if 'retrieval_recall@5' in val_results:
                writer.add_scalar('Val/Recall@5', val_results['retrieval_recall@5'], epoch)
            
            if 'accuracy' in val_results:
                writer.add_scalar('Val/Accuracy', val_results['accuracy'], epoch)
        else:
            val_results = {'average_loss': 0.0}
            current_recall = 0.0
        
        # 记录训练指标到TensorBoard
        writer.add_scalar('Train/Loss', train_results['average_loss'], epoch)
        writer.add_scalar('Train/LearningRate', train_results['learning_rate'], epoch)
        
        # 保存检查点
        is_best = current_recall > best_recall
        if is_best:
            best_recall = current_recall
        
        if epoch % save_freq == 0 or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_recall=best_recall,
                checkpoint_dir=checkpoint_dir,
                is_best=is_best
            )
        
        # 打印epoch总结
        if epoch % eval_freq == 0:
            logger.info(f"Epoch {epoch} - Train Loss: {train_results['average_loss']:.4f}, "
                       f"Val Loss: {val_results['average_loss']:.4f}, "
                       f"Val Recall@1: {current_recall:.4f}, Best Recall@1: {best_recall:.4f}")
        else:
            logger.info(f"Epoch {epoch} - Train Loss: {train_results['average_loss']:.4f}")
    
    # 训练完成
    logger.info(f"Training completed! Best Recall@1: {best_recall:.4f}")
    
    # 保存最终模型
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pth')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'best_recall': best_recall,
        'config': config
    }, final_checkpoint_path)
    
    logger.info(f"Final model saved to: {final_checkpoint_path}")
    
    # 分别保存编码器
    model.save_encoders_separately(checkpoint_dir)
    
    # 关闭TensorBoard写入器
    writer.close()


if __name__ == '__main__':
    main()
