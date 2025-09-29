#!/usr/bin/env python3
"""
Stage 0.5: Vision Domain Adaptation on URPC Dataset
阶段0.5：在URPC数据集上进行视觉领域适应的主训练脚本
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

from models.vision_encoder import ViTEncoder
from data_loader.dataset import URPCDataset
from data_loader.vision_adaptation_dataset import URPCVisionAdaptationDataset
from data_loader.augmentations import get_vision_transforms
from engine.trainer import train_one_epoch
from engine.evaluator import evaluate_one_epoch
from engine.losses import FocalLoss


def setup_logging(log_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"stage0_finetune_{timestamp}.log")
    
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
    
    # 检查是否使用视觉适应模式
    use_vision_adaptation = data_config.get('ignore_labels', False)
    use_all_splits = data_config.get('use_all_splits_for_training', False)
    
    # 训练集变换
    train_transforms = get_vision_transforms(
        config=aug_config,
        is_training=True,
        underwater_augmentation=aug_config.get('underwater_style', False)
    )
    
    # 验证集变换
    val_transforms = get_vision_transforms(
        config=aug_config,
        is_training=False,
        underwater_augmentation=False
    )
    
    if use_vision_adaptation:
        # 使用视觉适应数据集
        print("Using Vision Adaptation Dataset (ignoring labels)")
        train_dataset = URPCVisionAdaptationDataset(
            data_path=data_config['urpc_dataset_path'],
            split='all' if use_all_splits else 'train',
            transform=train_transforms,
            use_all_splits=use_all_splits
        )
        
        # 验证集使用一小部分数据
        val_dataset = URPCVisionAdaptationDataset(
            data_path=data_config['urpc_dataset_path'],
            split='val',
            transform=val_transforms,
            use_all_splits=False
        )
    else:
        # 使用原始URPC数据集（带标签）
        train_dataset = URPCDataset(
            data_path=data_config['urpc_dataset_path'],
            split='train',
            transform=train_transforms
        )
        
        val_dataset = URPCDataset(
            data_path=data_config['urpc_dataset_path'],
            split='val',
            transform=val_transforms
        )
    
    return train_dataset, val_dataset


def create_data_loaders(train_dataset, val_dataset, config: dict) -> tuple:
    """创建数据加载器"""
    data_config = config['data_params']
    
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


def create_model(config: dict, device: torch.device) -> nn.Module:
    """创建模型"""
    model_config = config['model_params']
    
    # 创建ViT编码器
    model = ViTEncoder(
        model_name=model_config['model_name'],
        pretrained=model_config['pretrained'],
        num_classes=model_config['num_classes'],
        freeze_layers=model_config.get('freeze_layers', 0)
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
            T_max=scheduler_config.get('T_max', 50),
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
    elif scheduler_config.get('type') == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 20),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_config.get('type') == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_config.get('milestones', [30, 45]),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    
    return optimizer, scheduler


def create_loss_function(config: dict, device: torch.device) -> nn.Module:
    """创建损失函数"""
    model_config = config['model_params']
    num_classes = model_config['num_classes']
    
    # 使用Focal Loss处理类别不平衡
    loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
    return loss_fn.to(device)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    best_acc: float,
    checkpoint_dir: str,
    is_best: bool = False
) -> None:
    """保存检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
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
        print(f"Best model saved with accuracy: {best_acc:.4f}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Stage 0.5: Vision Domain Adaptation')
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
    logger.info(f"Starting Stage 0.5 training with config: {args.config}")
    
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
    best_acc = 0.0
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
    
    # 训练循环
    train_config = config['training_params']
    epochs = train_config['epochs']
    log_freq = config['logging_params']['log_freq']
    save_freq = config['checkpoint_params']['save_freq']
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(start_epoch, epochs + 1):
        logger.info(f"\nEpoch {epoch}/{epochs}")
        
        # 训练阶段
        train_results = train_one_epoch(
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
        
        # 验证阶段
        val_results = evaluate_one_epoch(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            metrics=['loss', 'accuracy'],
            logger=logger
        )
        
        # 记录到TensorBoard
        writer.add_scalar('Train/Loss', train_results['average_loss'], epoch)
        writer.add_scalar('Train/LearningRate', train_results['learning_rate'], epoch)
        writer.add_scalar('Val/Loss', val_results['average_loss'], epoch)
        
        if 'accuracy' in val_results:
            writer.add_scalar('Val/Accuracy', val_results['accuracy'], epoch)
            current_acc = val_results['accuracy']
        else:
            current_acc = 0.0
        
        # 保存检查点
        is_best = current_acc > best_acc
        if is_best:
            best_acc = current_acc
        
        if epoch % save_freq == 0 or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_acc=best_acc,
                checkpoint_dir=checkpoint_dir,
                is_best=is_best
            )
        
        # 打印epoch总结
        logger.info(f"Epoch {epoch} - Train Loss: {train_results['average_loss']:.4f}, "
                   f"Val Loss: {val_results['average_loss']:.4f}, "
                   f"Val Acc: {current_acc:.4f}, Best Acc: {best_acc:.4f}")
    
    # 训练完成
    logger.info(f"Training completed! Best accuracy: {best_acc:.4f}")
    
    # 保存最终模型
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pth')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'best_acc': best_acc,
        'config': config
    }, final_checkpoint_path)
    
    logger.info(f"Final model saved to: {final_checkpoint_path}")
    
    # 关闭TensorBoard写入器
    writer.close()


if __name__ == '__main__':
    main()
