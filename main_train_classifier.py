#!/usr/bin/env python3
"""
Stage 1.5: Multimodal Object Classification
阶段1.5：多模态对象分类的主训练脚本
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.representation_model import RepresentationModel
from models.classifier import ObjectClassifier
from data_loader.dataset import ClassificationDataset
from data_loader.augmentations import get_vision_transforms, get_tactile_transforms
from engine.trainer import train_one_epoch
from engine.evaluator import evaluate_classification_epoch


def setup_logging(log_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"stage1_5_classification_{timestamp}.log")
    
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
        underwater_augmentation=False
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
    train_dataset = ClassificationDataset(
        data_path=data_config['dataset_path'],
        split='train',
        vision_transform=train_vision_transforms,
        tactile_transform=train_tactile_transforms,
        tactile_seq_len=100,  # 默认值，可根据需要调整
        stereo_mode=True,
        num_classes=data_config['num_classes']
    )
    
    val_dataset = ClassificationDataset(
        data_path=data_config['dataset_path'],
        split='val',
        vision_transform=val_vision_transforms,
        tactile_transform=val_tactile_transforms,
        tactile_seq_len=100,
        stereo_mode=True,
        num_classes=data_config['num_classes']
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


def create_models(config: dict, device: torch.device) -> tuple:
    """创建表征模型和分类器"""
    model_config = config['model_params']
    
    # 创建表征模型（冻结）
    representation_model = RepresentationModel(
        # 视觉编码器参数
        vision_encoder_weights_path=model_config.get('representation_model_checkpoint'),
        vision_model_name='vit_base_patch16_224',
        freeze_vision_encoder=False,  # 这里不冻结，因为我们要加载预训练权重
        
        # 触觉编码器参数
        tactile_feature_dim=54,
        tactile_seq_len=100,
        tactile_d_model=768,
        tactile_nhead=12,
        tactile_num_layers=6,
        tactile_dim_feedforward=3072,
        tactile_dropout=0.1,
        
        # 投影头参数
        embed_dim=128,
        projection_hidden_dim=768
    )
    
    # 加载预训练权重
    checkpoint_path = model_config['representation_model_checkpoint']
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading representation model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            representation_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            representation_model.load_state_dict(checkpoint)
        logger.info("Representation model loaded successfully")
    else:
        logger.warning(f"Representation model checkpoint not found at {checkpoint_path}")
    
    # 冻结表征模型
    for param in representation_model.parameters():
        param.requires_grad = False
    representation_model.eval()
    
    # 创建分类器
    classifier = ObjectClassifier(
        feature_dim=model_config['feature_dim'],
        hidden_dim=model_config['classifier_hidden_dim'],
        num_classes=config['data_params']['num_classes'],
        dropout=model_config.get('dropout', 0.2)
    )
    
    # 移动到设备
    representation_model = representation_model.to(device)
    classifier = classifier.to(device)
    
    # 打印模型信息
    logger.info("Representation Model Info:")
    representation_model.print_model_info()
    logger.info("Classifier Model Info:")
    classifier.print_model_info()
    
    return representation_model, classifier


def create_optimizer_and_scheduler(classifier: nn.Module, config: dict) -> tuple:
    """创建优化器和学习率调度器"""
    train_config = config['training_params']
    scheduler_config = config.get('scheduler_params', {})
    
    # 创建优化器（只优化分类器参数）
    if train_config['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(
            classifier.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
    elif train_config['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(
            classifier.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
            momentum=0.9
        )
    else:
        optimizer = optim.Adam(
            classifier.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
    
    # 创建学习率调度器
    scheduler = None
    if scheduler_config.get('type') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', 30),
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
    elif scheduler_config.get('type') == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 10),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    
    return optimizer, scheduler


def create_loss_function(config: dict, device: torch.device) -> nn.Module:
    """创建损失函数"""
    return nn.CrossEntropyLoss().to(device)


def custom_forward_fn(representation_model, classifier, batch_data):
    """自定义前向传播函数"""
    vision_data, tactile_data, labels = batch_data
    
    # 使用表征模型提取特征（不计算梯度）
    with torch.no_grad():
        vision_emb, tactile_emb = representation_model(vision_data, tactile_data)
    
    # 拼接特征
    features = torch.cat([vision_emb, tactile_emb], dim=1)
    
    # 分类
    logits = classifier(features)
    
    return logits, labels


def save_checkpoint(
    classifier: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    best_accuracy: float,
    checkpoint_dir: str,
    is_best: bool = False
) -> None:
    """保存检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'classifier_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy,
    }
    
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # 保存常规检查点
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # 保存最佳模型
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_classifier.pth')
        torch.save(checkpoint, best_path)
        print(f"Best classifier saved with accuracy: {best_accuracy:.4f}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Stage 1.5: Multimodal Object Classification')
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
    global logger
    logger = setup_logging(log_dir)
    logger.info(f"Starting Stage 1.5 training with config: {args.config}")
    
    # 创建数据集和数据加载器
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_datasets(config)
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, config)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # 打印类别分布
    train_dataset.print_class_distribution()
    val_dataset.print_class_distribution()
    
    # 创建模型
    logger.info("Creating models...")
    representation_model, classifier = create_models(config, device)
    
    # 创建优化器和调度器
    optimizer, scheduler = create_optimizer_and_scheduler(classifier, config)
    
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
    best_accuracy = 0.0
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
    
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
        train_results = train_classification_epoch(
            representation_model=representation_model,
            classifier=classifier,
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
            val_results = evaluate_classification_epoch(
                representation_model=representation_model,
                classifier=classifier,
                data_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                epoch=epoch,
                logger=logger
            )
            
            # 记录到TensorBoard
            writer.add_scalar('Val/Loss', val_results['average_loss'], epoch)
            writer.add_scalar('Val/Accuracy', val_results['accuracy'], epoch)
            
            current_accuracy = val_results['accuracy']
        else:
            val_results = {'average_loss': 0.0, 'accuracy': 0.0}
            current_accuracy = 0.0
        
        # 记录训练指标到TensorBoard
        writer.add_scalar('Train/Loss', train_results['average_loss'], epoch)
        writer.add_scalar('Train/LearningRate', train_results['learning_rate'], epoch)
        
        # 保存检查点
        is_best = current_accuracy > best_accuracy
        if is_best:
            best_accuracy = current_accuracy
        
        if epoch % save_freq == 0 or is_best:
            save_checkpoint(
                classifier=classifier,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_accuracy=best_accuracy,
                checkpoint_dir=checkpoint_dir,
                is_best=is_best
            )
        
        # 打印epoch总结
        if epoch % eval_freq == 0:
            logger.info(f"Epoch {epoch} - Train Loss: {train_results['average_loss']:.4f}, "
                       f"Val Loss: {val_results['average_loss']:.4f}, "
                       f"Val Accuracy: {current_accuracy:.4f}, Best Accuracy: {best_accuracy:.4f}")
        else:
            logger.info(f"Epoch {epoch} - Train Loss: {train_results['average_loss']:.4f}")
    
    # 训练完成
    logger.info(f"Training completed! Best Accuracy: {best_accuracy:.4f}")
    
    # 保存最终模型
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_classifier.pth')
    torch.save({
        'epoch': epochs,
        'classifier_state_dict': classifier.state_dict(),
        'best_accuracy': best_accuracy,
        'config': config
    }, final_checkpoint_path)
    
    logger.info(f"Final classifier saved to: {final_checkpoint_path}")
    
    # 关闭TensorBoard写入器
    writer.close()


def train_classification_epoch(
    representation_model: nn.Module,
    classifier: nn.Module,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    lr_scheduler=None,
    scaler=None,
    log_freq: int = 10,
    logger=None
) -> dict:
    """训练一个epoch的分类器"""
    classifier.train()
    representation_model.eval()  # 确保表征模型处于评估模式
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []
    
    for batch_idx, batch_data in enumerate(data_loader):
        # 数据移动到设备
        vision_data, tactile_data, labels = batch_data
        vision_data = vision_data.to(device)
        tactile_data = tactile_data.to(device)
        labels = labels.to(device)
        
        batch_size = vision_data.shape[0]
        
        # 清空梯度
        optimizer.zero_grad()
        
        # 前向传播
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits, _ = custom_forward_fn(representation_model, classifier, (vision_data, tactile_data, labels))
            loss = loss_fn(logits, labels)
        
        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # 更新统计信息
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # 收集预测结果
        predictions = torch.argmax(logits, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 定期打印日志
        if (batch_idx + 1) % log_freq == 0:
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = total_loss / total_samples
            logger.info(f'Epoch {epoch} | Batch {batch_idx + 1}/{len(data_loader)} | '
                       f'Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}')
    
    # 更新学习率调度器
    if lr_scheduler is not None:
        lr_scheduler.step()
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_predictions)
    avg_loss = total_loss / total_samples
    
    return {
        'epoch': epoch,
        'average_loss': avg_loss,
        'accuracy': accuracy,
        'total_samples': total_samples,
        'learning_rate': optimizer.param_groups[0]['lr']
    }


if __name__ == '__main__':
    main()
