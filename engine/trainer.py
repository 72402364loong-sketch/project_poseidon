"""
Training engine for Project Poseidon
通用的训练引擎，封装单个epoch的训练逻辑
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler, autocast
import time
from tqdm import tqdm
from typing import Optional, Dict, Any, Callable
import logging


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    lr_scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    log_freq: int = 10,
    grad_clip_norm: Optional[float] = None,
    loss_fn_kwargs: Optional[Dict[str, Any]] = None,
    custom_forward_fn: Optional[Callable] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    训练一个epoch
    
    Args:
        model: 要训练的模型
        data_loader: 数据加载器
        optimizer: 优化器
        loss_fn: 损失函数
        device: 设备
        epoch: 当前epoch编号
        lr_scheduler: 学习率调度器
        scaler: 混合精度梯度缩放器
        log_freq: 日志打印频率
        grad_clip_norm: 梯度裁剪范数
        loss_fn_kwargs: 损失函数额外参数
        custom_forward_fn: 自定义前向传播函数
        logger: 日志记录器
    
    Returns:
        包含训练统计信息的字典
    """
    model.train()
    
    if loss_fn_kwargs is None:
        loss_fn_kwargs = {}
    
    # 初始化统计变量
    total_loss = 0.0
    total_samples = 0
    batch_count = 0
    start_time = time.time()
    
    # 用于存储额外的损失信息
    loss_components = {}
    
    # 创建进度条
    pbar = tqdm(
        data_loader,
        desc=f'Epoch {epoch:3d}',
        leave=True,
        dynamic_ncols=True
    )
    
    for batch_idx, batch_data in enumerate(pbar):
        batch_start_time = time.time()
        
        try:
            # 数据移动到设备
            if isinstance(batch_data, (list, tuple)):
                batch_data = [data.to(device) if isinstance(data, torch.Tensor) else data 
                             for data in batch_data]
            elif isinstance(batch_data, dict):
                batch_data = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                             for key, value in batch_data.items()}
            else:
                batch_data = batch_data.to(device)
            
            batch_size = _get_batch_size(batch_data)
            
            # 清空梯度
            optimizer.zero_grad()
            
            # 前向传播
            with autocast(enabled=scaler is not None):
                if custom_forward_fn is not None:
                    # 使用自定义前向传播函数
                    model_output = custom_forward_fn(model, batch_data)
                else:
                    # 默认前向传播
                    if isinstance(batch_data, (list, tuple)):
                        model_output = model(*batch_data[:-1])  # 假设最后一个是标签
                        targets = batch_data[-1]
                    elif isinstance(batch_data, dict):
                        # 假设有'inputs'和'targets'键
                        inputs = batch_data.get('inputs', batch_data)
                        targets = batch_data.get('targets', None)
                        model_output = model(inputs)
                    else:
                        model_output = model(batch_data)
                        targets = None
                
                # 计算损失
                if hasattr(loss_fn, '__call__'):
                    if isinstance(model_output, (list, tuple)) and len(model_output) == 2:
                        # 对比学习情况（如InfoNCE）
                        loss_result = loss_fn(model_output[0], model_output[1], **loss_fn_kwargs)
                    elif targets is not None:
                        loss_result = loss_fn(model_output, targets, **loss_fn_kwargs)
                    else:
                        loss_result = loss_fn(model_output, **loss_fn_kwargs)
                else:
                    raise ValueError("loss_fn must be callable")
                
                # 处理损失结果
                if isinstance(loss_result, tuple):
                    loss, loss_dict = loss_result
                    # 更新损失组件统计
                    for key, value in loss_dict.items():
                        if key not in loss_components:
                            loss_components[key] = 0.0
                        loss_components[key] += value * batch_size
                else:
                    loss = loss_result
            
            # 检查损失是否有效
            if not torch.isfinite(loss):
                if logger:
                    logger.warning(f"Non-finite loss detected at batch {batch_idx}, skipping...")
                continue
            
            # 反向传播
            if scaler is not None:
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                
                # 梯度裁剪
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                
                optimizer.step()
            
            # 更新统计信息
            loss_value = loss.item()
            total_loss += loss_value * batch_size
            total_samples += batch_size
            batch_count += 1
            
            # 更新进度条
            avg_loss = total_loss / total_samples
            current_lr = optimizer.param_groups[0]['lr']
            
            pbar.set_postfix({
                'Loss': f'{loss_value:.4f}',
                'Avg': f'{avg_loss:.4f}',
                'LR': f'{current_lr:.2e}'
            })
            
            # 定期打印日志
            if (batch_idx + 1) % log_freq == 0:
                batch_time = time.time() - batch_start_time
                samples_per_sec = batch_size / batch_time
                
                log_msg = (
                    f'Epoch {epoch:3d} | Batch {batch_idx + 1:4d}/{len(data_loader):4d} | '
                    f'Loss: {loss_value:.4f} | Avg Loss: {avg_loss:.4f} | '
                    f'LR: {current_lr:.2e} | Speed: {samples_per_sec:.1f} samples/sec'
                )
                
                if logger:
                    logger.info(log_msg)
                else:
                    print(log_msg)
        
        except Exception as e:
            if logger:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
            else:
                print(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    # 更新学习率调度器
    if lr_scheduler is not None:
        if hasattr(lr_scheduler, 'step'):
            lr_scheduler.step()
    
    # 计算最终统计信息
    epoch_time = time.time() - start_time
    avg_loss = total_loss / max(total_samples, 1)
    
    # 计算损失组件的平均值
    avg_loss_components = {}
    for key, value in loss_components.items():
        avg_loss_components[key] = value / max(total_samples, 1)
    
    # 构建返回结果
    results = {
        'epoch': epoch,
        'average_loss': avg_loss,
        'total_samples': total_samples,
        'epoch_time': epoch_time,
        'samples_per_sec': total_samples / epoch_time,
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    
    # 添加损失组件
    results.update(avg_loss_components)
    
    # 打印epoch总结
    summary_msg = (
        f'Epoch {epoch:3d} Summary: '
        f'Avg Loss: {avg_loss:.4f} | '
        f'Samples: {total_samples} | '
        f'Time: {epoch_time:.1f}s | '
        f'Speed: {total_samples / epoch_time:.1f} samples/sec'
    )
    
    if logger:
        logger.info(summary_msg)
    else:
        print(summary_msg)
    
    return results


def _get_batch_size(batch_data) -> int:
    """
    获取批次大小
    
    Args:
        batch_data: 批次数据
    
    Returns:
        批次大小
    """
    if isinstance(batch_data, torch.Tensor):
        return batch_data.shape[0]
    elif isinstance(batch_data, (list, tuple)):
        if len(batch_data) > 0 and isinstance(batch_data[0], torch.Tensor):
            return batch_data[0].shape[0]
    elif isinstance(batch_data, dict):
        for value in batch_data.values():
            if isinstance(value, torch.Tensor):
                return value.shape[0]
    
    return 1  # 默认值


def train_representation_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    **kwargs
) -> Dict[str, float]:
    """
    表征学习专用的训练函数
    """
    def custom_forward_fn(model, batch_data):
        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
            vision_data, tactile_data = batch_data[0], batch_data[1]
            vision_embedding, tactile_embedding = model(vision_data, tactile_data)
            return vision_embedding, tactile_embedding
        else:
            raise ValueError("Expected batch_data to contain vision and tactile data")
    
    return train_one_epoch(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epoch=epoch,
        custom_forward_fn=custom_forward_fn,
        **kwargs
    )


def train_policy_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    **kwargs
) -> Dict[str, float]:
    """
    策略学习专用的训练函数
    """
    def custom_forward_fn(model, batch_data):
        if isinstance(batch_data, dict):
            states = batch_data['states']
            actions = batch_data['actions']
            predicted_actions = model(states)
            return predicted_actions, actions
        else:
            raise ValueError("Expected batch_data to be a dictionary with 'states' and 'actions'")
    
    def custom_loss_fn(model_output, **loss_kwargs):
        predicted_actions, target_actions = model_output
        return loss_fn(predicted_actions, target_actions, model=model, **loss_kwargs)
    
    # 替换损失函数
    original_loss_fn = loss_fn
    
    return train_one_epoch(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        loss_fn=custom_loss_fn,
        device=device,
        epoch=epoch,
        custom_forward_fn=custom_forward_fn,
        **kwargs
    )
