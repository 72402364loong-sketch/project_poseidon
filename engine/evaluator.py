"""
Evaluation engine for Project Poseidon
通用的评估引擎，封装单个epoch的评估逻辑
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import time
from tqdm import tqdm
from typing import Optional, Dict, Any, Callable, List
import logging
import numpy as np


def evaluate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    metrics: Optional[List[str]] = None,
    loss_fn_kwargs: Optional[Dict[str, Any]] = None,
    custom_forward_fn: Optional[Callable] = None,
    custom_metrics_fn: Optional[Callable] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    评估一个epoch
    
    Args:
        model: 要评估的模型
        data_loader: 数据加载器
        loss_fn: 损失函数
        device: 设备
        epoch: 当前epoch编号
        metrics: 要计算的指标列表
        loss_fn_kwargs: 损失函数额外参数
        custom_forward_fn: 自定义前向传播函数
        custom_metrics_fn: 自定义指标计算函数
        logger: 日志记录器
    
    Returns:
        包含评估统计信息的字典
    """
    model.eval()
    
    if loss_fn_kwargs is None:
        loss_fn_kwargs = {}
    
    if metrics is None:
        metrics = ['loss']
    
    # 初始化统计变量
    total_loss = 0.0
    total_samples = 0
    batch_count = 0
    start_time = time.time()
    
    # 用于存储额外的损失信息和指标
    loss_components = {}
    metric_values = {}
    
    # 用于存储所有预测和目标（某些指标需要）
    all_predictions = []
    all_targets = []
    all_embeddings = {'vision': [], 'tactile': []}
    
    # 创建进度条
    pbar = tqdm(
        data_loader,
        desc=f'Eval {epoch:3d}',
        leave=True,
        dynamic_ncols=True
    )
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(pbar):
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
                
                # 前向传播
                with autocast():
                    if custom_forward_fn is not None:
                        # 使用自定义前向传播函数
                        model_output = custom_forward_fn(model, batch_data)
                    else:
                        # 默认前向传播
                        if isinstance(batch_data, (list, tuple)):
                            model_output = model(*batch_data[:-1])
                            targets = batch_data[-1]
                        elif isinstance(batch_data, dict):
                            inputs = batch_data.get('inputs', batch_data)
                            targets = batch_data.get('targets', None)
                            model_output = model(inputs)
                        else:
                            model_output = model(batch_data)
                            targets = None
                    
                    # 计算损失
                    if hasattr(loss_fn, '__call__'):
                        if isinstance(model_output, (list, tuple)) and len(model_output) == 2:
                            # 对比学习情况
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
                
                # 更新统计信息
                loss_value = loss.item()
                total_loss += loss_value * batch_size
                total_samples += batch_size
                batch_count += 1
                
                # 存储预测和目标用于指标计算
                if isinstance(model_output, (list, tuple)):
                    if len(model_output) == 2:
                        # 对比学习情况，存储嵌入
                        all_embeddings['vision'].append(model_output[0].cpu())
                        all_embeddings['tactile'].append(model_output[1].cpu())
                    else:
                        all_predictions.append(model_output[0].cpu())
                        if targets is not None:
                            all_targets.append(targets.cpu())
                else:
                    all_predictions.append(model_output.cpu())
                    if targets is not None:
                        all_targets.append(targets.cpu())
                
                # 更新进度条
                avg_loss = total_loss / total_samples
                pbar.set_postfix({
                    'Loss': f'{loss_value:.4f}',
                    'Avg': f'{avg_loss:.4f}'
                })
                
            except Exception as e:
                if logger:
                    logger.error(f"Error in evaluation batch {batch_idx}: {str(e)}")
                else:
                    print(f"Error in evaluation batch {batch_idx}: {str(e)}")
                continue
    
    # 计算最终统计信息
    epoch_time = time.time() - start_time
    avg_loss = total_loss / max(total_samples, 1)
    
    # 计算损失组件的平均值
    avg_loss_components = {}
    for key, value in loss_components.items():
        avg_loss_components[key] = value / max(total_samples, 1)
    
    # 计算指标
    if custom_metrics_fn is not None:
        # 使用自定义指标函数
        computed_metrics = custom_metrics_fn(
            all_predictions, all_targets, all_embeddings, metrics
        )
        metric_values.update(computed_metrics)
    else:
        # 使用默认指标计算
        computed_metrics = _compute_default_metrics(
            all_predictions, all_targets, all_embeddings, metrics
        )
        metric_values.update(computed_metrics)
    
    # 构建返回结果
    results = {
        'epoch': epoch,
        'average_loss': avg_loss,
        'total_samples': total_samples,
        'epoch_time': epoch_time,
        'samples_per_sec': total_samples / epoch_time
    }
    
    # 添加损失组件
    results.update(avg_loss_components)
    
    # 添加指标
    results.update(metric_values)
    
    # 打印评估总结
    summary_msg = (
        f'Eval {epoch:3d} Summary: '
        f'Avg Loss: {avg_loss:.4f} | '
        f'Samples: {total_samples} | '
        f'Time: {epoch_time:.1f}s'
    )
    
    # 添加主要指标到总结
    if 'accuracy' in metric_values:
        summary_msg += f' | Acc: {metric_values["accuracy"]:.4f}'
    if 'retrieval_recall@1' in metric_values:
        summary_msg += f' | R@1: {metric_values["retrieval_recall@1"]:.4f}'
    
    if logger:
        logger.info(summary_msg)
    else:
        print(summary_msg)
    
    return results


# --- 新增 Start ---
def get_action_with_uncertainty(
    policy_model: nn.Module,
    state: torch.Tensor,
    hidden_state: Optional[tuple] = None,
    mc_samples: int = 25
) -> tuple:
    """
    通过MC Dropout执行多次前向传播，计算动作的均值和不确定性。
    
    Args:
        policy_model: 策略模型
        state: 当前状态，形状为 (batch_size, state_dim) 或 (state_dim,)
        hidden_state: LSTM隐藏状态
        mc_samples: MC Dropout采样次数
        
    Returns:
        (mean_action, arm_uncertainty, gripper_uncertainty)
        - mean_action: 平均动作，形状为 (batch_size, 7) 或 (7,)
        - arm_uncertainty: 机械臂动作的不确定性（6维方差和）
        - gripper_uncertainty: 夹爪动作的不确定性（1维方差）
    """
    policy_model.eval()      # 切换到评估模式
    policy_model.enable_dropout() # 但强制激活Dropout

    actions = []
    with torch.no_grad():
        for _ in range(mc_samples):
            action, _ = policy_model.predict_step(state, hidden_state)
            actions.append(action)
    
    # 将多次采样的动作堆叠起来
    actions_tensor = torch.stack(actions) # Shape: (mc_samples, batch, action_dim)
    
    # 计算均值作为最终执行的动作
    mean_action = actions_tensor.mean(dim=0)
    
    # 计算方差作为不确定性得分
    variances = actions_tensor.var(dim=0)  # Shape: (batch, action_dim)
    
    # 解耦的不确定性计算
    # 前6维：机械臂动作 [dx, dy, dz, d_roll, d_pitch, d_yaw]
    arm_variances = variances[..., :6]  # Shape: (batch, 6)
    arm_uncertainty = arm_variances.sum(dim=-1)  # Shape: (batch,)
    
    # 第7维：夹爪动作 [gripper_angle]
    gripper_uncertainty = variances[..., 6]  # Shape: (batch,)
    
    # 如果是单样本，返回标量
    if mean_action.dim() == 1:
        arm_uncertainty = arm_uncertainty.item()
        gripper_uncertainty = gripper_uncertainty.item()
    
    return mean_action, arm_uncertainty, gripper_uncertainty


def should_request_expert_annotation(
    arm_uncertainty: float,
    gripper_uncertainty: float,
    arm_threshold: float,
    gripper_threshold: float
) -> bool:
    """
    判断是否应该请求专家标注
    
    Args:
        arm_uncertainty: 机械臂动作的不确定性
        gripper_uncertainty: 夹爪动作的不确定性
        arm_threshold: 机械臂不确定性阈值
        gripper_threshold: 夹爪不确定性阈值
        
    Returns:
        是否应该请求专家标注
    """
    return (arm_uncertainty > arm_threshold or 
            gripper_uncertainty > gripper_threshold)
# --- 新增 End ---


def _get_batch_size(batch_data) -> int:
    """获取批次大小"""
    if isinstance(batch_data, torch.Tensor):
        return batch_data.shape[0]
    elif isinstance(batch_data, (list, tuple)):
        if len(batch_data) > 0 and isinstance(batch_data[0], torch.Tensor):
            return batch_data[0].shape[0]
    elif isinstance(batch_data, dict):
        for value in batch_data.values():
            if isinstance(value, torch.Tensor):
                return value.shape[0]
    return 1


def _compute_default_metrics(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    embeddings: Dict[str, List[torch.Tensor]],
    metrics: List[str]
) -> Dict[str, float]:
    """
    计算默认指标
    
    Args:
        predictions: 预测结果列表
        targets: 目标结果列表
        embeddings: 嵌入字典
        metrics: 指标列表
    
    Returns:
        指标字典
    """
    computed_metrics = {}
    
    # 对于对比学习任务
    if embeddings['vision'] and embeddings['tactile'] and len(embeddings['vision']) > 0:
        vision_emb = torch.cat(embeddings['vision'], dim=0)
        tactile_emb = torch.cat(embeddings['tactile'], dim=0)
        
        # 检索准确率
        if 'retrieval_recall@1' in metrics:
            recall_1 = _compute_retrieval_recall(vision_emb, tactile_emb, k=1)
            computed_metrics['retrieval_recall@1'] = recall_1
        
        if 'retrieval_recall@5' in metrics:
            recall_5 = _compute_retrieval_recall(vision_emb, tactile_emb, k=5)
            computed_metrics['retrieval_recall@5'] = recall_5
        
        # 对比学习准确率
        if 'contrastive_accuracy' in metrics:
            acc = _compute_contrastive_accuracy(vision_emb, tactile_emb)
            computed_metrics['contrastive_accuracy'] = acc
    
    # 对于分类/回归任务
    if predictions and targets:
        pred_tensor = torch.cat(predictions, dim=0)
        target_tensor = torch.cat(targets, dim=0)
        
        # 分类准确率
        if 'accuracy' in metrics and pred_tensor.dim() > 1:
            if pred_tensor.shape[1] > 1:  # 多类分类
                pred_classes = torch.argmax(pred_tensor, dim=1)
                if target_tensor.dim() > 1:
                    target_classes = torch.argmax(target_tensor, dim=1)
                else:
                    target_classes = target_tensor
                accuracy = (pred_classes == target_classes).float().mean().item()
                computed_metrics['accuracy'] = accuracy
        
        # 回归指标
        if 'mse' in metrics:
            mse = torch.nn.functional.mse_loss(pred_tensor, target_tensor).item()
            computed_metrics['mse'] = mse
        
        if 'mae' in metrics:
            mae = torch.nn.functional.l1_loss(pred_tensor, target_tensor).item()
            computed_metrics['mae'] = mae
        
        # 策略学习特定指标
        if 'action_smoothness' in metrics and pred_tensor.dim() == 3:
            smoothness = _compute_action_smoothness(pred_tensor)
            computed_metrics['action_smoothness'] = smoothness
    
    return computed_metrics


def _compute_retrieval_recall(
    vision_emb: torch.Tensor,
    tactile_emb: torch.Tensor,
    k: int = 1
) -> float:
    """
    计算检索召回率
    
    Args:
        vision_emb: 视觉嵌入
        tactile_emb: 触觉嵌入
        k: Top-K
    
    Returns:
        召回率
    """
    # L2归一化
    vision_emb = torch.nn.functional.normalize(vision_emb, p=2, dim=1)
    tactile_emb = torch.nn.functional.normalize(tactile_emb, p=2, dim=1)
    
    # 计算相似度矩阵
    similarity = torch.matmul(vision_emb, tactile_emb.T)
    
    # 视觉到触觉的检索
    _, v2t_indices = torch.topk(similarity, k, dim=1)
    correct_v2t = 0
    for i in range(vision_emb.shape[0]):
        if i in v2t_indices[i]:
            correct_v2t += 1
    
    # 触觉到视觉的检索
    _, t2v_indices = torch.topk(similarity.T, k, dim=1)
    correct_t2v = 0
    for i in range(tactile_emb.shape[0]):
        if i in t2v_indices[i]:
            correct_t2v += 1
    
    # 平均召回率
    total_queries = vision_emb.shape[0] + tactile_emb.shape[0]
    total_correct = correct_v2t + correct_t2v
    
    return total_correct / total_queries


def _compute_contrastive_accuracy(
    vision_emb: torch.Tensor,
    tactile_emb: torch.Tensor
) -> float:
    """
    计算对比学习准确率
    
    Args:
        vision_emb: 视觉嵌入
        tactile_emb: 触觉嵌入
    
    Returns:
        准确率
    """
    # L2归一化
    vision_emb = torch.nn.functional.normalize(vision_emb, p=2, dim=1)
    tactile_emb = torch.nn.functional.normalize(tactile_emb, p=2, dim=1)
    
    # 计算相似度矩阵
    similarity = torch.matmul(vision_emb, tactile_emb.T)
    
    # 创建标签（对角线为正样本）
    batch_size = vision_emb.shape[0]
    labels = torch.arange(batch_size)
    
    # 计算准确率
    _, pred_indices = torch.max(similarity, dim=1)
    accuracy = (pred_indices == labels).float().mean().item()
    
    return accuracy


def _compute_action_smoothness(actions: torch.Tensor) -> float:
    """
    计算动作平滑度
    
    Args:
        actions: 动作张量，形状为 (batch, seq, action_dim)
    
    Returns:
        平滑度指标
    """
    if actions.shape[1] <= 1:
        return 0.0
    
    # 计算相邻时间步的动作差异
    action_diff = actions[:, 1:] - actions[:, :-1]
    
    # 计算平滑度（差异的平均范数）
    smoothness = torch.norm(action_diff, dim=2).mean().item()
    
    return smoothness


def evaluate_representation_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    **kwargs
) -> Dict[str, float]:
    """
    表征学习专用的评估函数
    """
    def custom_forward_fn(model, batch_data):
        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
            vision_data, tactile_data = batch_data[0], batch_data[1]
            vision_embedding, tactile_embedding = model(vision_data, tactile_data)
            return vision_embedding, tactile_embedding
        else:
            raise ValueError("Expected batch_data to contain vision and tactile data")
    
    def custom_metrics_fn(predictions, targets, embeddings, metrics):
        """自定义指标计算函数"""
        computed_metrics = {}
        
        if embeddings['vision'] and embeddings['tactile']:
            vision_emb = torch.cat(embeddings['vision'], dim=0)
            tactile_emb = torch.cat(embeddings['tactile'], dim=0)
            
            # 检索指标
            for metric in metrics:
                if metric == 'retrieval_recall@1':
                    computed_metrics[metric] = _compute_retrieval_recall(vision_emb, tactile_emb, k=1)
                elif metric == 'retrieval_recall@5':
                    computed_metrics[metric] = _compute_retrieval_recall(vision_emb, tactile_emb, k=5)
                elif metric == 'accuracy':
                    computed_metrics[metric] = _compute_contrastive_accuracy(vision_emb, tactile_emb)
        
        return computed_metrics
    
    return evaluate_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_fn=loss_fn,
        device=device,
        epoch=epoch,
        custom_forward_fn=custom_forward_fn,
        custom_metrics_fn=custom_metrics_fn,
        **kwargs
    )


def evaluate_policy_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    **kwargs
) -> Dict[str, float]:
    """
    策略学习专用的评估函数
    """
    def custom_forward_fn(model, batch_data):
        if isinstance(batch_data, dict):
            states = batch_data['states']
            actions = batch_data['actions']
            predicted_actions = model(states)
            return predicted_actions, actions
        else:
            raise ValueError("Expected batch_data to be a dictionary")
    
    def custom_loss_fn(model_output, **loss_kwargs):
        predicted_actions, target_actions = model_output
        return loss_fn(predicted_actions, target_actions, **loss_kwargs)
    
    return evaluate_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_fn=custom_loss_fn,
        device=device,
        epoch=epoch,
        custom_forward_fn=custom_forward_fn,
        **kwargs
    )


def evaluate_classification_epoch(
    representation_model: nn.Module,
    classifier: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    分类任务专用的评估函数
    """
    representation_model.eval()
    classifier.eval()
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            # 数据移动到设备
            vision_data, tactile_data, labels = batch_data
            vision_data = vision_data.to(device)
            tactile_data = tactile_data.to(device)
            labels = labels.to(device)
            
            batch_size = vision_data.shape[0]
            
            # 前向传播
            with torch.cuda.amp.autocast():
                # 使用表征模型提取特征
                vision_emb, tactile_emb, _ = representation_model(vision_data, tactile_data)
                
                # 拼接特征
                features = torch.cat([vision_emb, tactile_emb], dim=1)
                
                # 分类
                logits = classifier(features)
                
                # 计算损失
                loss = loss_fn(logits, labels)
            
            # 更新统计信息
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 收集预测结果
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    avg_loss = total_loss / total_samples
    
    results = {
        'epoch': epoch,
        'average_loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_samples': total_samples
    }
    
    # 打印评估总结
    summary_msg = (
        f'Eval {epoch:3d} Summary: '
        f'Loss: {avg_loss:.4f} | '
        f'Acc: {accuracy:.4f} | '
        f'Prec: {precision:.4f} | '
        f'Rec: {recall:.4f} | '
        f'F1: {f1:.4f}'
    )
    
    if logger:
        logger.info(summary_msg)
    else:
        print(summary_msg)
    
    return results


# --- 新增 Start ---
def get_action_with_uncertainty(
    policy_model: nn.Module,
    state: torch.Tensor,
    hidden_state: Optional[tuple] = None,
    mc_samples: int = 25
) -> tuple:
    """
    通过MC Dropout执行多次前向传播，计算动作的均值和不确定性。
    
    Args:
        policy_model: 策略模型
        state: 当前状态，形状为 (batch_size, state_dim) 或 (state_dim,)
        hidden_state: LSTM隐藏状态
        mc_samples: MC Dropout采样次数
        
    Returns:
        (mean_action, arm_uncertainty, gripper_uncertainty)
        - mean_action: 平均动作，形状为 (batch_size, 7) 或 (7,)
        - arm_uncertainty: 机械臂动作的不确定性（6维方差和）
        - gripper_uncertainty: 夹爪动作的不确定性（1维方差）
    """
    policy_model.eval()      # 切换到评估模式
    policy_model.enable_dropout() # 但强制激活Dropout

    actions = []
    with torch.no_grad():
        for _ in range(mc_samples):
            action, _ = policy_model.predict_step(state, hidden_state)
            actions.append(action)
    
    # 将多次采样的动作堆叠起来
    actions_tensor = torch.stack(actions) # Shape: (mc_samples, batch, action_dim)
    
    # 计算均值作为最终执行的动作
    mean_action = actions_tensor.mean(dim=0)
    
    # 计算方差作为不确定性得分
    variances = actions_tensor.var(dim=0)  # Shape: (batch, action_dim)
    
    # 解耦的不确定性计算
    # 前6维：机械臂动作 [dx, dy, dz, d_roll, d_pitch, d_yaw]
    arm_variances = variances[..., :6]  # Shape: (batch, 6)
    arm_uncertainty = arm_variances.sum(dim=-1)  # Shape: (batch,)
    
    # 第7维：夹爪动作 [gripper_angle]
    gripper_uncertainty = variances[..., 6]  # Shape: (batch,)
    
    # 如果是单样本，返回标量
    if mean_action.dim() == 1:
        arm_uncertainty = arm_uncertainty.item()
        gripper_uncertainty = gripper_uncertainty.item()
    
    return mean_action, arm_uncertainty, gripper_uncertainty


def should_request_expert_annotation(
    arm_uncertainty: float,
    gripper_uncertainty: float,
    arm_threshold: float,
    gripper_threshold: float
) -> bool:
    """
    判断是否应该请求专家标注
    
    Args:
        arm_uncertainty: 机械臂动作的不确定性
        gripper_uncertainty: 夹爪动作的不确定性
        arm_threshold: 机械臂不确定性阈值
        gripper_threshold: 夹爪不确定性阈值
        
    Returns:
        是否应该请求专家标注
    """
    return (arm_uncertainty > arm_threshold or 
            gripper_uncertainty > gripper_threshold)
# --- 新增 End ---