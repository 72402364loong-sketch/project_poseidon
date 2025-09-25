"""
Loss functions for Project Poseidon
包含InfoNCE对比损失等损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class InfoNCELoss(nn.Module):
    """
    InfoNCE对比学习损失函数
    用于视觉-触觉多模态表征学习
    """
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        """
        Args:
            temperature: 温度系数τ，用于调节logits的尺度
            reduction: 损失归约方式 ('mean', 'sum', 'none')
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(
        self,
        vision_embeddings: torch.Tensor,
        tactile_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算InfoNCE损失
        
        Args:
            vision_embeddings: 视觉嵌入，形状为 (N, D)
            tactile_embeddings: 触觉嵌入，形状为 (N, D)
            labels: 标签，如果为None则假设对角线为正样本
        
        Returns:
            InfoNCE损失值
        """
        batch_size = vision_embeddings.shape[0]
        device = vision_embeddings.device
        
        # L2归一化，转换为单位向量
        vision_embeddings = F.normalize(vision_embeddings, p=2, dim=1)
        tactile_embeddings = F.normalize(tactile_embeddings, p=2, dim=1)
        
        # 计算相似度矩阵 (余弦相似度)
        logits = torch.matmul(vision_embeddings, tactile_embeddings.T) / self.temperature
        
        # 创建标签（如果未提供）
        if labels is None:
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
        
        # 计算对称损失
        # 视觉到触觉的损失
        loss_v2t = self.criterion(logits, labels)
        
        # 触觉到视觉的损失
        loss_t2v = self.criterion(logits.T, labels)
        
        # 对称损失的平均值
        total_loss = (loss_v2t + loss_t2v) / 2.0
        
        return total_loss
    
    def compute_accuracy(
        self,
        vision_embeddings: torch.Tensor,
        tactile_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        top_k: int = 1
    ) -> Tuple[float, float]:
        """
        计算检索准确率
        
        Args:
            vision_embeddings: 视觉嵌入
            tactile_embeddings: 触觉嵌入
            labels: 标签
            top_k: Top-K准确率
        
        Returns:
            (vision_to_tactile_acc, tactile_to_vision_acc)
        """
        batch_size = vision_embeddings.shape[0]
        device = vision_embeddings.device
        
        if labels is None:
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
        
        # L2归一化
        vision_embeddings = F.normalize(vision_embeddings, p=2, dim=1)
        tactile_embeddings = F.normalize(tactile_embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(vision_embeddings, tactile_embeddings.T)
        
        # 视觉到触觉的准确率
        _, v2t_pred = torch.topk(similarity, top_k, dim=1)
        v2t_correct = (v2t_pred == labels.unsqueeze(1)).any(dim=1).float()
        v2t_acc = v2t_correct.mean().item()
        
        # 触觉到视觉的准确率
        _, t2v_pred = torch.topk(similarity.T, top_k, dim=1)
        t2v_correct = (t2v_pred == labels.unsqueeze(1)).any(dim=1).float()
        t2v_acc = t2v_correct.mean().item()
        
        return v2t_acc, t2v_acc


class FocalLoss(nn.Module):
    """
    Focal Loss
    用于处理类别不平衡问题
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: 平衡因子
            gamma: 聚焦参数
            reduction: 损失归约方式
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 预测logits，形状为 (N, C)
            targets: 真实标签，形状为 (N,)
        
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss (Huber Loss)
    用于策略学习的动作回归
    """
    
    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            beta: 平滑参数
            reduction: 损失归约方式
        """
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 预测值
            targets: 目标值
        
        Returns:
            Smooth L1 loss
        """
        diff = torch.abs(inputs - targets)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class PolicyLoss(nn.Module):
    """
    策略学习损失函数
    结合动作损失和正则化项
    """
    
    def __init__(
        self,
        action_loss_type: str = 'mse',
        action_weight: float = 1.0,
        regularization_weight: float = 0.01,
        velocity_weight: float = 0.1,
        smoothness_weight: float = 0.05
    ):
        """
        Args:
            action_loss_type: 动作损失类型 ('mse', 'smooth_l1', 'huber')
            action_weight: 动作损失权重
            regularization_weight: 正则化权重
            velocity_weight: 速度正则化权重
            smoothness_weight: 平滑性正则化权重
        """
        super(PolicyLoss, self).__init__()
        
        self.action_weight = action_weight
        self.regularization_weight = regularization_weight
        self.velocity_weight = velocity_weight
        self.smoothness_weight = smoothness_weight
        
        # 选择动作损失函数
        if action_loss_type == 'mse':
            self.action_criterion = nn.MSELoss()
        elif action_loss_type == 'smooth_l1':
            self.action_criterion = SmoothL1Loss()
        elif action_loss_type == 'huber':
            self.action_criterion = nn.HuberLoss()
        else:
            self.action_criterion = nn.MSELoss()
    
    def forward(
        self,
        predicted_actions: torch.Tensor,
        target_actions: torch.Tensor,
        model: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算策略损失
        
        Args:
            predicted_actions: 预测动作，形状为 (N, seq_len, action_dim)
            target_actions: 目标动作，形状为 (N, seq_len, action_dim)
            model: 模型（用于计算正则化项）
        
        Returns:
            (total_loss, loss_dict)
        """
        loss_dict = {}
        
        # 主要动作损失
        action_loss = self.action_criterion(predicted_actions, target_actions)
        loss_dict['action_loss'] = action_loss.item()
        
        total_loss = self.action_weight * action_loss
        
        # 速度正则化（相邻时间步的动作差异）
        if self.velocity_weight > 0 and predicted_actions.shape[1] > 1:
            pred_velocity = predicted_actions[:, 1:] - predicted_actions[:, :-1]
            target_velocity = target_actions[:, 1:] - target_actions[:, :-1]
            velocity_loss = F.mse_loss(pred_velocity, target_velocity)
            
            total_loss += self.velocity_weight * velocity_loss
            loss_dict['velocity_loss'] = velocity_loss.item()
        
        # 平滑性正则化（动作的二阶导数）
        if self.smoothness_weight > 0 and predicted_actions.shape[1] > 2:
            pred_acceleration = predicted_actions[:, 2:] - 2 * predicted_actions[:, 1:-1] + predicted_actions[:, :-2]
            smoothness_loss = torch.mean(pred_acceleration ** 2)
            
            total_loss += self.smoothness_weight * smoothness_loss
            loss_dict['smoothness_loss'] = smoothness_loss.item()
        
        # 模型参数正则化
        if self.regularization_weight > 0 and model is not None:
            l2_reg = torch.tensor(0.0, device=predicted_actions.device)
            for param in model.parameters():
                if param.requires_grad:
                    l2_reg += torch.norm(param, p=2)
            
            total_loss += self.regularization_weight * l2_reg
            loss_dict['l2_regularization'] = l2_reg.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


class ContrastiveLoss(nn.Module):
    """
    对比损失函数
    用于二元对比学习
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            margin: 边界值
            reduction: 损失归约方式
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embedding1: 第一个嵌入
            embedding2: 第二个嵌入
            labels: 标签，1表示相似，0表示不相似
        
        Returns:
            对比损失
        """
        # 计算欧几里得距离
        distance = F.pairwise_distance(embedding1, embedding2, p=2)
        
        # 对比损失
        loss = labels * distance.pow(2) + \
               (1 - labels) * F.relu(self.margin - distance).pow(2)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class TripletLoss(nn.Module):
    """
    三元组损失函数
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            margin: 边界值
            reduction: 损失归约方式
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor: 锚点嵌入
            positive: 正样本嵌入
            negative: 负样本嵌入
        
        Returns:
            三元组损失
        """
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_distance - neg_distance + self.margin)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
