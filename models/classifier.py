"""
Object Classifier for Project Poseidon
基于多模态特征的对象分类器
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class ObjectClassifier(nn.Module):
    """
    多模态对象分类器
    接收来自表征模型的视觉和触觉特征，进行物体类别分类
    """
    
    def __init__(
        self,
        feature_dim: int = 1536,  # 768 (vision) + 768 (tactile)
        hidden_dim: int = 512,
        num_classes: int = 15,
        dropout: float = 0.2
    ):
        """
        Args:
            feature_dim: 输入特征维度（视觉+触觉拼接后的维度）
            hidden_dim: 隐藏层维度
            num_classes: 物体类别数量
            dropout: dropout比率
        """
        super(ObjectClassifier, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 分类器网络
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 输入特征，形状为 (N, feature_dim)
                     可以是视觉特征、触觉特征或两者拼接
        
        Returns:
            分类logits，形状为 (N, num_classes)
        """
        return self.classifier(features)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'feature_dim': self.feature_dim,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes
        }
    
    def print_model_info(self) -> None:
        """打印模型信息"""
        info = self.get_model_info()
        
        print("Object Classifier Info:")
        print(f"  Total Parameters: {info['total_parameters']:,}")
        print(f"  Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"  Feature Dimension: {info['feature_dim']}")
        print(f"  Hidden Dimension: {info['hidden_dim']}")
        print(f"  Number of Classes: {info['num_classes']}")


class MultimodalObjectClassifier(nn.Module):
    """
    多模态对象分类器（增强版）
    可以分别处理视觉和触觉特征，然后进行融合分类
    """
    
    def __init__(
        self,
        vision_feature_dim: int = 768,
        tactile_feature_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 15,
        dropout: float = 0.2,
        fusion_method: str = 'concat'  # 'concat', 'add', 'attention'
    ):
        """
        Args:
            vision_feature_dim: 视觉特征维度
            tactile_feature_dim: 触觉特征维度
            hidden_dim: 隐藏层维度
            num_classes: 物体类别数量
            dropout: dropout比率
            fusion_method: 特征融合方法
        """
        super(MultimodalObjectClassifier, self).__init__()
        
        self.vision_feature_dim = vision_feature_dim
        self.tactile_feature_dim = tactile_feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        
        # 特征投影层（可选）
        if fusion_method == 'attention':
            self.vision_proj = nn.Linear(vision_feature_dim, hidden_dim)
            self.tactile_proj = nn.Linear(tactile_feature_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # 确定融合后的特征维度
        if fusion_method == 'concat':
            fused_dim = vision_feature_dim + tactile_feature_dim
        elif fusion_method in ['add', 'attention']:
            fused_dim = hidden_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # 分类器网络
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        vision_features: torch.Tensor, 
        tactile_features: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            vision_features: 视觉特征，形状为 (N, vision_feature_dim)
            tactile_features: 触觉特征，形状为 (N, tactile_feature_dim)
        
        Returns:
            分类logits，形状为 (N, num_classes)
        """
        if self.fusion_method == 'concat':
            # 直接拼接
            fused_features = torch.cat([vision_features, tactile_features], dim=1)
        elif self.fusion_method == 'add':
            # 相加（需要维度相同）
            if vision_features.shape[1] != tactile_features.shape[1]:
                vision_features = self.vision_proj(vision_features)
                tactile_features = self.tactile_proj(tactile_features)
            fused_features = vision_features + tactile_features
        elif self.fusion_method == 'attention':
            # 注意力融合
            vision_proj = self.vision_proj(vision_features).unsqueeze(1)  # (N, 1, hidden_dim)
            tactile_proj = self.tactile_proj(tactile_features).unsqueeze(1)  # (N, 1, hidden_dim)
            
            # 拼接作为query, key, value
            features = torch.cat([vision_proj, tactile_proj], dim=1)  # (N, 2, hidden_dim)
            attended_features, _ = self.attention(features, features, features)
            fused_features = attended_features.mean(dim=1)  # (N, hidden_dim)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return self.classifier(fused_features)
