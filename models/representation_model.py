"""
Representation Model for Project Poseidon
集成视觉和触觉编码器的多模态表征学习模型
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import os

from .vision_encoder import ViTEncoder
from .tactile_encoder import TransformerTactileEncoder


class RepresentationModel(nn.Module):
    """
    多模态表征学习模型
    集成视觉编码器和触觉编码器，通过对比学习训练统一的多模态表征
    """
    
    def __init__(
        self,
        # 视觉编码器参数
        vision_encoder_weights_path: Optional[str] = None,
        vision_model_name: str = 'vit_base_patch16_224',
        freeze_vision_encoder: bool = False,
        
        # 触觉编码器参数
        tactile_feature_dim: int = 54,
        tactile_seq_len: int = 100,
        tactile_d_model: int = 768,
        tactile_nhead: int = 12,
        tactile_num_layers: int = 6,
        tactile_dim_feedforward: int = 3072,
        tactile_dropout: float = 0.1,
        
        # 投影头参数
        embed_dim: int = 128,
        projection_hidden_dim: int = 768,
        projection_dropout: float = 0.1
    ):
        """
        Args:
            vision_encoder_weights_path: 预训练视觉编码器权重路径
            vision_model_name: 视觉模型名称
            freeze_vision_encoder: 是否冻结视觉编码器
            tactile_feature_dim: 触觉特征维度
            tactile_seq_len: 触觉序列长度
            tactile_d_model: 触觉编码器内部维度
            tactile_nhead: 触觉编码器注意力头数
            tactile_num_layers: 触觉编码器层数
            tactile_dim_feedforward: 触觉编码器前馈网络维度
            tactile_dropout: 触觉编码器dropout
            embed_dim: 共享嵌入空间维度
            projection_hidden_dim: 投影头隐藏层维度
            projection_dropout: 投影头dropout
        """
        super(RepresentationModel, self).__init__()
        
        # 保存配置
        self.embed_dim = embed_dim
        self.projection_hidden_dim = projection_hidden_dim
        self.freeze_vision_encoder = freeze_vision_encoder
        
        # 初始化视觉编码器
        self.vision_encoder = ViTEncoder(
            model_name=vision_model_name,
            pretrained=True,  # 总是从预训练开始
            num_classes=None  # 移除分类头，用于特征提取
        )
        
        # 加载视觉编码器权重（如果提供）
        if vision_encoder_weights_path and os.path.exists(vision_encoder_weights_path):
            self._load_vision_encoder_weights(vision_encoder_weights_path)
        
        # 冻结视觉编码器（如果需要）
        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        # 初始化触觉编码器
        self.tactile_encoder = TransformerTactileEncoder(
            feature_dim=tactile_feature_dim,
            seq_len=tactile_seq_len,
            d_model=tactile_d_model,
            nhead=tactile_nhead,
            num_layers=tactile_num_layers,
            dim_feedforward=tactile_dim_feedforward,
            dropout=tactile_dropout
        )
        
        # 视觉投影头：768 -> 768 -> 128
        self.vision_projection_head = nn.Sequential(
            nn.Linear(self.vision_encoder.feature_dim, projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(projection_dropout),
            nn.Linear(projection_hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)  # 添加Layer Normalization提升稳定性
        )
        
        # 触觉投影头：768 -> 768 -> 128
        self.tactile_projection_head = nn.Sequential(
            nn.Linear(tactile_d_model, projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(projection_dropout),
            nn.Linear(projection_hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 初始化投影头权重
        self._init_projection_heads()
    
    def _load_vision_encoder_weights(self, weights_path: str) -> None:
        """
        加载预训练的视觉编码器权重
        
        Args:
            weights_path: 权重文件路径
        """
        try:
            # 加载检查点
            checkpoint = torch.load(weights_path, map_location='cpu')
            
            # 提取模型权重
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 过滤出视觉编码器相关的权重
            vision_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('vision_encoder.'):
                    # 移除前缀
                    new_key = key.replace('vision_encoder.', '')
                    vision_state_dict[new_key] = value
                elif not key.startswith(('tactile_encoder.', 'projection_head')):
                    # 如果没有前缀，直接使用
                    vision_state_dict[key] = value
            
            # 加载权重
            missing_keys, unexpected_keys = self.vision_encoder.load_state_dict(
                vision_state_dict, strict=False
            )
            
            if missing_keys:
                print(f"Warning: Missing keys in vision encoder: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in vision encoder: {unexpected_keys}")
            
            print(f"Successfully loaded vision encoder weights from {weights_path}")
            
        except Exception as e:
            print(f"Warning: Failed to load vision encoder weights from {weights_path}: {e}")
            print("Continuing with ImageNet pretrained weights...")
    
    def _init_projection_heads(self) -> None:
        """初始化投影头权重"""
        for module in [self.vision_projection_head, self.tactile_projection_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self, 
        image: torch.Tensor, 
        tactile_sequence: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            image: 视觉输入，形状为 (N, 3, H, W)
            tactile_sequence: 触觉输入，形状为 (N, seq_len, feature_dim)
            return_features: 是否返回编码器的原始特征
        
        Returns:
            视觉嵌入和触觉嵌入的元组，每个形状为 (N, embed_dim)
            如果return_features=True，还会返回原始特征
        """
        # 视觉特征提取
        vision_features = self.vision_encoder(image)  # (N, 768)
        
        # 触觉特征提取
        tactile_features = self.tactile_encoder(tactile_sequence)  # (N, 768)
        
        # 通过投影头映射到共享空间
        vision_embedding = self.vision_projection_head(vision_features)  # (N, embed_dim)
        tactile_embedding = self.tactile_projection_head(tactile_features)  # (N, embed_dim)
        
        if return_features:
            return (vision_embedding, tactile_embedding), (vision_features, tactile_features)
        else:
            return vision_embedding, tactile_embedding
    
    def encode_vision(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        仅编码视觉输入
        
        Args:
            image: 视觉输入，形状为 (N, 3, H, W)
        
        Returns:
            (vision_embedding, vision_features)
        """
        with torch.set_grad_enabled(not self.freeze_vision_encoder):
            vision_features = self.vision_encoder(image)
        
        vision_embedding = self.vision_projection_head(vision_features)
        return vision_embedding, vision_features
    
    def encode_tactile(self, tactile_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        仅编码触觉输入
        
        Args:
            tactile_sequence: 触觉输入，形状为 (N, seq_len, feature_dim)
        
        Returns:
            (tactile_embedding, tactile_features)
        """
        tactile_features = self.tactile_encoder(tactile_sequence)
        tactile_embedding = self.tactile_projection_head(tactile_features)
        return tactile_embedding, tactile_features
    
    def compute_similarity(
        self, 
        vision_embedding: torch.Tensor, 
        tactile_embedding: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        计算视觉和触觉嵌入之间的相似度
        
        Args:
            vision_embedding: 视觉嵌入，形状为 (N, embed_dim)
            tactile_embedding: 触觉嵌入，形状为 (M, embed_dim)
            temperature: 温度系数
        
        Returns:
            相似度矩阵，形状为 (N, M)
        """
        # L2归一化
        vision_embedding = nn.functional.normalize(vision_embedding, p=2, dim=1)
        tactile_embedding = nn.functional.normalize(tactile_embedding, p=2, dim=1)
        
        # 计算余弦相似度
        similarity = torch.matmul(vision_embedding, tactile_embedding.T) / temperature
        
        return similarity
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        vision_trainable = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
        
        tactile_params = sum(p.numel() for p in self.tactile_encoder.parameters())
        tactile_trainable = sum(p.numel() for p in self.tactile_encoder.parameters() if p.requires_grad)
        
        projection_params = sum(p.numel() for p in self.vision_projection_head.parameters()) + \
                          sum(p.numel() for p in self.tactile_projection_head.parameters())
        projection_trainable = sum(p.numel() for p in self.vision_projection_head.parameters() if p.requires_grad) + \
                             sum(p.numel() for p in self.tactile_projection_head.parameters() if p.requires_grad)
        
        total_params = vision_params + tactile_params + projection_params
        total_trainable = vision_trainable + tactile_trainable + projection_trainable
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': total_trainable,
            'vision_encoder': {
                'parameters': vision_params,
                'trainable': vision_trainable,
                'frozen': not (vision_trainable > 0)
            },
            'tactile_encoder': {
                'parameters': tactile_params,
                'trainable': tactile_trainable
            },
            'projection_heads': {
                'parameters': projection_params,
                'trainable': projection_trainable
            },
            'embed_dim': self.embed_dim
        }
    
    def print_model_info(self) -> None:
        """打印模型信息"""
        info = self.get_model_info()
        
        print("Representation Model Info:")
        print(f"  Total Parameters: {info['total_parameters']:,}")
        print(f"  Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"  Embedding Dimension: {info['embed_dim']}")
        print()
        print("  Vision Encoder:")
        print(f"    Parameters: {info['vision_encoder']['parameters']:,}")
        print(f"    Trainable: {info['vision_encoder']['trainable']:,}")
        print(f"    Frozen: {info['vision_encoder']['frozen']}")
        print()
        print("  Tactile Encoder:")
        print(f"    Parameters: {info['tactile_encoder']['parameters']:,}")
        print(f"    Trainable: {info['tactile_encoder']['trainable']:,}")
        print()
        print("  Projection Heads:")
        print(f"    Parameters: {info['projection_heads']['parameters']:,}")
        print(f"    Trainable: {info['projection_heads']['trainable']:,}")
    
    def save_encoders_separately(self, save_dir: str) -> None:
        """
        分别保存编码器（用于后续阶段）
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存视觉编码器
        vision_path = os.path.join(save_dir, 'vision_encoder.pth')
        torch.save(self.vision_encoder.state_dict(), vision_path)
        
        # 保存触觉编码器
        tactile_path = os.path.join(save_dir, 'tactile_encoder.pth')
        torch.save(self.tactile_encoder.state_dict(), tactile_path)
        
        print(f"Encoders saved to {save_dir}")
    
    def freeze_encoders(self) -> None:
        """冻结所有编码器（用于策略学习阶段）"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.tactile_encoder.parameters():
            param.requires_grad = False
        
        print("All encoders frozen for policy learning stage")
