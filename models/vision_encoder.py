"""
Vision Encoder for Project Poseidon
基于ViT的视觉编码器，用于提取图像特征
"""

import torch
import torch.nn as nn
import timm
from typing import Optional


class ViTEncoder(nn.Module):
    """
    Vision Transformer编码器
    封装预训练的ViT模型，用于从输入图像中提取高质量的特征向量
    """
    
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        num_classes: Optional[int] = None,
        freeze_layers: int = 0
    ):
        """
        Args:
            model_name: ViT模型名称
            pretrained: 是否使用ImageNet预训练权重
            num_classes: 分类头的类别数，如果为None则移除分类头
            freeze_layers: 冻结的层数（从底层开始）
        """
        super(ViTEncoder, self).__init__()
        
        self.model_name = model_name
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.freeze_layers = freeze_layers
        
        # 加载预训练的ViT模型
        try:
            self.vit = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes if num_classes is not None else 0
            )
        except Exception as e:
            print(f"⚠️  网络连接失败，尝试使用本地缓存或离线模式: {e}")
            # 尝试不使用预训练权重
            if pretrained:
                print("🔄 尝试不使用预训练权重加载模型...")
                self.vit = timm.create_model(
                    model_name,
                    pretrained=False,
                    num_classes=num_classes if num_classes is not None else 0
                )
                print("⚠️  警告：模型已加载但未使用预训练权重，性能可能受影响")
            else:
                raise e
        
        # 如果不需要分类，将head替换为Identity
        if num_classes is None:
            self.vit.head = nn.Identity()
        
        # 冻结指定层数的权重
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        # 获取特征维度
        self.feature_dim = self._get_feature_dim()
    
    def _freeze_layers(self, num_layers: int) -> None:
        """
        冻结指定数量的Transformer层
        
        Args:
            num_layers: 要冻结的层数
        """
        # 冻结patch embedding和位置编码
        if hasattr(self.vit, 'patch_embed'):
            for param in self.vit.patch_embed.parameters():
                param.requires_grad = False
        
        if hasattr(self.vit, 'pos_embed'):
            self.vit.pos_embed.requires_grad = False
        
        if hasattr(self.vit, 'cls_token'):
            self.vit.cls_token.requires_grad = False
        
        # 冻结指定数量的Transformer块
        if hasattr(self.vit, 'blocks'):
            for i, block in enumerate(self.vit.blocks):
                if i < num_layers:
                    for param in block.parameters():
                        param.requires_grad = False
        elif hasattr(self.vit, 'layers'):
            # 某些ViT实现使用'layers'而不是'blocks'
            for i, layer in enumerate(self.vit.layers):
                if i < num_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
    
    def _get_feature_dim(self) -> int:
        """获取特征维度"""
        # 创建一个dummy输入来获取输出维度
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            dummy_output = self.vit(dummy_input)
            if isinstance(dummy_output, tuple):
                feature_dim = dummy_output[0].shape[-1]
            else:
                feature_dim = dummy_output.shape[-1]
        
        return feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像张量，形状为 (N, 3, H, W)
        
        Returns:
            特征向量，形状为 (N, feature_dim)
        """
        # 确保输入尺寸正确
        if x.shape[-2:] != (224, 224):
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        # 通过ViT模型
        features = self.vit(x)
        
        # 如果输出是元组（某些ViT实现），取第一个元素
        if isinstance(features, tuple):
            features = features[0]
        
        return features
    
    def get_attention_maps(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        获取注意力图（可选功能，用于可视化）
        
        Args:
            x: 输入图像张量
            layer_idx: 要提取注意力的层索引，-1表示最后一层
        
        Returns:
            注意力图张量
        """
        # 这个功能需要修改ViT的forward方法来返回注意力权重
        # 这里提供一个基础实现框架
        
        def hook_fn(module, input, output):
            # 存储注意力权重
            if hasattr(output, 'attn_weights'):
                return output.attn_weights
            return None
        
        # 注册hook
        target_layer = None
        if hasattr(self.vit, 'blocks'):
            if layer_idx == -1:
                target_layer = self.vit.blocks[-1]
            else:
                target_layer = self.vit.blocks[layer_idx]
        
        if target_layer is not None:
            handle = target_layer.register_forward_hook(hook_fn)
            
            # 前向传播
            with torch.no_grad():
                _ = self.forward(x)
            
            # 移除hook
            handle.remove()
        
        # 注意：这是一个简化的实现，实际的注意力提取可能需要更复杂的处理
        return None
    
    def unfreeze_layers(self, num_layers: int = None) -> None:
        """
        解冻指定数量的层
        
        Args:
            num_layers: 要解冻的层数，如果为None则解冻所有层
        """
        if num_layers is None:
            # 解冻所有参数
            for param in self.vit.parameters():
                param.requires_grad = True
        else:
            # 解冻指定数量的层（从顶层开始）
            if hasattr(self.vit, 'blocks'):
                total_layers = len(self.vit.blocks)
                start_layer = max(0, total_layers - num_layers)
                
                for i in range(start_layer, total_layers):
                    for param in self.vit.blocks[i].parameters():
                        param.requires_grad = True
            
            # 解冻分类头（如果存在）
            if hasattr(self.vit, 'head') and not isinstance(self.vit.head, nn.Identity):
                for param in self.vit.head.parameters():
                    param.requires_grad = True
    
    def get_trainable_parameters(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
    
    def get_total_parameters(self) -> int:
        """获取总参数数量"""
        return sum(p.numel() for p in self.vit.parameters())
    
    def print_model_info(self) -> None:
        """打印模型信息"""
        trainable_params = self.get_trainable_parameters()
        total_params = self.get_total_parameters()
        
        print(f"ViT Encoder Model Info:")
        print(f"  Model Name: {self.model_name}")
        print(f"  Pretrained: {self.pretrained}")
        print(f"  Feature Dimension: {self.feature_dim}")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print(f"  Frozen Parameters: {total_params - trainable_params:,}")
        print(f"  Trainable Ratio: {trainable_params / total_params * 100:.2f}%")
