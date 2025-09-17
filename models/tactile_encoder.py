"""
Tactile Encoder for Project Poseidon
基于Transformer的触觉编码器，用于处理触觉时间序列数据
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码
    为Transformer提供位置信息
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout比例
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除数项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加批次维度并注册为buffer
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (seq_len, batch_size, d_model)
        
        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerTactileEncoder(nn.Module):
    """
    基于Transformer的触觉编码器
    接收多维触觉时间序列数据，输出固定维度的特征向量
    """
    
    def __init__(
        self,
        feature_dim: int = 54,
        seq_len: int = 100,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 6,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        """
        Args:
            feature_dim: 输入特征维度（18传感器 × 3轴 = 54）
            seq_len: 输入序列长度
            d_model: Transformer模型内部维度
            nhead: 多头注意力的头数
            num_layers: Transformer编码器层数
            dim_feedforward: 前馈网络隐藏层维度
            dropout: Dropout比例
            activation: 激活函数类型
        """
        super(TransformerTactileEncoder, self).__init__()
        
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # 输入嵌入层：将54维特征映射到d_model维
        self.input_embedding = nn.Linear(feature_dim, d_model)
        
        # CLS token：用于序列表征的可学习参数
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=seq_len + 1,  # +1 for CLS token
            dropout=dropout
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False  # 使用 (seq_len, batch, d_model) 格式
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self) -> None:
        """初始化模型参数"""
        # 初始化CLS token
        nn.init.normal_(self.cls_token, std=0.02)
        
        # 初始化线性层
        nn.init.xavier_uniform_(self.input_embedding.weight)
        nn.init.constant_(self.input_embedding.bias, 0)
    
    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入触觉序列，形状为 (N, seq_len, feature_dim)
            src_key_padding_mask: 填充掩码，形状为 (N, seq_len+1)
        
        Returns:
            触觉特征向量，形状为 (N, d_model)
        """
        batch_size, seq_len, feature_dim = x.shape
        
        # 确保输入维度正确
        assert feature_dim == self.feature_dim, f"Expected feature_dim {self.feature_dim}, got {feature_dim}"
        
        # 输入嵌入：(N, seq_len, feature_dim) -> (N, seq_len, d_model)
        embedded = self.input_embedding(x)
        
        # 扩展CLS token到当前批次大小：(1, 1, d_model) -> (N, 1, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # 拼接CLS token和嵌入向量：(N, seq_len+1, d_model)
        sequence = torch.cat([cls_tokens, embedded], dim=1)
        
        # 转换为Transformer期望的格式：(seq_len+1, N, d_model)
        sequence = sequence.transpose(0, 1)
        
        # 添加位置编码
        sequence = self.positional_encoding(sequence)
        
        # 应用Dropout
        sequence = self.dropout_layer(sequence)
        
        # 通过Transformer编码器
        encoded = self.transformer_encoder(
            sequence, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # 提取CLS token的输出（第一个位置）
        cls_output = encoded[0]  # (N, d_model)
        
        # Layer Normalization
        cls_output = self.layer_norm(cls_output)
        
        return cls_output
    
    def create_padding_mask(self, x: torch.Tensor, valid_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        创建填充掩码
        
        Args:
            x: 输入序列，形状为 (N, seq_len, feature_dim)
            valid_lengths: 每个序列的有效长度，形状为 (N,)
        
        Returns:
            填充掩码，形状为 (N, seq_len+1)，True表示需要忽略的位置
        """
        batch_size, seq_len, _ = x.shape
        
        if valid_lengths is None:
            # 如果没有提供有效长度，假设所有位置都有效
            return torch.zeros(batch_size, seq_len + 1, dtype=torch.bool, device=x.device)
        
        # 创建掩码
        mask = torch.zeros(batch_size, seq_len + 1, dtype=torch.bool, device=x.device)
        
        for i, valid_len in enumerate(valid_lengths):
            if valid_len < seq_len:
                # 掩盖无效位置（+1是因为有CLS token）
                mask[i, valid_len + 1:] = True
        
        return mask
    
    def get_attention_weights(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        获取指定层的注意力权重（用于可视化）
        
        Args:
            x: 输入序列
            layer_idx: 层索引，-1表示最后一层
        
        Returns:
            注意力权重张量
        """
        # 这是一个简化的实现，实际需要修改Transformer来返回注意力权重
        # 可以通过hook或修改forward方法来实现
        
        attention_weights = []
        
        def hook_fn(module, input, output):
            # 注意力权重通常在MultiheadAttention的输出中
            if hasattr(output, 'attn_output_weights'):
                attention_weights.append(output.attn_output_weights)
        
        # 注册hook到指定层
        target_layer = self.transformer_encoder.layers[layer_idx]
        handle = target_layer.self_attn.register_forward_hook(hook_fn)
        
        try:
            # 前向传播
            with torch.no_grad():
                _ = self.forward(x)
        finally:
            # 移除hook
            handle.remove()
        
        return attention_weights[0] if attention_weights else None
    
    def get_sequence_representations(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取所有位置的表征（而不仅仅是CLS token）
        
        Args:
            x: 输入序列，形状为 (N, seq_len, feature_dim)
        
        Returns:
            所有位置的表征，形状为 (N, seq_len+1, d_model)
        """
        batch_size, seq_len, feature_dim = x.shape
        
        # 输入嵌入
        embedded = self.input_embedding(x)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls_tokens, embedded], dim=1)
        
        # 转换格式并添加位置编码
        sequence = sequence.transpose(0, 1)
        sequence = self.positional_encoding(sequence)
        sequence = self.dropout_layer(sequence)
        
        # 通过Transformer
        encoded = self.transformer_encoder(sequence)
        
        # 转换回 (N, seq_len+1, d_model) 格式
        encoded = encoded.transpose(0, 1)
        
        return encoded
    
    def print_model_info(self) -> None:
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Tactile Encoder Model Info:")
        print(f"  Input Feature Dim: {self.feature_dim}")
        print(f"  Sequence Length: {self.seq_len}")
        print(f"  Model Dimension: {self.d_model}")
        print(f"  Number of Heads: {self.nhead}")
        print(f"  Number of Layers: {self.num_layers}")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print(f"  Model Size (MB): {total_params * 4 / 1024 / 1024:.2f}")
