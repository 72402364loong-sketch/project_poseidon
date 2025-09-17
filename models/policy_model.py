"""
Policy Model for Project Poseidon
基于LSTM的策略模型，用于动态策略学习和DAgger训练
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, List
import numpy as np


class PolicyModel(nn.Module):
    """
    基于LSTM的策略模型
    具备记忆能力，能够处理时序信息并学习从感知到动作的映射
    """
    
    def __init__(
        self,
        # 状态向量参数
        vision_feature_dim: int = 768,
        tactile_feature_dim: int = 768,
        geometry_feature_dim: int = 3,  # 3D坐标 (X, Y, Z)
        
        # LSTM参数
        lstm_hidden_dim: int = 512,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.1,
        
        # 输出参数
        action_dim: int = 6,  # 6DOF机器人动作
        
        # MLP参数
        mlp_hidden_dims: List[int] = None,
        mlp_dropout: float = 0.1,
        
        # 其他参数
        use_layer_norm: bool = True,
        activation: str = 'relu'
    ):
        """
        Args:
            vision_feature_dim: 视觉特征维度
            tactile_feature_dim: 触觉特征维度
            geometry_feature_dim: 几何特征维度
            lstm_hidden_dim: LSTM隐藏层维度
            lstm_num_layers: LSTM层数
            lstm_dropout: LSTM dropout
            action_dim: 动作空间维度
            mlp_hidden_dims: MLP隐藏层维度列表
            mlp_dropout: MLP dropout
            use_layer_norm: 是否使用Layer Normalization
            activation: 激活函数类型
        """
        super(PolicyModel, self).__init__()
        
        # 保存配置
        self.vision_feature_dim = vision_feature_dim
        self.tactile_feature_dim = tactile_feature_dim
        self.geometry_feature_dim = geometry_feature_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.action_dim = action_dim
        self.use_layer_norm = use_layer_norm
        
        # 计算状态向量总维度
        self.state_dim = vision_feature_dim + tactile_feature_dim + geometry_feature_dim
        
        # 状态预处理层
        self.state_preprocessor = nn.Sequential(
            nn.Linear(self.state_dim, lstm_hidden_dim),
            nn.LayerNorm(lstm_hidden_dim) if use_layer_norm else nn.Identity(),
            self._get_activation(activation),
            nn.Dropout(mlp_dropout)
        )
        
        # LSTM核心网络
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            batch_first=True,  # 使用 (batch, seq, feature) 格式
            bidirectional=False
        )
        
        # MLP输出头
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [256, 128]
        
        mlp_layers = []
        input_dim = lstm_hidden_dim
        
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                self._get_activation(activation),
                nn.Dropout(mlp_dropout)
            ])
            input_dim = hidden_dim
        
        # 最终输出层
        mlp_layers.append(nn.Linear(input_dim, action_dim))
        
        self.mlp_head = nn.Sequential(*mlp_layers)
        
        # 初始化权重
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        if activation.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif activation.lower() == 'gelu':
            return nn.GELU()
        elif activation.lower() == 'leaky_relu':
            return nn.LeakyReLU(inplace=True)
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        else:
            return nn.ReLU(inplace=True)
    
    def _init_weights(self) -> None:
        """初始化权重"""
        # 初始化线性层
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                # 初始化LSTM权重
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
                        # 设置forget gate bias为1（提升训练稳定性）
                        if 'bias_ih' in name:
                            param.data[module.hidden_size:2*module.hidden_size].fill_(1.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(
        self,
        states: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_hidden: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            states: 状态序列，形状为 (batch_size, seq_len, state_dim)
            hidden_state: LSTM隐藏状态 (h_0, c_0)
            return_hidden: 是否返回最终的隐藏状态
        
        Returns:
            动作预测，形状为 (batch_size, seq_len, action_dim)
            如果return_hidden=True，还返回最终隐藏状态
        """
        batch_size, seq_len, _ = states.shape
        
        # 状态预处理
        processed_states = self.state_preprocessor(states)  # (batch, seq, lstm_hidden_dim)
        
        # 通过LSTM
        lstm_output, final_hidden = self.lstm(processed_states, hidden_state)
        # lstm_output: (batch, seq, lstm_hidden_dim)
        
        # 通过MLP输出头
        actions = self.mlp_head(lstm_output)  # (batch, seq, action_dim)
        
        if return_hidden:
            return actions, final_hidden
        else:
            return actions
    
    def predict_step(
        self,
        state: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        单步预测（用于实时控制）
        
        Args:
            state: 单个状态，形状为 (batch_size, state_dim) 或 (state_dim,)
            hidden_state: LSTM隐藏状态
        
        Returns:
            (action, new_hidden_state)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)  # 添加batch维度
        
        if state.dim() == 2:
            state = state.unsqueeze(1)  # 添加序列维度: (batch, 1, state_dim)
        
        # 前向传播
        action, new_hidden = self.forward(state, hidden_state, return_hidden=True)
        
        # 移除序列维度
        action = action.squeeze(1)  # (batch, action_dim)
        
        return action, new_hidden
    
    def init_hidden_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化LSTM隐藏状态
        
        Args:
            batch_size: 批次大小
            device: 设备
        
        Returns:
            (h_0, c_0)
        """
        h_0 = torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_hidden_dim,
            device=device, dtype=torch.float32
        )
        c_0 = torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_hidden_dim,
            device=device, dtype=torch.float32
        )
        
        return h_0, c_0
    
    def reset_hidden_state(self, hidden_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        重置隐藏状态（保持形状但清零）
        
        Args:
            hidden_state: 当前隐藏状态
        
        Returns:
            重置后的隐藏状态
        """
        h, c = hidden_state
        return torch.zeros_like(h), torch.zeros_like(c)
    
    def compute_action_statistics(self, actions: torch.Tensor) -> Dict[str, float]:
        """
        计算动作统计信息
        
        Args:
            actions: 动作张量，形状为 (batch, seq, action_dim) 或 (batch, action_dim)
        
        Returns:
            统计信息字典
        """
        if actions.dim() == 3:
            # 展平序列维度
            actions = actions.reshape(-1, actions.shape[-1])
        
        stats = {}
        
        # 基本统计
        stats['mean'] = actions.mean(dim=0).cpu().numpy().tolist()
        stats['std'] = actions.std(dim=0).cpu().numpy().tolist()
        stats['min'] = actions.min(dim=0)[0].cpu().numpy().tolist()
        stats['max'] = actions.max(dim=0)[0].cpu().numpy().tolist()
        
        # 整体统计
        stats['overall_mean'] = float(actions.mean())
        stats['overall_std'] = float(actions.std())
        stats['overall_norm'] = float(torch.norm(actions, dim=1).mean())
        
        return stats
    
    def apply_action_constraints(self, actions: torch.Tensor, constraints: Dict[str, Any]) -> torch.Tensor:
        """
        应用动作约束
        
        Args:
            actions: 原始动作
            constraints: 约束字典，包含'min', 'max', 'velocity_limit'等
        
        Returns:
            约束后的动作
        """
        constrained_actions = actions.clone()
        
        # 位置约束
        if 'min' in constraints and 'max' in constraints:
            min_vals = torch.tensor(constraints['min'], device=actions.device, dtype=actions.dtype)
            max_vals = torch.tensor(constraints['max'], device=actions.device, dtype=actions.dtype)
            constrained_actions = torch.clamp(constrained_actions, min_vals, max_vals)
        
        # 速度约束
        if 'velocity_limit' in constraints:
            velocity_limit = constraints['velocity_limit']
            action_norm = torch.norm(constrained_actions, dim=-1, keepdim=True)
            scale_factor = torch.clamp(velocity_limit / (action_norm + 1e-8), max=1.0)
            constrained_actions = constrained_actions * scale_factor
        
        return constrained_actions
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        lstm_params = sum(p.numel() for p in self.lstm.parameters())
        mlp_params = sum(p.numel() for p in self.mlp_head.parameters())
        preprocessor_params = sum(p.numel() for p in self.state_preprocessor.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'lstm_hidden_dim': self.lstm_hidden_dim,
            'lstm_num_layers': self.lstm_num_layers,
            'component_parameters': {
                'lstm': lstm_params,
                'mlp_head': mlp_params,
                'preprocessor': preprocessor_params
            }
        }
    
    def print_model_info(self) -> None:
        """打印模型信息"""
        info = self.get_model_info()
        
        print("Policy Model Info:")
        print(f"  Total Parameters: {info['total_parameters']:,}")
        print(f"  Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"  State Dimension: {info['state_dim']}")
        print(f"  Action Dimension: {info['action_dim']}")
        print(f"  LSTM Hidden Dimension: {info['lstm_hidden_dim']}")
        print(f"  LSTM Layers: {info['lstm_num_layers']}")
        print()
        print("  Component Parameters:")
        print(f"    State Preprocessor: {info['component_parameters']['preprocessor']:,}")
        print(f"    LSTM: {info['component_parameters']['lstm']:,}")
        print(f"    MLP Head: {info['component_parameters']['mlp_head']:,}")
        print(f"    Model Size (MB): {info['total_parameters'] * 4 / 1024 / 1024:.2f}")


class DAggerTrainer:
    """
    DAgger训练器
    实现数据聚合的迭代模仿学习
    """
    
    def __init__(
        self,
        policy_model: PolicyModel,
        expert_policy: Any,  # 专家策略（可以是函数或另一个模型）
        device: torch.device,
        max_iterations: int = 10,
        episodes_per_iteration: int = 50,
        expert_data_ratio: float = 0.5
    ):
        """
        Args:
            policy_model: 策略模型
            expert_policy: 专家策略
            device: 设备
            max_iterations: 最大迭代次数
            episodes_per_iteration: 每次迭代收集的episode数
            expert_data_ratio: 专家数据比例
        """
        self.policy_model = policy_model
        self.expert_policy = expert_policy
        self.device = device
        self.max_iterations = max_iterations
        self.episodes_per_iteration = episodes_per_iteration
        self.expert_data_ratio = expert_data_ratio
        
        # 数据存储
        self.aggregated_states = []
        self.aggregated_actions = []
        self.iteration_data = []
    
    def collect_expert_corrections(
        self,
        policy_states: List[torch.Tensor],
        policy_actions: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        收集专家纠错数据
        
        Args:
            policy_states: 策略生成的状态序列
            policy_actions: 策略生成的动作序列
        
        Returns:
            (corrected_states, expert_actions)
        """
        corrected_states = []
        expert_actions = []
        
        for states, actions in zip(policy_states, policy_actions):
            # 对每个状态，获取专家的正确动作
            for state in states:
                if callable(self.expert_policy):
                    expert_action = self.expert_policy(state.cpu().numpy())
                    expert_action = torch.tensor(expert_action, dtype=torch.float32)
                else:
                    # 如果专家是另一个模型
                    with torch.no_grad():
                        expert_action = self.expert_policy.predict_step(state.unsqueeze(0))[0].squeeze(0)
                
                corrected_states.append(state)
                expert_actions.append(expert_action)
        
        return corrected_states, expert_actions
    
    def aggregate_data(
        self,
        new_states: List[torch.Tensor],
        new_actions: List[torch.Tensor]
    ) -> None:
        """
        聚合新数据到训练集
        
        Args:
            new_states: 新的状态数据
            new_actions: 新的动作数据
        """
        self.aggregated_states.extend(new_states)
        self.aggregated_actions.extend(new_actions)
    
    def get_training_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取训练数据
        
        Returns:
            (states_tensor, actions_tensor)
        """
        if not self.aggregated_states:
            return None, None
        
        states = torch.stack(self.aggregated_states)
        actions = torch.stack(self.aggregated_actions)
        
        return states, actions
    
    def save_iteration_data(self, iteration: int, data: Dict[str, Any]) -> None:
        """保存迭代数据"""
        data['iteration'] = iteration
        self.iteration_data.append(data)
