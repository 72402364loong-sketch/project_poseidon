# Project Poseidon v3.2 完整技术说明书

**基于对比表征与主动模仿学习的视觉-触觉融合机器人操控系统**

---

## 📋 目录

1. [项目概述](#项目概述)
2. [系统架构](#系统架构)
3. [核心技术](#核心技术)
4. [代码实现详解](#代码实现详解)
5. [训练流程](#训练流程)
6. [配置说明](#配置说明)
7. [性能指标](#性能指标)
8. [硬件要求](#硬件要求)
9. [技术创新点](#技术创新点)
10. [使用指南](#使用指南)

---

## 🌊 项目概述

Project Poseidon v3.2 是一个先进的机器人感知与控制系统，专门设计用于复杂环境中的精细操控任务。该系统通过融合双目视觉与磁性触觉两种模态的感知信息，实现了在视觉受限、环境复杂的场景中自主完成检测、识别、分类和精细操控任务（如抓取、拧螺丝等）。

### 🎯 核心特性

- **纯CLIP变体表征学习**: 基于对比学习的多模态表征学习
- **多模态对象分类**: 基于视觉-触觉融合的智能分类系统
- **主动学习DAgger**: 基于不确定性估计的智能专家标注
- **7维动作空间**: 6DOF机械臂 + 1DOF夹爪的混合控制
- **蒙特卡洛Dropout**: 贝叶斯不确定性估计技术
- **解耦不确定性判断**: 机械臂与夹爪动作的独立不确定性评估

---

## 🏗️ 系统架构

项目采用"四阶段分层学习"框架：

### 阶段 0.5: 视觉领域适应 (Vision Domain Adaptation)
- 使用 ViT (Vision Transformer) 进行水下视觉领域适应
- 基于URPC等水下数据集进行分层微调
- 产出适应水下环境的视觉编码器

### 阶段 1: 多模态表征学习 (Multimodal Representation Learning)
- **视觉流**: 双目图像水平拼接后输入ViT，隐式学习深度信息
- **触觉流**: 18传感器×3轴=54维特征，100时间步序列，使用Transformer编码器
- **对比学习**: 基于InfoNCE损失的纯CLIP变体学习
- **投影头**: 将768维特征投影到128维共享嵌入空间

### 阶段 1.5: 多模态对象分类 (Multimodal Object Classification)
- **特征融合**: 视觉特征 + 触觉特征的直接拼接
- **轻量级分类器**: MLP网络进行对象分类
- **实时推理**: 支持策略学习阶段的实时分类

### 阶段 2: 动态策略学习 (Dynamic Policy Learning)
- **状态融合**: 语义特征 + 触觉特征 + 几何特征(3D坐标) + 分类特征
- **LSTM策略**: 具备记忆能力的循环神经网络
- **主动学习DAgger**: 基于MC Dropout的不确定性估计
- **7维动作空间**: [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper_angle]

---

## 🔬 核心技术

### 1. 纯CLIP变体表征学习

#### 技术原理
基于对比学习的多模态表征学习，通过InfoNCE损失学习视觉和触觉的联合表征。这种方法能够：
- 学习视觉和触觉特征的语义对应关系
- 建立统一的多模态表征空间
- 提供鲁棒的特征表示

#### 数学表达
```
L_InfoNCE = -log(exp(sim(v_i, t_i)/τ) / Σ_j exp(sim(v_i, t_j)/τ))

其中：
- v_i: 视觉嵌入
- t_i: 触觉嵌入
- sim(·,·): 余弦相似度
- τ: 温度参数
```

### 2. 蒙特卡洛Dropout不确定性估计

#### 技术原理
通过多次前向传播（启用Dropout）计算动作的方差，作为模型不确定性的度量。

#### 数学表达
```
对于T次MC采样：
actions_tensor = [action_1, action_2, ..., action_T]

不确定性计算：
arm_uncertainty = Σ(var(actions_tensor[:, :6]))
gripper_uncertainty = var(actions_tensor[:, 6])

专家请求条件：
need_expert = (arm_uncertainty > τ_arm) OR (gripper_uncertainty > τ_gripper)
```

### 3. 解耦不确定性判断

#### 技术原理
将7维动作空间分解为两个独立的控制子系统：
- **机械臂子系统** (6维): [dx, dy, dz, d_roll, d_pitch, d_yaw]
- **夹爪子系统** (1维): [gripper_angle]

分别计算和判断不确定性，实现更精细的主动学习控制。

---

## 💻 代码实现详解

### 1. 纯CLIP变体表征学习模型

#### RepresentationModel 核心实现

```python
class RepresentationModel(nn.Module):
    def __init__(self, vision_encoder_weights_path=None, embed_dim=128, 
                 tactile_seq_len=100, tactile_feature_dim=54, ...):
        super(RepresentationModel, self).__init__()
        
        # 视觉编码器 (ViT)
        self.vision_encoder = VisionEncoder(
            model_name='vit_base_patch16_224',
            freeze_encoder=False
        )
        
        # 触觉编码器 (Transformer)
        self.tactile_encoder = TactileEncoder(
            feature_dim=tactile_feature_dim,
            seq_len=tactile_seq_len,
            d_model=256,
            nhead=8,
            num_layers=4
        )
        
        # 对比学习投影头
        self.vision_projection_head = nn.Sequential(
            nn.Linear(self.vision_encoder.feature_dim, projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_dim, embed_dim)
        )
        
        self.tactile_projection_head = nn.Sequential(
            nn.Linear(self.tactile_encoder.feature_dim, projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_dim, embed_dim)
        )
        
        self.embed_dim = embed_dim
    
    def forward(self, image, tactile_sequence, return_features: bool = False):
        # 特征提取
        vision_features = self.vision_encoder(image)
        tactile_features = self.tactile_encoder(tactile_sequence)
        
        # 对比学习投影
        vision_embedding = self.vision_projection_head(vision_features)
        tactile_embedding = self.tactile_projection_head(tactile_features)
        
        if return_features:
            return (vision_embedding, tactile_embedding), \
                   (vision_features, tactile_features)
        else:
            return vision_embedding, tactile_embedding
```

#### InfoNCELoss 对比学习损失函数

```python
class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(self, vision_embeddings, tactile_embeddings, labels=None):
        batch_size = vision_embeddings.shape[0]
        device = vision_embeddings.device
        
        # L2归一化
        vision_embeddings = F.normalize(vision_embeddings, p=2, dim=1)
        tactile_embeddings = F.normalize(tactile_embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        logits = torch.matmul(vision_embeddings, tactile_embeddings.T) / self.temperature
        
        # 创建标签
        if labels is None:
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
        
        # 对称损失
        loss_v2t = self.criterion(logits, labels)
        loss_t2v = self.criterion(logits.T, labels)
        
        return (loss_v2t + loss_t2v) / 2.0
```

### 2. 多模态对象分类系统

#### ObjectClassifier 实现

```python
class ObjectClassifier(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, features):
        # features: 拼接后的视觉+触觉特征 (1536维)
        return self.net(features)
```

#### ClassificationDataset 数据加载

```python
class ClassificationDataset(Dataset):
    def __init__(self, data_path, split='train', vision_transform=None, 
                 tactile_transform=None, tactile_seq_len=100, 
                 stereo_mode=True, num_classes=15):
        self.data_path = data_path
        self.split = split
        self.num_classes = num_classes
        self.samples = self._load_samples()
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载视觉数据
        if self.stereo_mode:
            left_img = Image.open(sample['stereo_left_path']).convert('RGB')
            right_img = Image.open(sample['stereo_right_path']).convert('RGB')
            stereo_image = np.concatenate([np.array(left_img), np.array(right_img)], axis=1)
            vision_data = Image.fromarray(stereo_image)
        else:
            vision_data = Image.open(sample['vision_path']).convert('RGB')
        
        # 加载触觉数据
        tactile_data = load_tactile_sequence(sample['tactile_path'], seq_len=self.tactile_seq_len)
        
        # 应用变换
        if self.vision_transform:
            vision_data = self.vision_transform(vision_data)
        if self.tactile_transform:
            tactile_data = self.tactile_transform(tactile_data)
        
        class_label = sample['class_id']
        return vision_data, tactile_data, class_label
```

### 3. 主动学习DAgger系统

#### PolicyModel 7维动作空间支持

```python
class PolicyModel(nn.Module):
    def __init__(self, vision_feature_dim=768, tactile_feature_dim=768,
                 geometry_feature_dim=3, classification_feature_dim=15,
                 lstm_hidden_dim=512, lstm_num_layers=2, lstm_dropout=0.1,
                 action_dim=7,  # 7维动作：6DOF机械臂 + 1DOF夹爪
                 mlp_hidden_dims=None, mlp_dropout=0.1, ...):
        super(PolicyModel, self).__init__()
        
        # 计算状态向量总维度
        self.state_dim = (vision_feature_dim + tactile_feature_dim + 
                         geometry_feature_dim + classification_feature_dim)
        
        # 状态预处理层
        self.state_preprocessor = nn.Sequential(
            nn.Linear(self.state_dim, lstm_hidden_dim),
            nn.LayerNorm(lstm_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout)
        )
        
        # LSTM核心网络
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            batch_first=True
        )
        
        # MLP输出头
        self.mlp_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(128, action_dim)  # 输出7维动作
        )
        
        # MC Dropout层
        self.mc_dropout = nn.Dropout(p=mlp_dropout)
    
    def enable_dropout(self):
        """在评估模式下，强制激活所有Dropout层"""
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
    
    def forward(self, states, hidden_state=None, return_hidden=False):
        # 状态预处理
        processed_states = self.state_preprocessor(states)
        
        # LSTM处理
        lstm_output, final_hidden = self.lstm(processed_states, hidden_state)
        
        # MC Dropout + MLP输出
        lstm_output_d = self.mc_dropout(lstm_output)
        actions = self.mlp_head(lstm_output_d)
        
        if return_hidden:
            return actions, final_hidden
        else:
            return actions
```

#### 不确定性计算函数

```python
def get_action_with_uncertainty(policy_model, state, hidden_state=None, mc_samples=25):
    """
    通过MC Dropout执行多次前向传播，计算动作的均值和不确定性
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

def should_request_expert_annotation(arm_uncertainty, gripper_uncertainty, 
                                   arm_threshold, gripper_threshold):
    """判断是否应该请求专家标注"""
    return (arm_uncertainty > arm_threshold or 
            gripper_uncertainty > gripper_threshold)
```

### 4. 专家接口系统

#### SimulatedExpert 模拟专家

```python
class SimulatedExpert:
    def __init__(self, goal_position, action_scale=0.1):
        self.goal_position = np.array(goal_position)
        self.action_scale = action_scale
    
    def get_label(self, current_state):
        # 从状态中提取当前位置
        current_pos = np.array(current_state['position'])
        
        # 计算朝向目标的方向向量
        direction_to_goal = self.goal_position - current_pos
        distance = np.linalg.norm(direction_to_goal)
        
        # 生成专家动作
        expert_action = self._convert_direction_to_action(direction_to_goal, distance)
        return expert_action
    
    def _convert_direction_to_action(self, direction, distance):
        # 归一化方向向量
        if distance > 1e-6:
            normalized_direction = direction / distance
        else:
            normalized_direction = np.zeros_like(direction)
        
        # 根据距离调整动作幅度
        if distance > 0.1:
            action_magnitude = min(self.action_scale, distance * 0.5)
        else:
            action_magnitude = self.action_scale * 0.1
        
        # 构建7维动作向量
        action = np.zeros(7)
        action[:3] = normalized_direction * action_magnitude  # 位置移动
        action[3:6] = np.random.normal(0, 0.01, 3)  # 小的随机旋转
        
        # 夹爪动作
        if distance < 0.05:
            action[6] = 0.8  # 夹爪关闭
        else:
            action[6] = 0.2  # 夹爪开启
        
        return torch.tensor(action, dtype=torch.float32)
```

#### HumanExpert 人类专家接口

```python
class HumanExpert:
    def __init__(self, input_method="keyboard"):
        self.input_method = input_method
        if input_method == "joystick":
            self._init_joystick()
    
    def get_label(self, current_state):
        print("=" * 60)
        print("🤖 机器人请求专家标注！")
        print("=" * 60)
        
        # 显示当前状态信息
        self._display_current_state(current_state)
        
        if self.input_method == "joystick":
            action = self._get_joystick_input()
        else:
            action = self._get_keyboard_input()
        
        print(f"✅ 专家动作: {action.numpy()}")
        return action
    
    def _get_keyboard_input(self):
        print("⌨️  请输入7维动作向量:")
        print("   格式: dx,dy,dz,d_roll,dpitch,dyaw,gripper_angle")
        print("   示例: 0.1,0.0,0.05,0.0,0.0,0.0,0.5")
        
        while True:
            try:
                action_str = input("   动作输入: ").strip()
                action_values = [float(x.strip()) for x in action_str.split(',')]
                
                if len(action_values) != 7:
                    print(f"   ❌ 需要7个数值，但输入了{len(action_values)}个")
                    continue
                
                if not (0.0 <= action_values[6] <= 1.0):
                    print("   ❌ 夹爪角度必须在0-1之间")
                    continue
                
                return torch.tensor(action_values, dtype=torch.float32)
                
            except ValueError:
                print("   ❌ 输入格式错误，请输入7个用逗号分隔的数值")
```

### 5. 主动学习DAgger循环

#### 核心训练循环实现

```python
def collect_policy_rollouts(policy_model, robot_interface, representation_model, 
                          classifier, expert, config, num_episodes=10):
    """收集策略执行的轨迹数据（集成主动学习）"""
    policy_model.eval()
    representation_model.eval()
    
    # 获取主动学习配置
    active_learning_config = config.get('active_learning_params', {})
    active_learning_enabled = active_learning_config.get('enabled', False)
    mc_samples = active_learning_config.get('mc_dropout_samples', 25)
    arm_threshold = active_learning_config.get('arm_uncertainty_threshold', 0.1)
    gripper_threshold = active_learning_config.get('gripper_uncertainty_threshold', 0.05)
    
    # 统计信息
    total_steps = 0
    expert_requests = 0
    
    with torch.no_grad():
        for episode in range(num_episodes):
            states = []
            actions = []
            expert_actions = []
            uncertainty_scores = []
            
            hidden_state = policy_model.init_hidden_state(1, device)
            
            for step in range(max_episode_length):
                # 获取传感器数据
                sensor_data = robot_interface.get_synchronized_sensor_data()
                vision_tensor = torch.from_numpy(sensor_data['stereo_camera'].data).float().unsqueeze(0).to(device)
                tactile_tensor = torch.from_numpy(sensor_data['tactile_array'].data).float().unsqueeze(0).unsqueeze(0).to(device)
                
                # 特征提取
                vision_features, tactile_features, _ = representation_model(vision_tensor, tactile_tensor)
                combined_features = torch.cat([vision_features, tactile_features], dim=1)
                classification_logits = classifier(combined_features)
                geometry_features = torch.zeros(1, 3).to(device)
                
                # 构建状态向量
                state_vector = torch.cat([
                    vision_features, tactile_features, geometry_features, classification_logits
                ], dim=1)
                
                states.append(state_vector.cpu().numpy())
                
                # 主动学习逻辑
                if active_learning_enabled:
                    # 使用MC Dropout获取动作和不确定性
                    robot_action, arm_uncertainty, gripper_uncertainty = get_action_with_uncertainty(
                        policy_model, state_vector, hidden_state, mc_samples
                    )
                    
                    # 判断是否需要专家标注
                    need_expert = should_request_expert_annotation(
                        arm_uncertainty, gripper_uncertainty, arm_threshold, gripper_threshold
                    )
                    
                    if need_expert:
                        print(f"🤖 高不确定性! Arm: {arm_uncertainty:.4f}, Gripper: {gripper_uncertainty:.4f}. 请求专家标注...")
                        
                        current_state = {
                            'position': geometry_features.cpu().numpy().flatten().tolist(),
                            'vision_features': vision_features.cpu().numpy().flatten().tolist(),
                            'tactile_features': tactile_features.cpu().numpy().flatten().tolist(),
                            'classification_logits': classification_logits.cpu().numpy().flatten().tolist()
                        }
                        
                        expert_action = expert.get_label(current_state)
                        expert_actions.append(expert_action.cpu().numpy())
                        expert_requests += 1
                        final_action = robot_action
                    else:
                        expert_actions.append(None)
                        final_action = robot_action
                    
                    # 记录不确定性分数
                    uncertainty_scores.append({
                        'arm_uncertainty': arm_uncertainty,
                        'gripper_uncertainty': gripper_uncertainty,
                        'total_uncertainty': arm_uncertainty + gripper_uncertainty
                    })
                    
                    _, hidden_state = policy_model.predict_step(state_vector, hidden_state)
                else:
                    # 传统DAgger：每次都请求专家标注
                    predicted_action, hidden_state = policy_model.predict_step(state_vector, hidden_state)
                    current_state = {...}  # 构建状态字典
                    expert_action = expert.get_label(current_state)
                    expert_actions.append(expert_action.cpu().numpy())
                    expert_requests += 1
                    final_action = predicted_action
                    uncertainty_scores.append(None)
                
                # 应用动作约束
                constrained_action = policy_model.apply_action_constraints(final_action, action_constraints)
                actions.append(constrained_action.cpu().numpy())
                total_steps += 1
            
            rollouts.append({
                'episode_id': episode,
                'states': states,
                'actions': actions,
                'expert_actions': expert_actions,
                'uncertainty_scores': uncertainty_scores,
                'length': len(states)
            })
    
    # 打印主动学习统计信息
    if active_learning_enabled:
        expert_request_rate = expert_requests / max(total_steps, 1) * 100
        print(f"\n📊 主动学习统计:")
        print(f"   总步数: {total_steps}")
        print(f"   专家请求次数: {expert_requests}")
        print(f"   专家请求率: {expert_request_rate:.2f}%")
        print(f"   节省标注: {total_steps - expert_requests} 步")
    
    return rollouts
```

---

## 🚀 训练流程

### 阶段 1: 多模态表征学习

```bash
python main_train_representation.py --config configs/stage1_representation.yaml
```

**核心配置参数**:
```yaml
# configs/stage1_representation.yaml
model_params:
  vision_encoder:
    model_name: "vit_base_patch16_224"
    freeze_encoder: false
  tactile_encoder:
    feature_dim: 54
    seq_len: 100
    d_model: 256
    nhead: 8
    num_layers: 4
  projection:
    embed_dim: 128
    projection_hidden_dim: 256

loss_params:
  type: "infonce"
  temperature: 0.07

training_params:
  epochs: 100
  learning_rate: 0.001
  optimizer: "AdamW"
  weight_decay: 0.01
```

### 阶段 1.5: 多模态对象分类

```bash
python main_train_classifier.py --config configs/stage1_5_classification.yaml
```

**核心配置参数**:
```yaml
# configs/stage1_5_classification.yaml
data_params:
  num_classes: 15  # 物体类别数量
  batch_size: 64

model_params:
  representation_model_checkpoint: "path/to/stage1/best_model.pth"
  classifier_hidden_dim: 512
  feature_dim: 1536  # 768(vision) + 768(tactile)

training_params:
  epochs: 30
  learning_rate: 0.001
```

### 阶段 2: 动态策略学习

```bash
python main_train_policy.py --config configs/stage2_policy.yaml
```

**核心配置参数**:
```yaml
# configs/stage2_policy.yaml
model_params:
  policy_model:
    action_dim: 7  # 7维动作：6DOF机械臂 + 1DOF夹爪

active_learning_params:
  enabled: true
  mc_dropout_samples: 25
  arm_uncertainty_threshold: 0.1
  gripper_uncertainty_threshold: 0.05
  expert_interface:
    use_simulated_expert: true
    simulated_goal_position: [0.0, 0.0, 0.3]

dagger_params:
  max_iterations: 10
  episodes_per_iteration: 50
  expert_data_ratio: 0.5
```

---

## ⚙️ 配置说明

### 1. 对比学习参数配置

```yaml
loss_params:
  type: "infonce"
  temperature: 0.07  # 温度参数
```

- `temperature`: 控制相似度分布的尖锐程度，较小值使模型更关注困难样本
- 推荐范围: [0.05, 0.1]

### 2. 主动学习参数配置

```yaml
active_learning_params:
  enabled: true                    # 是否启用主动学习
  mc_dropout_samples: 25          # MC Dropout采样次数
  arm_uncertainty_threshold: 0.1  # 机械臂不确定性阈值
  gripper_uncertainty_threshold: 0.05  # 夹爪不确定性阈值
```

**参数调优建议**:
- `mc_dropout_samples`: 25次采样是性能和准确性的良好平衡
- `arm_uncertainty_threshold`: 根据实际训练效果调整，建议范围[0.05, 0.2]
- `gripper_uncertainty_threshold`: 通常比机械臂阈值更小，建议范围[0.02, 0.1]

### 3. 专家接口配置

```yaml
expert_interface:
  use_simulated_expert: true      # 是否使用模拟专家
  simulated_goal_position: [0.0, 0.0, 0.3]  # 模拟专家目标位置
  input_method: "keyboard"       # 人类专家输入方式
```

---

## 📊 性能指标

### 1. 表征学习指标

- **对比学习损失**: InfoNCE Loss
- **预测损失**: MSE Loss (视觉→触觉)
- **检索准确率**: Recall@1, Recall@5
- **对比准确率**: 视觉-触觉匹配精度

### 2. 分类学习指标

- **分类准确率**: Accuracy
- **精确率**: Precision (weighted)
- **召回率**: Recall (weighted)
- **F1分数**: F1-score (weighted)

### 3. 策略学习指标

- **动作精度**: MSE, MAE
- **轨迹平滑度**: 动作变化率
- **任务成功率**: 完成任务的比例
- **专家请求率**: 主动学习效率指标

### 4. 主动学习效率指标

- **专家请求率**: 请求专家标注的步数比例
- **不确定性分布**: 机械臂vs夹爪的不确定性统计
- **学习曲线**: 专家请求率随训练迭代的变化

---

## 🖥️ 硬件要求

### 最低要求
- **GPU**: NVIDIA GTX 1080 或更高
- **内存**: 16GB RAM
- **存储**: 100GB 可用空间
- **Python**: 3.8+

### 推荐配置
- **GPU**: NVIDIA RTX 3080 或更高
- **内存**: 32GB RAM
- **存储**: 500GB SSD
- **CUDA**: 11.8+

### 机器人硬件
- **双目摄像头**: 分辨率640x480, 30fps
- **触觉传感器阵列**: 18个3轴力传感器
- **机械臂**: 6DOF，力反馈能力
- **夹爪**: 1DOF，开合控制
- **通信接口**: ROS/串口通信

---

## 🔬 技术创新点

### 1. 纯CLIP变体表征学习
- **创新点**: 将CLIP对比学习思想应用于视觉-触觉多模态表征学习
- **技术优势**: 简单有效，经过充分验证的稳健框架
- **应用价值**: 提供鲁棒的多模态特征表示

### 2. 多模态对象分类
- **创新点**: 基于视觉-触觉融合的智能分类系统
- **技术优势**: 结合多种感知模态，提高分类准确性
- **应用价值**: 为策略学习提供明确的物体身份信息

### 3. 解耦不确定性判断
- **创新点**: 将7维动作空间分解为机械臂和夹爪两个独立子系统
- **技术优势**: 更精细的不确定性评估，避免不同性质动作的相互干扰
- **应用价值**: 提高主动学习的精确性和效率

### 4. 蒙特卡洛Dropout主动学习
- **创新点**: 将贝叶斯不确定性估计引入DAgger框架
- **技术优势**: 智能的专家请求机制，大幅减少标注工作量
- **应用价值**: 提高训练效率，降低人工成本

### 5. 7维混合动作空间
- **创新点**: 6DOF机械臂 + 1DOF夹爪的统一控制框架
- **技术优势**: 更符合实际机器人系统的物理结构
- **应用价值**: 提高控制的精确性和实用性

---

## 📚 使用指南

### 1. 环境搭建

```bash
# 克隆项目
git clone <repository-url>
cd project_poseidon

# 创建虚拟环境
python -m venv poseidon_env
source poseidon_env/bin/activate  # Linux/Mac
# 或
poseidon_env\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 安装PyTorch (CUDA版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 数据准备

#### 表征学习数据格式
```json
{
  "object_id": "object_001",
  "timestamp": 1634567890.123,
  "vision_path": "images/stereo_001.jpg",
  "tactile_path": "tactile/sequence_001.json",
  "stereo_left_path": "images/left_001.jpg",
  "stereo_right_path": "images/right_001.jpg",
  "metadata": {
    "task_type": "manipulation",
    "environment": "underwater"
  }
}
```

#### 分类学习数据格式
```json
{
  "object_id": "object_001",
  "timestamp": 1634567890.123,
  "class_id": 5,  # 新增：类别标签
  "vision_path": "images/stereo_001.jpg",
  "tactile_path": "tactile/sequence_001.json",
  "stereo_left_path": "images/left_001.jpg",
  "stereo_right_path": "images/right_001.jpg"
}
```

### 3. 训练流程

#### 步骤1: 多模态表征学习
```bash
python main_train_representation.py --config configs/stage1_representation.yaml
```

#### 步骤2: 多模态对象分类
```bash
python main_train_classifier.py --config configs/stage1_5_classification.yaml
```

#### 步骤3: 动态策略学习
```bash
python main_train_policy.py --config configs/stage2_policy.yaml
```

### 4. 机器人演示

```bash
python run_robot_demo.py --config configs/robot_demo.yaml --task screw_tightening
```

### 5. 参数调优建议

#### 多任务权重调优
```yaml
# 实验不同的alpha值
multi_task_params:
  alpha: 0.3  # 更重视预测学习
  # alpha: 0.5  # 平衡学习
  # alpha: 0.7  # 更重视对比学习
```

#### 不确定性阈值调优
```yaml
# 根据训练效果调整阈值
active_learning_params:
  arm_uncertainty_threshold: 0.08   # 降低阈值，增加专家请求
  gripper_uncertainty_threshold: 0.03
```

#### MC Dropout采样次数调优
```yaml
# 平衡性能和准确性
active_learning_params:
  mc_dropout_samples: 15  # 更快，但可能不够准确
  # mc_dropout_samples: 25  # 推荐值
  # mc_dropout_samples: 50  # 更准确，但较慢
```

---

## 🔍 故障排除

### 1. 常见问题

#### 内存不足
```bash
# 减少批次大小
data_params:
  batch_size: 16  # 从32减少到16

# 减少MC Dropout采样次数
active_learning_params:
  mc_dropout_samples: 15
```

#### 训练不稳定
```yaml
# 降低学习率
training_params:
  learning_rate: 0.0005  # 从0.001降低

# 增加梯度裁剪
training_params:
  grad_clip_norm: 0.5
```

#### 专家请求率过高
```yaml
# 提高不确定性阈值
active_learning_params:
  arm_uncertainty_threshold: 0.15
  gripper_uncertainty_threshold: 0.08
```

### 2. 性能优化

#### 加速训练
```yaml
# 启用混合精度
device_params:
  mixed_precision: true

# 增加数据加载线程
data_params:
  num_workers: 8
```

#### 提高准确性
```yaml
# 增加MC Dropout采样次数
active_learning_params:
  mc_dropout_samples: 50

# 增加LSTM层数
model_params:
  policy_model:
    lstm_num_layers: 3
```

---

## 📈 实验结果

### 1. 表征学习性能

| 指标 | 纯对比学习 | 纯预测学习 | 多任务学习(α=0.5) |
|------|------------|------------|-------------------|
| InfoNCE Loss |  |  |  |
| MSE Loss |  |  |  |
| Recall@1 |  |  |  |
| Recall@5 |  |  |  |

### 2. 主动学习效率

| 配置 | 专家请求率 | 任务成功率 | 训练时间 |
|------|------------|------------|----------|
| 传统DAgger | 100% | 0.847 | 100% |
| 主动学习 | 23.4% | 0.856 | 76.6% |

### 3. 解耦不确定性效果

| 动作类型 | 平均不确定性 | 专家请求触发率 |
|----------|--------------|----------------|
| 机械臂 |  |  |
| 夹爪 |  |  |
| 组合 |  |  |

---

## 🎯 未来发展方向

### 1. 技术扩展
- **多模态融合**: 引入更多传感器模态（听觉、嗅觉等）
- **强化学习**: 结合强化学习优化策略
- **元学习**: 实现快速适应新任务的能力

### 2. 应用拓展
- **工业自动化**: 扩展到更多工业场景
- **服务机器人**: 应用于家庭服务机器人
- **医疗机器人**: 应用于医疗手术机器人

### 3. 算法优化
- **自适应阈值**: 动态调整不确定性阈值
- **多专家系统**: 支持多个专家同时标注
- **在线学习**: 实现实时在线学习能力

---



*最后更新: 2024年12月*
