# Project Poseidon v3.0

**基于纯CLIP变体的视觉-触觉融合机器人操控系统**

## 🌊 项目概述

Project Poseidon 是一个先进的机器人感知与控制系统，专门设计用于水下环境中的精细操控任务。该系统通过融合双目视觉与磁性触觉两种模态的感知信息，实现了在视觉受限、环境复杂的水下场景中自主完成检测、识别、分类和精细操控任务（如抓取、拧螺丝等）。

### 🎯 核心特性

- **多模态融合**: 视觉-触觉深度融合，提供鲁棒的感知能力
- **水下适应**: 专门针对水下环境的视觉领域适应
- **纯对比学习**: 基于CLIP思想的InfoNCE损失多模态表征学习
- **迭代学习**: DAgger算法解决分布偏移问题
- **记忆机制**: LSTM处理时序信息和历史状态
- **实时控制**: 高精度时间戳同步和实时力反馈
- **分类器训练**: 基于预训练表征的物体分类器

## 🏗️ 系统架构

项目采用"四阶段分层学习"框架：

### 阶段 0.5: 视觉领域适应 (Vision Domain Adaptation)
- 使用 ViT (Vision Transformer) 进行水下视觉领域适应
- 基于URPC等水下数据集进行分层微调
- 产出适应水下环境的视觉编码器

### 阶段 1: 多模态表征学习 (Multimodal Representation Learning)
- **视觉流**: 双目图像水平拼接后输入ViT，隐式学习深度信息
- **触觉流**: 18传感器×3轴=54维特征，100时间步序列，使用Transformer编码器
- **纯对比学习**: 基于InfoNCE损失的CLIP变体学习，无多任务学习
- **投影头**: 将768维特征投影到128维共享嵌入空间
- **表征模型**: 简化的RepresentationModel，只包含视觉和触觉编码器

### 阶段 1.5: 物体分类器训练 (Object Classifier Training)
- **预训练表征**: 使用阶段1训练好的表征模型提取特征
- **分类器网络**: 简单的MLP分类器，输入为拼接的视觉+触觉特征
- **冻结表征**: 表征模型参数冻结，只训练分类器
- **交叉熵损失**: 标准的分类损失函数

### 阶段 2: 动态策略学习 (Dynamic Policy Learning)
- **状态融合**: 语义特征 + 触觉特征 + 几何特征(3D坐标)
- **LSTM策略**: 具备记忆能力的循环神经网络
- **DAgger训练**: 迭代模仿学习，主动收集纠错数据

## 📁 项目结构

```
project_poseidon/
├── configs/                          # 配置文件
│   ├── stage0_vision_finetune.yaml   # 阶段0.5配置
│   ├── stage1_representation.yaml    # 表征学习配置  
│   └── stage2_policy.yaml            # 策略学习配置
│
├── data_loader/                      # 数据加载模块
│   ├── dataset.py                    # 数据集类
│   ├── augmentations.py              # 数据增强
│   ├── samplers.py                   # 采样器
│   └── utils.py                      # 数据处理工具
│
├── models/                           # 模型定义
│   ├── vision_encoder.py             # ViT视觉编码器
│   ├── tactile_encoder.py            # Transformer触觉编码器
│   ├── representation_model.py       # 纯CLIP变体表征模型
│   ├── classifier.py                 # 物体分类器
│   └── policy_model.py               # LSTM策略模型
│
├── engine/                           # 训练引擎
│   ├── trainer.py                    # 训练器
│   ├── evaluator.py                  # 评估器
│   └── losses.py                     # 损失函数
│
├── robot/                            # 机器人接口
│   └── interface.py                  # 硬件通信接口
│
├── main_finetune_vision_on_urpc.py   # 阶段0.5主脚本
├── main_train_representation.py      # 阶段1主脚本
├── main_train_classifier.py          # 分类器训练脚本
├── main_train_policy.py              # 阶段2主脚本
├── run_robot_demo.py                 # 机器人演示脚本
└── requirements.txt                  # 依赖包列表
```

## 🚀 快速开始

### 环境配置

1. **克隆项目**
```bash
git clone <repository-url>
cd project_poseidon
```

2. **创建虚拟环境**
```bash
python -m venv poseidon_env
source poseidon_env/bin/activate  # Linux/Mac
# 或
poseidon_env\Scripts\activate     # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **GPU支持** (推荐)
```bash
# 根据你的CUDA版本安装PyTorch
# 访问 https://pytorch.org/get-started/locally/ 获取具体命令
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 训练流程

#### 阶段 0.5: 视觉领域适应

```bash
python main_finetune_vision_on_urpc.py --config configs/stage0_vision_finetune.yaml
```

#### 阶段 1: 多模态表征学习

```bash
python main_train_representation.py --config configs/stage1_representation.yaml
```

#### 阶段 1.5: 物体分类器训练

```bash
python main_train_classifier.py --config configs/stage1_classifier.yaml
```

#### 阶段 2: 动态策略学习

```bash
python main_train_policy.py --config configs/stage2_policy.yaml
```

#### 机器人演示

```bash
python run_robot_demo.py --config configs/robot_demo.yaml --task screw_tightening
```

## 📊 数据格式

### 表征学习数据格式
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

### 策略学习数据格式
```json
{
  "trajectory_id": "traj_001",
  "episode_id": "ep_001", 
  "task_type": "screw_tightening",
  "success": true,
  "length": 150,
  "data_path": "trajectories/traj_001.json",
  "metadata": {
    "expert": "human",
    "difficulty": "medium"
  }
}
```

## 🔧 配置说明

### 主要配置参数

- **数据参数**: 数据路径、批次大小、数据增强设置
- **模型参数**: 网络架构、隐藏层维度、注意力头数等
- **训练参数**: 学习率、优化器、调度器、混合精度等
- **DAgger参数**: 迭代次数、专家数据比例、episode数量等
- **机器人参数**: 控制频率、力限制、安全参数等

详细配置说明请参考 `configs/` 目录下的YAML文件。

## 📈 性能指标

### 表征学习指标
- **对比损失**: InfoNCE loss
- **检索准确率**: Recall@1, Recall@5
- **对比准确率**: 视觉-触觉匹配精度

### 分类器指标
- **分类准确率**: Top-1, Top-5准确率
- **交叉熵损失**: 分类损失
- **混淆矩阵**: 各类别分类性能

### 策略学习指标  
- **动作精度**: MSE, MAE
- **轨迹平滑度**: 动作变化率
- **任务成功率**: 完成任务的比例
- **安全性**: 力限制遵守率

## 🤖 硬件要求

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
- **通信接口**: ROS/串口通信

## 🔬 技术创新点

1. **水下环境特化**: 首个专门针对水下环境的视触融合系统
2. **分层学习架构**: 四阶段渐进式学习，可解释性强
3. **纯CLIP变体**: 简化的对比学习架构，专注于视觉-触觉对齐
4. **分类器训练**: 基于预训练表征的物体分类器
5. **DAgger迭代学习**: 主动学习解决分布偏移问题
6. **实时同步系统**: 高精度多传感器时间同步

## 📚 相关论文

- **CLIP**: Learning Transferable Visual Models From Natural Language Supervision
- **DAgger**: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning
- **ViT**: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- **InfoNCE**: Representation Learning with Contrastive Predictive Coding

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- OpenAI CLIP 团队提供的对比学习框架
- Hugging Face Transformers 社区
- PyTorch 深度学习框架
- 水下机器人研究社区的宝贵经验分享

## 📞 联系方式

- **项目维护者**: [Your Name]
- **邮箱**: [your.email@example.com]
- **项目主页**: [Project URL]

---

**Project Poseidon** - 让机器人在水下世界中自由操控 🌊🤖
