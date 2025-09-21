"""
专家接口模块
提供模拟专家和人类专家接口，用于DAgger训练中的主动学习
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
import pygame
import logging


class SimulatedExpert:
    """
    模拟专家类
    用于开发和调试，提供基于规则的简单专家策略
    """
    
    def __init__(self, goal_position: List[float], action_scale: float = 0.1):
        """
        Args:
            goal_position: 目标位置 [x, y, z]
            action_scale: 动作缩放因子
        """
        self.goal_position = np.array(goal_position)
        self.action_scale = action_scale
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"模拟专家已初始化，目标位置: {self.goal_position}")
    
    def get_label(self, current_state: Dict[str, Any]) -> torch.Tensor:
        """
        获取专家标注的动作
        
        Args:
            current_state: 当前状态字典，包含位置信息等
            
        Returns:
            专家动作张量，形状为 (7,) - [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper_angle]
        """
        # 从状态中提取当前位置
        if 'position' in current_state:
            current_pos = np.array(current_state['position'])
        elif 'geometry_features' in current_state:
            # 如果状态包含几何特征，使用前3维作为位置
            current_pos = np.array(current_state['geometry_features'][:3])
        else:
            # 默认位置
            current_pos = np.array([0.0, 0.0, 0.0])
        
        # 计算朝向目标的方向向量
        direction_to_goal = self.goal_position - current_pos
        
        # 计算距离
        distance = np.linalg.norm(direction_to_goal)
        
        # 生成专家动作
        expert_action = self._convert_direction_to_action(direction_to_goal, distance)
        
        self.logger.debug(f"[模拟专家] 当前位置: {current_pos}, 目标位置: {self.goal_position}")
        self.logger.debug(f"[模拟专家] 方向向量: {direction_to_goal}, 距离: {distance:.3f}")
        self.logger.debug(f"[模拟专家] 专家动作: {expert_action}")
        
        return expert_action
    
    def _convert_direction_to_action(self, direction: np.ndarray, distance: float) -> torch.Tensor:
        """
        将方向向量转换为动作指令
        
        Args:
            direction: 方向向量
            distance: 到目标的距离
            
        Returns:
            动作张量
        """
        # 归一化方向向量
        if distance > 1e-6:
            normalized_direction = direction / distance
        else:
            normalized_direction = np.zeros_like(direction)
        
        # 根据距离调整动作幅度
        if distance > 0.1:
            # 距离较远时，使用较大的动作幅度
            action_magnitude = min(self.action_scale, distance * 0.5)
        else:
            # 距离较近时，使用较小的动作幅度
            action_magnitude = self.action_scale * 0.1
        
        # 构建7维动作向量
        action = np.zeros(7)
        
        # 前6维：机械臂动作 [dx, dy, dz, d_roll, d_pitch, d_yaw]
        action[:3] = normalized_direction * action_magnitude  # 位置移动
        action[3:6] = np.random.normal(0, 0.01, 3)  # 小的随机旋转
        
        # 第7维：夹爪动作
        if distance < 0.05:
            # 接近目标时，关闭夹爪
            action[6] = 0.8  # 夹爪关闭
        else:
            # 远离目标时，保持夹爪开启
            action[6] = 0.2  # 夹爪开启
        
        return torch.tensor(action, dtype=torch.float32)


class HumanExpert:
    """
    人类专家类
    提供真实的人类专家标注接口
    """
    
    def __init__(self, input_method: str = "keyboard"):
        """
        Args:
            input_method: 输入方法 ("keyboard", "joystick")
        """
        self.input_method = input_method
        self.logger = logging.getLogger(__name__)
        
        if input_method == "joystick":
            self._init_joystick()
        else:
            self.logger.info("人类专家接口已初始化（键盘输入模式）")
    
    def _init_joystick(self):
        """初始化游戏手柄"""
        try:
            pygame.init()
            pygame.joystick.init()
            
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                self.logger.info(f"游戏手柄已连接: {self.joystick.get_name()}")
            else:
                self.logger.warning("未检测到游戏手柄，将使用键盘输入")
                self.input_method = "keyboard"
        except Exception as e:
            self.logger.error(f"游戏手柄初始化失败: {e}")
            self.input_method = "keyboard"
    
    def get_label(self, current_state: Dict[str, Any]) -> torch.Tensor:
        """
        获取人类专家标注的动作
        
        Args:
            current_state: 当前状态字典
            
        Returns:
            专家动作张量，形状为 (7,)
        """
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
        print("=" * 60)
        
        return action
    
    def _display_current_state(self, current_state: Dict[str, Any]):
        """显示当前状态信息"""
        print("📊 当前状态:")
        
        if 'position' in current_state:
            pos = current_state['position']
            print(f"   位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        if 'vision_features' in current_state:
            print(f"   视觉特征维度: {len(current_state['vision_features'])}")
        
        if 'tactile_features' in current_state:
            print(f"   触觉特征维度: {len(current_state['tactile_features'])}")
        
        if 'classification_logits' in current_state:
            logits = current_state['classification_logits']
            predicted_class = torch.argmax(logits).item()
            confidence = torch.softmax(logits, dim=0)[predicted_class].item()
            print(f"   预测类别: {predicted_class}, 置信度: {confidence:.3f}")
        
        print()
    
    def _get_keyboard_input(self) -> torch.Tensor:
        """通过键盘获取专家输入"""
        print("⌨️  请输入7维动作向量:")
        print("   格式: dx,dy,dz,d_roll,d_pitch,d_yaw,gripper_angle")
        print("   示例: 0.1,0.0,0.05,0.0,0.0,0.0,0.5")
        print("   说明: 前6维为机械臂动作，第7维为夹爪角度(0-1)")
        
        while True:
            try:
                action_str = input("   动作输入: ").strip()
                if not action_str:
                    print("   ❌ 输入不能为空，请重新输入")
                    continue
                
                # 解析输入
                action_values = [float(x.strip()) for x in action_str.split(',')]
                
                if len(action_values) != 7:
                    print(f"   ❌ 需要7个数值，但输入了{len(action_values)}个，请重新输入")
                    continue
                
                # 验证夹爪角度范围
                if not (0.0 <= action_values[6] <= 1.0):
                    print("   ❌ 夹爪角度必须在0-1之间，请重新输入")
                    continue
                
                return torch.tensor(action_values, dtype=torch.float32)
                
            except ValueError:
                print("   ❌ 输入格式错误，请输入7个用逗号分隔的数值")
            except KeyboardInterrupt:
                print("\n   ⚠️  用户中断，使用默认动作")
                return torch.zeros(7, dtype=torch.float32)
    
    def _get_joystick_input(self) -> torch.Tensor:
        """通过游戏手柄获取专家输入"""
        print("🎮 游戏手柄输入模式")
        print("   左摇杆: 控制x,y移动")
        print("   右摇杆: 控制z移动和旋转")
        print("   扳机: 控制夹爪")
        print("   按任意按钮确认动作")
        
        action = torch.zeros(7, dtype=torch.float32)
        
        try:
            while True:
                pygame.event.pump()
                
                # 读取摇杆输入
                left_x = self.joystick.get_axis(0)  # 左摇杆X
                left_y = self.joystick.get_axis(1)  # 左摇杆Y
                right_x = self.joystick.get_axis(2)  # 右摇杆X
                right_y = self.joystick.get_axis(3)  # 右摇杆Y
                
                # 读取扳机输入
                left_trigger = (self.joystick.get_axis(4) + 1.0) / 2.0  # 左扳机
                right_trigger = (self.joystick.get_axis(5) + 1.0) / 2.0  # 右扳机
                
                # 映射到动作
                action[0] = left_x * 0.1  # dx
                action[1] = -left_y * 0.1  # dy (注意Y轴方向)
                action[2] = right_y * 0.1  # dz
                action[3] = right_x * 0.05  # d_roll
                action[4] = 0.0  # d_pitch (暂时固定)
                action[5] = 0.0  # d_yaw (暂时固定)
                action[6] = right_trigger  # gripper_angle
                
                # 显示当前动作
                print(f"\r   当前动作: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, "
                      f"{action[3]:.3f}, {action[4]:.3f}, {action[5]:.3f}, {action[6]:.3f}]", end="")
                
                # 检查确认按钮
                for i in range(self.joystick.get_numbuttons()):
                    if self.joystick.get_button(i):
                        print("\n   ✅ 动作已确认")
                        return action
                
                # 检查退出按钮（通常是"B"按钮）
                if self.joystick.get_button(1):  # 通常是B按钮
                    print("\n   ⚠️  用户取消，使用默认动作")
                    return torch.zeros(7, dtype=torch.float32)
                
        except Exception as e:
            self.logger.error(f"游戏手柄输入错误: {e}")
            print(f"\n   ❌ 游戏手柄输入错误: {e}")
            return torch.zeros(7, dtype=torch.float32)
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'joystick'):
            pygame.quit()


class ExpertFactory:
    """
    专家工厂类
    根据配置创建相应的专家实例
    """
    
    @staticmethod
    def create_expert(config: Dict[str, Any]) -> Any:
        """
        根据配置创建专家实例
        
        Args:
            config: 配置字典，包含expert_interface相关配置
            
        Returns:
            专家实例
        """
        expert_config = config.get('expert_interface', {})
        use_simulated = expert_config.get('use_simulated_expert', True)
        
        if use_simulated:
            goal_position = expert_config.get('simulated_goal_position', [0.0, 0.0, 0.3])
            return SimulatedExpert(goal_position=goal_position)
        else:
            input_method = expert_config.get('input_method', 'keyboard')
            return HumanExpert(input_method=input_method)
