"""
Robot Interface for Project Poseidon
机器人硬件通信与控制接口，负责同步触发和高精度时间戳
"""

import time
import threading
import queue
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import deque
import json
import os

# 尝试导入ROS相关包（如果可用）
try:
    import rospy
    from sensor_msgs.msg import Image, PointCloud2
    from geometry_msgs.msg import Twist, Pose
    from std_msgs.msg import Header, Float64MultiArray
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Warning: ROS not available. Some functionality may be limited.")


@dataclass
class SensorData:
    """传感器数据结构"""
    timestamp: float
    data: np.ndarray
    sensor_id: str
    data_type: str  # 'vision', 'tactile', 'force', etc.
    metadata: Dict[str, Any] = None


@dataclass
class RobotState:
    """机器人状态结构"""
    timestamp: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    end_effector_pose: np.ndarray
    force_torque: np.ndarray
    metadata: Dict[str, Any] = None


@dataclass
class RobotCommand:
    """机器人命令结构"""
    timestamp: float
    command_type: str  # 'position', 'velocity', 'force'
    target_values: np.ndarray
    duration: float = 0.0
    metadata: Dict[str, Any] = None


class SensorInterface(ABC):
    """传感器接口基类"""
    
    def __init__(self, sensor_id: str, frequency: float = 30.0):
        self.sensor_id = sensor_id
        self.frequency = frequency
        self.is_running = False
        self.data_queue = queue.Queue(maxsize=1000)
        self.thread = None
        self.logger = logging.getLogger(f"Sensor_{sensor_id}")
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化传感器"""
        pass
    
    @abstractmethod
    def read_data(self) -> Optional[SensorData]:
        """读取传感器数据"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """清理资源"""
        pass
    
    def start_streaming(self) -> None:
        """开始数据流"""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._streaming_loop)
        self.thread.daemon = True
        self.thread.start()
        self.logger.info(f"Started streaming for sensor {self.sensor_id}")
    
    def stop_streaming(self) -> None:
        """停止数据流"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.logger.info(f"Stopped streaming for sensor {self.sensor_id}")
    
    def _streaming_loop(self) -> None:
        """数据流循环"""
        interval = 1.0 / self.frequency
        
        while self.is_running:
            start_time = time.time()
            
            try:
                data = self.read_data()
                if data:
                    if not self.data_queue.full():
                        self.data_queue.put(data)
                    else:
                        # 如果队列满了，移除最旧的数据
                        try:
                            self.data_queue.get_nowait()
                            self.data_queue.put(data)
                        except queue.Empty:
                            pass
            except Exception as e:
                self.logger.error(f"Error reading sensor data: {e}")
            
            # 控制频率
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_latest_data(self) -> Optional[SensorData]:
        """获取最新数据"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None


class VisionSensor(SensorInterface):
    """视觉传感器接口"""
    
    def __init__(self, sensor_id: str, camera_index: int = 0, frequency: float = 30.0):
        super().__init__(sensor_id, frequency)
        self.camera_index = camera_index
        self.camera = None
    
    def initialize(self) -> bool:
        """初始化摄像头"""
        try:
            # 尝试使用OpenCV
            import cv2
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                self.logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # 设置摄像头参数
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, self.frequency)
            
            self.logger.info(f"Vision sensor {self.sensor_id} initialized")
            return True
            
        except ImportError:
            self.logger.error("OpenCV not available")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize vision sensor: {e}")
            return False
    
    def read_data(self) -> Optional[SensorData]:
        """读取图像数据"""
        if not self.camera:
            return None
        
        try:
            import cv2
            ret, frame = self.camera.read()
            if ret:
                timestamp = time.time()
                return SensorData(
                    timestamp=timestamp,
                    data=frame,
                    sensor_id=self.sensor_id,
                    data_type='vision',
                    metadata={'shape': frame.shape}
                )
        except Exception as e:
            self.logger.error(f"Error reading vision data: {e}")
        
        return None
    
    def cleanup(self) -> None:
        """清理摄像头资源"""
        if self.camera:
            self.camera.release()
            self.camera = None


class TactileSensor(SensorInterface):
    """触觉传感器接口"""
    
    def __init__(self, sensor_id: str, num_sensors: int = 18, frequency: float = 1000.0):
        super().__init__(sensor_id, frequency)
        self.num_sensors = num_sensors
        self.feature_dim = num_sensors * 3  # 每个传感器3轴数据
        self.mock_mode = True  # 模拟模式，实际使用时需要连接真实硬件
    
    def initialize(self) -> bool:
        """初始化触觉传感器"""
        if self.mock_mode:
            self.logger.info(f"Tactile sensor {self.sensor_id} initialized in mock mode")
            return True
        
        # 实际硬件初始化代码
        # 这里需要根据具体的触觉传感器硬件进行实现
        try:
            # 示例：初始化串口通信或其他硬件接口
            self.logger.info(f"Tactile sensor {self.sensor_id} initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize tactile sensor: {e}")
            return False
    
    def read_data(self) -> Optional[SensorData]:
        """读取触觉数据"""
        timestamp = time.time()
        
        if self.mock_mode:
            # 生成模拟数据
            data = np.random.normal(0, 0.1, (self.feature_dim,))
            # 添加一些模拟的接触信号
            if np.random.random() < 0.1:  # 10%概率有接触
                contact_indices = np.random.choice(self.feature_dim, size=3, replace=False)
                data[contact_indices] += np.random.uniform(0.5, 2.0, size=3)
        else:
            # 实际硬件数据读取
            # 这里需要根据具体硬件实现
            data = self._read_hardware_data()
        
        return SensorData(
            timestamp=timestamp,
            data=data,
            sensor_id=self.sensor_id,
            data_type='tactile',
            metadata={'num_sensors': self.num_sensors, 'feature_dim': self.feature_dim}
        )
    
    def _read_hardware_data(self) -> np.ndarray:
        """读取硬件数据（需要根据具体硬件实现）"""
        # 占位符实现
        return np.zeros(self.feature_dim)
    
    def cleanup(self) -> None:
        """清理触觉传感器资源"""
        if not self.mock_mode:
            # 清理硬件资源
            pass


class DataSynchronizer:
    """数据同步器，确保多传感器数据的时间同步"""
    
    def __init__(self, tolerance: float = 1e-3):
        """
        Args:
            tolerance: 时间同步容差（秒）
        """
        self.tolerance = tolerance
        self.sensor_buffers = {}
        self.sync_lock = threading.Lock()
        self.logger = logging.getLogger("DataSynchronizer")
    
    def add_sensor(self, sensor_id: str, buffer_size: int = 100) -> None:
        """添加传感器到同步器"""
        with self.sync_lock:
            self.sensor_buffers[sensor_id] = deque(maxlen=buffer_size)
        self.logger.info(f"Added sensor {sensor_id} to synchronizer")
    
    def add_data(self, sensor_id: str, data: SensorData) -> None:
        """添加传感器数据"""
        if sensor_id not in self.sensor_buffers:
            self.add_sensor(sensor_id)
        
        with self.sync_lock:
            self.sensor_buffers[sensor_id].append(data)
    
    def get_synchronized_data(self, target_timestamp: Optional[float] = None) -> Optional[Dict[str, SensorData]]:
        """
        获取同步的传感器数据
        
        Args:
            target_timestamp: 目标时间戳，如果为None则使用最新的可同步时间戳
        
        Returns:
            同步的传感器数据字典，如果无法同步则返回None
        """
        with self.sync_lock:
            if not self.sensor_buffers:
                return None
            
            # 如果没有指定目标时间戳，找到最新的可同步时间戳
            if target_timestamp is None:
                target_timestamp = self._find_sync_timestamp()
                if target_timestamp is None:
                    return None
            
            synchronized_data = {}
            
            for sensor_id, buffer in self.sensor_buffers.items():
                closest_data = self._find_closest_data(buffer, target_timestamp)
                if closest_data is None:
                    return None  # 无法为所有传感器找到同步数据
                
                # 检查时间差是否在容差范围内
                time_diff = abs(closest_data.timestamp - target_timestamp)
                if time_diff > self.tolerance:
                    return None
                
                synchronized_data[sensor_id] = closest_data
            
            return synchronized_data
    
    def _find_sync_timestamp(self) -> Optional[float]:
        """找到最新的可同步时间戳"""
        if not self.sensor_buffers:
            return None
        
        # 找到所有传感器中最早的最新时间戳
        latest_timestamps = []
        for buffer in self.sensor_buffers.values():
            if buffer:
                latest_timestamps.append(buffer[-1].timestamp)
        
        if not latest_timestamps:
            return None
        
        return min(latest_timestamps)
    
    def _find_closest_data(self, buffer: deque, target_timestamp: float) -> Optional[SensorData]:
        """在缓冲区中找到最接近目标时间戳的数据"""
        if not buffer:
            return None
        
        closest_data = None
        min_diff = float('inf')
        
        for data in buffer:
            time_diff = abs(data.timestamp - target_timestamp)
            if time_diff < min_diff:
                min_diff = time_diff
                closest_data = data
        
        return closest_data


class RobotInterface:
    """机器人接口主类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 机器人配置字典
        """
        self.config = config
        self.logger = logging.getLogger("RobotInterface")
        
        # 传感器管理
        self.sensors = {}
        self.synchronizer = DataSynchronizer(
            tolerance=config.get('sync_tolerance', 1e-3)
        )
        
        # 机器人状态
        self.current_state = None
        self.is_connected = False
        self.is_emergency_stop = False
        
        # 控制参数
        self.control_frequency = config.get('control_frequency', 100)
        self.force_limit = config.get('force_limit', 50.0)
        self.velocity_limit = config.get('velocity_limit', 0.1)
        
        # ROS相关
        self.use_ros = config.get('use_ros', False) and ROS_AVAILABLE
        if self.use_ros:
            self._init_ros()
        
        # 数据记录
        self.data_recording = config.get('data_recording', False)
        self.recorded_data = []
    
    def _init_ros(self) -> None:
        """初始化ROS接口"""
        if not ROS_AVAILABLE:
            self.logger.warning("ROS not available, disabling ROS interface")
            self.use_ros = False
            return
        
        try:
            rospy.init_node('poseidon_robot_interface', anonymous=True)
            
            # 发布器和订阅器的初始化
            # 这里需要根据具体的机器人ROS接口进行配置
            self.logger.info("ROS interface initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ROS: {e}")
            self.use_ros = False
    
    def add_sensor(self, sensor: SensorInterface) -> None:
        """添加传感器"""
        if sensor.initialize():
            self.sensors[sensor.sensor_id] = sensor
            self.synchronizer.add_sensor(sensor.sensor_id)
            self.logger.info(f"Added sensor: {sensor.sensor_id}")
        else:
            self.logger.error(f"Failed to initialize sensor: {sensor.sensor_id}")
    
    def connect(self) -> bool:
        """连接机器人"""
        try:
            # 启动所有传感器
            for sensor in self.sensors.values():
                sensor.start_streaming()
            
            # 机器人连接逻辑
            # 这里需要根据具体的机器人进行实现
            self.is_connected = True
            self.logger.info("Robot connected successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect robot: {e}")
            return False
    
    def disconnect(self) -> None:
        """断开机器人连接"""
        try:
            # 停止所有传感器
            for sensor in self.sensors.values():
                sensor.stop_streaming()
                sensor.cleanup()
            
            # 机器人断开连接逻辑
            self.is_connected = False
            self.logger.info("Robot disconnected")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
    
    def get_synchronized_sensor_data(self) -> Optional[Dict[str, SensorData]]:
        """获取同步的传感器数据"""
        # 收集所有传感器的最新数据
        for sensor_id, sensor in self.sensors.items():
            latest_data = sensor.get_latest_data()
            if latest_data:
                self.synchronizer.add_data(sensor_id, latest_data)
        
        # 获取同步数据
        return self.synchronizer.get_synchronized_data()
    
    def execute_command(self, command: RobotCommand) -> bool:
        """执行机器人命令"""
        if not self.is_connected:
            self.logger.error("Robot not connected")
            return False
        
        if self.is_emergency_stop:
            self.logger.error("Emergency stop active")
            return False
        
        try:
            # 安全检查
            if not self._safety_check(command):
                self.logger.error("Safety check failed")
                return False
            
            # 执行命令
            success = self._execute_command_impl(command)
            
            if self.data_recording:
                self.recorded_data.append({
                    'timestamp': command.timestamp,
                    'command': command,
                    'success': success
                })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            return False
    
    def _safety_check(self, command: RobotCommand) -> bool:
        """安全检查"""
        # 力限制检查
        if command.command_type == 'force':
            max_force = np.max(np.abs(command.target_values))
            if max_force > self.force_limit:
                self.logger.warning(f"Force limit exceeded: {max_force} > {self.force_limit}")
                return False
        
        # 速度限制检查
        if command.command_type == 'velocity':
            max_velocity = np.max(np.abs(command.target_values))
            if max_velocity > self.velocity_limit:
                self.logger.warning(f"Velocity limit exceeded: {max_velocity} > {self.velocity_limit}")
                return False
        
        return True
    
    def _execute_command_impl(self, command: RobotCommand) -> bool:
        """执行命令的具体实现"""
        # 这里需要根据具体的机器人接口进行实现
        # 例如：发送ROS消息、调用机器人API等
        
        if self.use_ros:
            return self._execute_ros_command(command)
        else:
            return self._execute_direct_command(command)
    
    def _execute_ros_command(self, command: RobotCommand) -> bool:
        """通过ROS执行命令"""
        # ROS命令执行的具体实现
        self.logger.info(f"Executing ROS command: {command.command_type}")
        return True
    
    def _execute_direct_command(self, command: RobotCommand) -> bool:
        """直接执行命令"""
        # 直接命令执行的具体实现
        self.logger.info(f"Executing direct command: {command.command_type}")
        return True
    
    def emergency_stop(self) -> None:
        """紧急停止"""
        self.is_emergency_stop = True
        self.logger.warning("Emergency stop activated")
        
        # 发送停止命令到机器人
        stop_command = RobotCommand(
            timestamp=time.time(),
            command_type='stop',
            target_values=np.zeros(6)
        )
        self._execute_command_impl(stop_command)
    
    def reset_emergency_stop(self) -> None:
        """重置紧急停止"""
        self.is_emergency_stop = False
        self.logger.info("Emergency stop reset")
    
    def get_robot_state(self) -> Optional[RobotState]:
        """获取机器人状态"""
        # 这里需要根据具体的机器人接口获取状态
        return self.current_state
    
    def start_data_recording(self) -> None:
        """开始数据记录"""
        self.data_recording = True
        self.recorded_data = []
        self.logger.info("Data recording started")
    
    def stop_data_recording(self) -> None:
        """停止数据记录"""
        self.data_recording = False
        self.logger.info("Data recording stopped")
    
    def save_recorded_data(self, filepath: str) -> None:
        """保存记录的数据"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.recorded_data, f, indent=2, default=str)
            self.logger.info(f"Recorded data saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save recorded data: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()
