"""
最大速度检测模块
分析羽毛球落地前的最大速度
"""
import numpy as np
import time
from typing import List, Tuple, Optional
from PyQt6.QtCore import QObject, pyqtSignal


class SpeedDetector(QObject):
    """速度检测器"""
    
    # 信号定义
    max_speed_detected = pyqtSignal(float, float, np.ndarray)  # (max_speed, time_to_landing, position)
    speed_analysis_complete = pyqtSignal(dict)  # 完整的速度分析结果
    
    def __init__(self):
        super().__init__()
        
        # 速度计算参数
        self.min_points_for_analysis = 5  # 最少需要5个点进行分析
        self.smoothing_window = 3  # 平滑窗口
        
        # 速度数据
        self.speeds = []
        self.timestamps = []
        self.positions = []
        
    def reset(self):
        """重置检测器"""
        self.speeds.clear()
        self.timestamps.clear()
        self.positions.clear()
    
    def calculate_velocity(self, positions: List[np.ndarray], timestamps: List[float]) -> List[float]:
        """计算速度序列"""
        if len(positions) < 2 or len(timestamps) < 2:
            return []
        
        velocities = []
        for i in range(1, len(positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                dx = positions[i] - positions[i-1]
                velocity = np.linalg.norm(dx) / dt  # cm/s
                velocities.append(velocity)
            else:
                velocities.append(0.0)
        
        return velocities
    
    def smooth_speeds(self, speeds: List[float]) -> List[float]:
        """平滑速度数据"""
        if len(speeds) < self.smoothing_window:
            return speeds
        
        smoothed = []
        half_window = self.smoothing_window // 2
        
        for i in range(len(speeds)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(speeds), i + half_window + 1)
            window_speeds = speeds[start_idx:end_idx]
            smoothed.append(sum(window_speeds) / len(window_speeds))
        
        return smoothed
    
    def detect_max_speed_before_landing(self, 
                                      trajectory_3d: List[np.ndarray], 
                                      timestamps: List[float]) -> Optional[dict]:
        """检测落地前最大速度"""
        if len(trajectory_3d) < self.min_points_for_analysis:
            return None
        
        try:
            # 计算速度
            speeds = self.calculate_velocity(trajectory_3d, timestamps)
            if not speeds:
                return None
            
            # 平滑处理
            smoothed_speeds = self.smooth_speeds(speeds)
            
            # 找到最大速度及其位置
            max_speed_idx = np.argmax(smoothed_speeds)
            max_speed = smoothed_speeds[max_speed_idx]
            max_speed_position = trajectory_3d[max_speed_idx + 1]  # 速度对应的位置索引+1
            max_speed_time = timestamps[max_speed_idx + 1]
            
            # 计算距离落地的时间（假设最后一个点是落地点）
            landing_time = timestamps[-1]
            time_to_landing = landing_time - max_speed_time
            
            # 分析速度变化
            speed_analysis = self.analyze_speed_pattern(smoothed_speeds, timestamps[1:])
            
            result = {
                'max_speed': max_speed,  # cm/s
                'max_speed_kmh': max_speed * 0.036,  # km/h
                'max_speed_position': max_speed_position,
                'max_speed_time': max_speed_time,
                'time_to_landing': time_to_landing,
                'speed_pattern': speed_analysis,
                'all_speeds': smoothed_speeds,
                'speed_timestamps': timestamps[1:],
                'trajectory_length': len(trajectory_3d)
            }
            
            # 发送信号
            self.max_speed_detected.emit(max_speed, time_to_landing, max_speed_position)
            self.speed_analysis_complete.emit(result)
            
            return result
            
        except Exception as e:
            print(f"Speed detection error: {e}")
            return None
    
    def analyze_speed_pattern(self, speeds: List[float], timestamps: List[float]) -> dict:
        """分析速度模式"""
        if len(speeds) < 3:
            return {}
        
        speeds_array = np.array(speeds)
        
        # 计算速度统计
        mean_speed = np.mean(speeds_array)
        std_speed = np.std(speeds_array)
        min_speed = np.min(speeds_array)
        max_speed = np.max(speeds_array)
        
        # 检测加速和减速阶段
        accelerations = []
        for i in range(1, len(speeds)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                acceleration = (speeds[i] - speeds[i-1]) / dt
                accelerations.append(acceleration)
        
        # 找到主要的加速/减速阶段
        acceleration_phases = self.detect_acceleration_phases(accelerations)
        
        return {
            'mean_speed': mean_speed,
            'std_speed': std_speed,
            'min_speed': min_speed,
            'max_speed': max_speed,
            'speed_range': max_speed - min_speed,
            'acceleration_phases': acceleration_phases,
            'total_accelerations': len([a for a in accelerations if a > 0]),
            'total_decelerations': len([a for a in accelerations if a < 0])
        }
    
    def detect_acceleration_phases(self, accelerations: List[float]) -> List[dict]:
        """检测加速/减速阶段"""
        if not accelerations:
            return []
        
        phases = []
        current_phase = None
        
        for i, acc in enumerate(accelerations):
            phase_type = 'acceleration' if acc > 0 else 'deceleration' if acc < 0 else 'constant'
            
            if current_phase is None or current_phase['type'] != phase_type:
                # 开始新阶段
                if current_phase is not None:
                    phases.append(current_phase)
                
                current_phase = {
                    'type': phase_type,
                    'start_idx': i,
                    'accelerations': [acc],
                    'duration': 1
                }
            else:
                # 继续当前阶段
                current_phase['accelerations'].append(acc)
                current_phase['duration'] += 1
        
        # 添加最后一个阶段
        if current_phase is not None:
            phases.append(current_phase)
        
        # 计算每个阶段的统计信息
        for phase in phases:
            accs = phase['accelerations']
            phase['mean_acceleration'] = np.mean(accs)
            phase['max_acceleration'] = np.max(np.abs(accs))
        
        return phases
    
    def get_speed_summary(self) -> str:
        """获取速度检测摘要"""
        if not self.speeds:
            return "No speed data available"
        
        max_speed = max(self.speeds)
        avg_speed = sum(self.speeds) / len(self.speeds)
        
        return f"Max Speed: {max_speed:.1f} cm/s ({max_speed*0.036:.1f} km/h)\n" \
               f"Avg Speed: {avg_speed:.1f} cm/s ({avg_speed*0.036:.1f} km/h)\n" \
               f"Data Points: {len(self.speeds)}"