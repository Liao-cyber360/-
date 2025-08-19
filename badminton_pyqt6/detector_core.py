"""
检测算法核心模块
从原有detector.py迁移算法逻辑，去除UI依赖
"""
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
from .config import config


class LandingDetector:
    """羽毛球落地检测器"""
    
    def __init__(self, threshold=5, confirmation_frames=3, height_threshold=20.0):
        """
        初始化落地检测器
        
        参数:
            threshold: 速度阈值，低于此值视为可能落地（像素/帧）
            confirmation_frames: 连续多少帧低于阈值判定为落地
            height_threshold: 高度阈值，Z坐标低于此值视为接近地面（单位：厘米）
        """
        self.threshold = threshold
        self.confirmation_frames = confirmation_frames
        self.height_threshold = height_threshold
        self.previous_positions = deque(maxlen=10)
        self.previous_3d_positions = deque(maxlen=10)
        self.slow_frame_counter = 0
        self.last_landing_time = 0
        self.landing_cooldown = 15.0
    
    def detect_landing(self, position, timestamp, position_3d=None):
        """
        检测羽毛球是否落地
        
        参数:
            position: 当前球的2D位置 (x, y)
            timestamp: 当前时间戳
            position_3d: 当前球的3D位置 (x, y, z)，如果有的话
        
        返回:
            landing_detected: 是否检测到落地
        """
        # 冷却时间检查
        if timestamp - self.last_landing_time < self.landing_cooldown:
            return False
        
        # 添加当前位置
        self.previous_positions.append((position, timestamp))
        if position_3d is not None:
            self.previous_3d_positions.append((position_3d, timestamp))
        
        # 需要足够的历史数据
        if len(self.previous_positions) < 3:
            return False
        
        # 计算速度
        current_pos, current_time = self.previous_positions[-1]
        prev_pos, prev_time = self.previous_positions[-2]
        
        time_diff = current_time - prev_time
        if time_diff > 0:
            velocity = np.linalg.norm(np.array(current_pos) - np.array(prev_pos)) / time_diff
        else:
            velocity = 0
        
        # 检查速度是否低于阈值
        if velocity < self.threshold:
            self.slow_frame_counter += 1
        else:
            self.slow_frame_counter = 0
        
        # 3D高度检查
        height_check = True
        if position_3d is not None:
            height_check = position_3d[2] < self.height_threshold
        
        # 落地判定
        if (self.slow_frame_counter >= self.confirmation_frames and height_check):
            self.last_landing_time = timestamp
            self.slow_frame_counter = 0
            return True
        
        return False
    
    def reset(self):
        """重置检测器状态"""
        self.previous_positions.clear()
        self.previous_3d_positions.clear()
        self.slow_frame_counter = 0


class ShuttlecockDetector:
    """羽毛球检测器核心"""
    
    def __init__(self, model_path, camera_id=0, confidence_threshold=0.3):
        """
        初始化检测器
        
        参数:
            model_path: YOLO模型路径
            camera_id: 相机ID
            confidence_threshold: 置信度阈值
        """
        self.model_path = model_path
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        
        # 加载模型
        self.model = YOLO(model_path)
        
        # 轨迹数据
        self.trajectory = deque(maxlen=config.trajectory_buffer_size)
        self.trajectory_3d = deque(maxlen=config.trajectory_buffer_size)
        self.timestamps = deque(maxlen=config.trajectory_buffer_size)
        
        # 落地检测器
        self.landing_detector = LandingDetector()
        
        # 检测状态
        self.last_detection_time = 0
        self.detection_count = 0
    
    def detect(self, frame, timestamp):
        """
        检测羽毛球
        
        参数:
            frame: 输入帧
            timestamp: 时间戳
        
        返回:
            processed_frame: 处理后的帧
            position: 检测到的位置 (x, y) 或 None
            landing_detected: 是否检测到落地
        """
        processed_frame = frame.copy()
        position = None
        landing_detected = False
        
        # YOLO检测
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=0.5,
            verbose=False
        )
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                # 取置信度最高的检测结果
                best_box = None
                best_conf = 0
                
                for box in result.boxes:
                    conf = box.conf.cpu().numpy()[0]
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box
                
                if best_box is not None:
                    # 提取位置
                    xyxy = best_box.xyxy.cpu().numpy()[0]
                    x1, y1, x2, y2 = xyxy
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    position = (center_x, center_y)
                    
                    # 添加到轨迹
                    self.trajectory.append(position)
                    self.timestamps.append(timestamp)
                    
                    # 更新统计
                    self.last_detection_time = timestamp
                    self.detection_count += 1
                    
                    # 检测落地
                    landing_detected = self.landing_detector.detect_landing(
                        position, timestamp
                    )
                    
                    # 绘制检测结果
                    cv2.rectangle(processed_frame, (int(x1), int(y1)), 
                                (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(processed_frame, position, 5, (0, 255, 0), -1)
                    cv2.putText(processed_frame, f'Conf: {best_conf:.2f}',
                              (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 1)
        
        # 绘制轨迹
        self._draw_trajectory(processed_frame)
        
        return processed_frame, position, landing_detected
    
    def _draw_trajectory(self, frame):
        """绘制轨迹"""
        if len(self.trajectory) < 2:
            return
        
        # 绘制轨迹线
        points = list(self.trajectory)
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], (255, 0, 0), 2)
        
        # 绘制轨迹点
        for i, point in enumerate(points):
            alpha = (i + 1) / len(points)
            color = (int(255 * alpha), 0, int(255 * (1 - alpha)))
            cv2.circle(frame, point, 3, color, -1)
    
    def get_trajectory_data(self):
        """获取轨迹数据"""
        return {
            'trajectory': list(self.trajectory),
            'timestamps': list(self.timestamps),
            'trajectory_3d': list(self.trajectory_3d)
        }
    
    def reset_trajectory(self):
        """重置轨迹数据"""
        self.trajectory.clear()
        self.trajectory_3d.clear()
        self.timestamps.clear()
        self.landing_detector.reset()
        self.detection_count = 0


class StereoProcessor:
    """双目视觉处理器核心"""
    
    def __init__(self):
        """初始化双目处理器"""
        self.camera1_matrix = None
        self.camera1_dist = None
        self.camera2_matrix = None
        self.camera2_dist = None
        self.rotation_matrix = None
        self.translation_vector = None
        
        # 3D轨迹数据
        self.trajectory_3d = deque(maxlen=config.trajectory_buffer_size)
        self.trajectory_timestamps = deque(maxlen=config.trajectory_buffer_size)
        
        # 标定状态
        self.is_calibrated = False
    
    def load_calibration_data(self, camera1_params, camera2_params, stereo_params=None):
        """
        加载标定数据
        
        参数:
            camera1_params: 相机1内参文件路径
            camera2_params: 相机2内参文件路径
            stereo_params: 双目标定参数文件路径
        """
        # 加载相机1参数
        fs = cv2.FileStorage(camera1_params, cv2.FILE_STORAGE_READ)
        self.camera1_matrix = fs.getNode("camera_matrix").mat()
        self.camera1_dist = fs.getNode("distortion_coefficients").mat().flatten()
        fs.release()
        
        # 加载相机2参数
        fs = cv2.FileStorage(camera2_params, cv2.FILE_STORAGE_READ)
        self.camera2_matrix = fs.getNode("camera_matrix").mat()
        self.camera2_dist = fs.getNode("distortion_coefficients").mat().flatten()
        
        # 尝试加载外参
        rotation_node = fs.getNode("rotation_matrix")
        translation_node = fs.getNode("translation_vector")
        
        if not rotation_node.empty() and not translation_node.empty():
            self.rotation_matrix = rotation_node.mat()
            self.translation_vector = translation_node.mat()
            self.is_calibrated = True
        
        fs.release()
    
    def triangulate_point(self, point1, point2):
        """
        三角测量计算3D点
        
        参数:
            point1: 相机1中的2D点
            point2: 相机2中的2D点
        
        返回:
            point_3d: 3D点坐标 (x, y, z) 或 None
        """
        if not self.is_calibrated:
            return None
        
        # 构建投影矩阵
        P1 = np.hstack([self.camera1_matrix, np.zeros((3, 1))])
        
        if self.rotation_matrix is not None and self.translation_vector is not None:
            RT = np.hstack([self.rotation_matrix, self.translation_vector.reshape(-1, 1)])
            P2 = self.camera2_matrix @ RT
        else:
            P2 = np.hstack([self.camera2_matrix, np.zeros((3, 1))])
        
        # 三角测量
        point1_norm = np.array([[point1[0], point1[1]]], dtype=np.float32)
        point2_norm = np.array([[point2[0], point2[1]]], dtype=np.float32)
        
        # 去畸变
        point1_undist = cv2.undistortPoints(point1_norm, self.camera1_matrix, self.camera1_dist)
        point2_undist = cv2.undistortPoints(point2_norm, self.camera2_matrix, self.camera2_dist)
        
        # 三角测量
        points_4d = cv2.triangulatePoints(P1, P2, point1_undist.T, point2_undist.T)
        
        if points_4d[3, 0] != 0:
            point_3d = (points_4d[:3, 0] / points_4d[3, 0]).flatten()
            return point_3d
        
        return None
    
    def reconstruct_3d_trajectory(self, trajectory1, timestamps1, trajectory2, timestamps2):
        """
        重建3D轨迹
        
        参数:
            trajectory1: 相机1轨迹
            timestamps1: 相机1时间戳
            trajectory2: 相机2轨迹
            timestamps2: 相机2时间戳
        
        返回:
            trajectory_3d: 3D轨迹点列表
            trajectory_timestamps: 对应时间戳
        """
        trajectory_3d = []
        trajectory_timestamps = []
        
        if len(trajectory1) < 3 or len(trajectory2) < 3:
            return trajectory_3d, trajectory_timestamps
        
        # 时间同步和3D重建
        for i, ts1 in enumerate(timestamps1):
            # 查找最近的时间戳
            closest_idx = None
            min_diff = 0.02  # 20ms阈值
            
            for j, ts2 in enumerate(timestamps2):
                diff = abs(ts1 - ts2)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = j
            
            if closest_idx is not None:
                # 三角测量
                point_3d = self.triangulate_point(trajectory1[i], trajectory2[closest_idx])
                
                if point_3d is not None and self.is_point_valid(point_3d):
                    trajectory_3d.append(point_3d)
                    trajectory_timestamps.append(ts1)
        
        return trajectory_3d, trajectory_timestamps
    
    def is_point_valid(self, point_3d):
        """检查3D点是否有效"""
        if point_3d is None:
            return False
        
        # 基本范围检查
        x, y, z = point_3d
        
        # 检查是否在合理的场地范围内
        if not (-100 <= x <= 710):  # X方向扩展范围
            return False
        if not (-100 <= y <= 770):   # Y方向扩展范围
            return False
        if not (0 <= z <= 500):      # Z方向高度限制
            return False
        
        return True
    
    def is_point_above_court(self, point_3d):
        """检查点是否在场地上方"""
        if not self.is_point_valid(point_3d):
            return False
        
        x, y, z = point_3d
        
        # 场地边界检查（稍微扩展）
        court_margin = 50
        if not (-court_margin <= x <= 610 + court_margin):
            return False
        if not (-court_margin <= y <= 670 + court_margin):
            return False
        if z < 10:  # 最低高度限制
            return False
        
        return True
    
    def filter_trajectory_by_court(self, trajectory1, timestamps1, trajectory2, timestamps2):
        """根据场地范围过滤轨迹"""
        if len(trajectory1) < 3 or len(trajectory2) < 3:
            return trajectory1, timestamps1, trajectory2, timestamps2
        
        # 计算所有3D点
        point_pairs = []
        
        for i, ts1 in enumerate(timestamps1):
            closest_idx = None
            min_diff = 0.02
            
            for j, ts2 in enumerate(timestamps2):
                diff = abs(ts1 - ts2)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = j
            
            if closest_idx is not None:
                if i < len(trajectory1) and closest_idx < len(trajectory2):
                    point_3d = self.triangulate_point(trajectory1[i], trajectory2[closest_idx])
                    
                    if point_3d is not None:
                        point_pairs.append((i, closest_idx, point_3d))
        
        # 筛选场地上方的点
        valid_pairs = []
        for idx1, idx2, point_3d in point_pairs:
            if self.is_point_above_court(point_3d):
                valid_pairs.append((idx1, idx2))
        
        # 提取筛选后的轨迹
        if valid_pairs:
            valid_idx1 = [pair[0] for pair in valid_pairs]
            valid_idx2 = [pair[1] for pair in valid_pairs]
            
            filtered_traj1 = [trajectory1[i] for i in valid_idx1]
            filtered_ts1 = [timestamps1[i] for i in valid_idx1]
            filtered_traj2 = [trajectory2[i] for i in valid_idx2]
            filtered_ts2 = [timestamps2[i] for i in valid_idx2]
            
            return filtered_traj1, filtered_ts1, filtered_traj2, filtered_ts2
        
        return trajectory1, timestamps1, trajectory2, timestamps2
    
    def reset(self):
        """重置处理器状态"""
        self.trajectory_3d.clear()
        self.trajectory_timestamps.clear()


def create_detector(model_path, camera_id=0):
    """创建检测器实例的工厂函数"""
    return ShuttlecockDetector(model_path, camera_id)


def create_stereo_processor():
    """创建双目处理器实例的工厂函数"""
    return StereoProcessor()