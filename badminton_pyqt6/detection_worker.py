"""
检测处理线程
负责后台YOLO检测和3D重建
"""
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QTimer
from collections import deque
from detector_core import ShuttlecockDetector, StereoProcessor
from config import config


class DetectionWorker(QThread):
    """检测处理工作线程"""
    
    # 信号定义
    detection_result = pyqtSignal(int, dict)  # (camera_id, result)
    trajectory_updated = pyqtSignal(int, dict)  # (camera_id, trajectory_data)
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    performance_stats = pyqtSignal(dict)  # 性能统计
    
    def __init__(self, camera_id=0, model_path=None):
        super().__init__()
        
        self.camera_id = camera_id
        self.model_path = model_path
        
        # 检测器
        self.detector = None
        
        # 处理队列
        self.frame_queue = deque(maxlen=10)
        self.processing = False
        
        # 线程同步
        self.mutex = QMutex()
        
        # 性能统计
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = 0
        self.last_fps_update = 0
        self.processing_times = deque(maxlen=30)
        
        # 配置参数
        self.confidence_threshold = 0.3
        self.iou_threshold = 0.5
        
        # 状态更新定时器
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.emit_performance_stats)
        self.stats_timer.start(1000)  # 每秒更新
    
    def initialize_detector(self):
        """初始化检测器"""
        try:
            if self.model_path:
                self.detector = ShuttlecockDetector(
                    self.model_path,
                    camera_id=self.camera_id,
                    confidence_threshold=self.confidence_threshold
                )
                self.status_changed.emit(f"Detector {self.camera_id} initialized")
                return True
            else:
                self.error_occurred.emit(f"No model path specified for detector {self.camera_id}")
                return False
                
        except Exception as e:
            self.error_occurred.emit(f"Detector {self.camera_id} initialization error: {str(e)}")
            return False
    
    def add_frame(self, frame, timestamp):
        """添加帧到处理队列"""
        self.mutex.lock()
        self.frame_queue.append((frame, timestamp))
        self.mutex.unlock()
    
    def update_parameters(self, parameters):
        """更新检测参数"""
        if 'confidence_threshold' in parameters:
            self.confidence_threshold = parameters['confidence_threshold']
            if self.detector:
                self.detector.confidence_threshold = self.confidence_threshold
        
        if 'iou_threshold' in parameters:
            self.iou_threshold = parameters['iou_threshold']
        
        if 'buffer_size' in parameters:
            buffer_size = parameters['buffer_size']
            if self.detector:
                self.detector.trajectory = deque(maxlen=buffer_size)
                self.detector.timestamps = deque(maxlen=buffer_size)
    
    def reset_trajectory(self):
        """重置轨迹"""
        if self.detector:
            self.detector.reset_trajectory()
    
    def run(self):
        """主线程循环"""
        self.processing = True
        self.start_time = time.time()
        
        # 初始化检测器
        if not self.initialize_detector():
            return
        
        try:
            while self.processing:
                self.mutex.lock()
                
                # 检查是否有待处理的帧
                if self.frame_queue:
                    frame, timestamp = self.frame_queue.popleft()
                    self.mutex.unlock()
                    
                    # 处理帧
                    self.process_frame(frame, timestamp)
                    
                else:
                    self.mutex.unlock()
                    # 没有帧时短暂休眠
                    self.msleep(10)
        
        except Exception as e:
            self.error_occurred.emit(f"Detection worker {self.camera_id} error: {str(e)}")
        
        finally:
            self.status_changed.emit(f"Detector {self.camera_id} stopped")
    
    def process_frame(self, frame, timestamp):
        """处理单帧"""
        start_time = time.time()
        
        try:
            # 检测羽毛球
            processed_frame, position, landing_detected = self.detector.detect(frame, timestamp)
            
            # 记录处理时间
            processing_time = (time.time() - start_time) * 1000  # 转换为毫秒
            self.processing_times.append(processing_time)
            
            # 更新统计
            self.frame_count += 1
            if position is not None:
                self.detection_count += 1
            
            # 创建结果
            result = {
                'frame': processed_frame,
                'position': position,
                'landing_detected': landing_detected,
                'timestamp': timestamp,
                'processing_time': processing_time,
                'confidence': 0.0  # 可以从检测器获取
            }
            
            # 发送检测结果
            self.detection_result.emit(self.camera_id, result)
            
            # 发送轨迹更新
            trajectory_data = self.detector.get_trajectory_data()
            self.trajectory_updated.emit(self.camera_id, trajectory_data)
            
        except Exception as e:
            self.error_occurred.emit(f"Frame processing error: {str(e)}")
    
    def emit_performance_stats(self):
        """发送性能统计"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed > 0:
            fps = self.frame_count / elapsed
            detection_rate = self.detection_count / max(1, self.frame_count)
            
            avg_processing_time = 0
            if self.processing_times:
                avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            
            stats = {
                'camera_id': self.camera_id,
                'fps': fps,
                'detection_rate': detection_rate,
                'frame_count': self.frame_count,
                'detection_count': self.detection_count,
                'avg_processing_time': avg_processing_time,
                'queue_size': len(self.frame_queue)
            }
            
            self.performance_stats.emit(stats)
    
    def stop(self):
        """停止处理"""
        self.processing = False
        self.stats_timer.stop()


class StereoDetectionWorker(QThread):
    """双目检测和3D重建工作线程"""
    
    # 信号定义
    stereo_result = pyqtSignal(dict)  # 双目处理结果
    trajectory_3d_updated = pyqtSignal(dict)  # 3D轨迹更新
    calibration_status = pyqtSignal(bool, str)  # 标定状态
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    performance_stats = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        
        # 双目处理器
        self.stereo_processor = StereoProcessor()
        
        # 检测结果缓存
        self.detection_buffer1 = deque(maxlen=30)
        self.detection_buffer2 = deque(maxlen=30)
        
        # 处理状态
        self.processing = False
        self.calibrated = False
        
        # 线程同步
        self.mutex = QMutex()
        
        # 3D轨迹质量评估
        self.trajectory_quality = 0.0
        self.last_3d_points = deque(maxlen=10)
        
        # 性能统计
        self.processed_pairs = 0
        self.successful_reconstructions = 0
        self.start_time = 0
        
        # 定时器
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.emit_performance_stats)
        self.stats_timer.start(1000)
    
    def load_calibration(self, camera1_params, camera2_params, stereo_params=None):
        """加载标定数据"""
        try:
            self.stereo_processor.load_calibration_data(
                camera1_params, camera2_params, stereo_params
            )
            self.calibrated = self.stereo_processor.is_calibrated
            
            if self.calibrated:
                self.calibration_status.emit(True, "Stereo calibration loaded successfully")
                self.status_changed.emit("Stereo system calibrated")
            else:
                self.calibration_status.emit(False, "Stereo calibration incomplete")
                self.status_changed.emit("Stereo system not calibrated")
            
            return self.calibrated
            
        except Exception as e:
            error_msg = f"Calibration loading error: {str(e)}"
            self.error_occurred.emit(error_msg)
            self.calibration_status.emit(False, error_msg)
            return False
    
    def add_detection_result(self, camera_id, result):
        """添加检测结果"""
        self.mutex.lock()
        
        if camera_id == 0:
            self.detection_buffer1.append(result)
        elif camera_id == 1:
            self.detection_buffer2.append(result)
        
        self.mutex.unlock()
        
        # 尝试进行3D重建
        self.try_stereo_matching()
    
    def try_stereo_matching(self):
        """尝试双目匹配和3D重建"""
        if not self.calibrated:
            return
        
        self.mutex.lock()
        
        # 检查是否有足够的检测结果
        if len(self.detection_buffer1) == 0 or len(self.detection_buffer2) == 0:
            self.mutex.unlock()
            return
        
        # 寻找时间戳匹配的检测结果
        matched_pairs = self.find_temporal_matches()
        
        self.mutex.unlock()
        
        if matched_pairs:
            self.process_matched_pairs(matched_pairs)
    
    def find_temporal_matches(self):
        """查找时间匹配的检测对"""
        matches = []
        time_threshold = 0.05  # 50ms阈值
        
        for result1 in list(self.detection_buffer1):
            if result1['position'] is None:
                continue
            
            best_match = None
            min_time_diff = float('inf')
            
            for result2 in list(self.detection_buffer2):
                if result2['position'] is None:
                    continue
                
                time_diff = abs(result1['timestamp'] - result2['timestamp'])
                
                if time_diff < time_threshold and time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_match = result2
            
            if best_match:
                matches.append((result1, best_match, min_time_diff))
        
        return matches
    
    def process_matched_pairs(self, matched_pairs):
        """处理匹配的检测对"""
        try:
            for result1, result2, time_diff in matched_pairs:
                self.processed_pairs += 1
                
                # 3D重建
                point_3d = self.stereo_processor.triangulate_point(
                    result1['position'], result2['position']
                )
                
                if point_3d is not None and self.stereo_processor.is_point_valid(point_3d):
                    self.successful_reconstructions += 1
                    
                    # 添加到3D轨迹
                    self.stereo_processor.trajectory_3d.append(point_3d)
                    avg_timestamp = (result1['timestamp'] + result2['timestamp']) / 2
                    self.stereo_processor.trajectory_timestamps.append(avg_timestamp)
                    
                    # 更新质量评估
                    self.update_trajectory_quality(point_3d)
                    
                    # 创建双目结果
                    stereo_result = {
                        'point_3d': point_3d,
                        'timestamp': avg_timestamp,
                        'time_sync_diff': time_diff,
                        'position1': result1['position'],
                        'position2': result2['position'],
                        'quality': self.trajectory_quality
                    }
                    
                    self.stereo_result.emit(stereo_result)
                
                # 移除已处理的结果
                self.remove_processed_results(result1, result2)
            
            # 发送3D轨迹更新
            self.emit_3d_trajectory_update()
            
        except Exception as e:
            self.error_occurred.emit(f"Stereo processing error: {str(e)}")
    
    def update_trajectory_quality(self, point_3d):
        """更新轨迹质量评估"""
        self.last_3d_points.append(point_3d)
        
        if len(self.last_3d_points) < 3:
            self.trajectory_quality = 0.5
            return
        
        # 计算轨迹平滑度
        smoothness = self.calculate_trajectory_smoothness()
        
        # 计算高度一致性
        height_consistency = self.calculate_height_consistency()
        
        # 计算场地位置合理性
        court_validity = self.calculate_court_validity()
        
        # 综合质量评分
        self.trajectory_quality = (smoothness * 0.4 + 
                                 height_consistency * 0.3 + 
                                 court_validity * 0.3)
        
        self.trajectory_quality = max(0.0, min(1.0, self.trajectory_quality))
    
    def calculate_trajectory_smoothness(self):
        """计算轨迹平滑度"""
        if len(self.last_3d_points) < 3:
            return 0.5
        
        # 计算二阶差分（加速度变化）
        points = np.array(list(self.last_3d_points))
        if len(points) < 3:
            return 0.5
        
        # 一阶差分（速度）
        velocities = np.diff(points, axis=0)
        
        if len(velocities) < 2:
            return 0.7
        
        # 二阶差分（加速度）
        accelerations = np.diff(velocities, axis=0)
        
        # 计算加速度变化的方差（越小越平滑）
        acc_variance = np.var(accelerations)
        
        # 转换为0-1分数（方差越小分数越高）
        smoothness = 1.0 / (1.0 + acc_variance * 100)
        
        return smoothness
    
    def calculate_height_consistency(self):
        """计算高度一致性"""
        if len(self.last_3d_points) < 2:
            return 0.5
        
        heights = [p[2] for p in self.last_3d_points]
        
        # 检查高度是否在合理范围内
        valid_heights = [h for h in heights if 10 <= h <= 500]
        
        if len(valid_heights) == 0:
            return 0.0
        
        # 高度变化的合理性
        height_changes = np.diff(heights)
        reasonable_changes = [abs(change) <= 100 for change in height_changes]
        
        consistency = sum(reasonable_changes) / max(1, len(reasonable_changes))
        
        return consistency
    
    def calculate_court_validity(self):
        """计算场地位置合理性"""
        if len(self.last_3d_points) == 0:
            return 0.5
        
        valid_count = 0
        
        for point in self.last_3d_points:
            if self.stereo_processor.is_point_above_court(point):
                valid_count += 1
        
        validity = valid_count / len(self.last_3d_points)
        
        return validity
    
    def remove_processed_results(self, result1, result2):
        """移除已处理的结果"""
        try:
            if result1 in self.detection_buffer1:
                self.detection_buffer1.remove(result1)
            if result2 in self.detection_buffer2:
                self.detection_buffer2.remove(result2)
        except ValueError:
            pass  # 结果可能已被其他线程移除
    
    def emit_3d_trajectory_update(self):
        """发送3D轨迹更新"""
        trajectory_data = {
            'trajectory_3d': list(self.stereo_processor.trajectory_3d),
            'timestamps': list(self.stereo_processor.trajectory_timestamps),
            'quality': self.trajectory_quality,
            'point_count': len(self.stereo_processor.trajectory_3d)
        }
        
        self.trajectory_3d_updated.emit(trajectory_data)
    
    def emit_performance_stats(self):
        """发送性能统计"""
        current_time = time.time()
        elapsed = current_time - self.start_time or 1
        
        reconstruction_rate = 0
        if self.processed_pairs > 0:
            reconstruction_rate = self.successful_reconstructions / self.processed_pairs
        
        stats = {
            'processed_pairs': self.processed_pairs,
            'successful_reconstructions': self.successful_reconstructions,
            'reconstruction_rate': reconstruction_rate,
            'trajectory_quality': self.trajectory_quality,
            'processing_fps': self.processed_pairs / elapsed,
            'buffer1_size': len(self.detection_buffer1),
            'buffer2_size': len(self.detection_buffer2)
        }
        
        self.performance_stats.emit(stats)
    
    def reset_trajectory(self):
        """重置3D轨迹"""
        self.stereo_processor.reset()
        self.last_3d_points.clear()
        self.trajectory_quality = 0.0
        self.processed_pairs = 0
        self.successful_reconstructions = 0
    
    def get_trajectory_data(self):
        """获取3D轨迹数据"""
        return {
            'trajectory_3d': list(self.stereo_processor.trajectory_3d),
            'timestamps': list(self.stereo_processor.trajectory_timestamps),
            'quality': self.trajectory_quality
        }
    
    def run(self):
        """主线程循环"""
        self.processing = True
        self.start_time = time.time()
        
        try:
            while self.processing:
                # 清理过期的检测结果
                self.cleanup_old_detections()
                
                # 短暂休眠
                self.msleep(50)
        
        except Exception as e:
            self.error_occurred.emit(f"Stereo worker error: {str(e)}")
        
        finally:
            self.status_changed.emit("Stereo worker stopped")
    
    def cleanup_old_detections(self):
        """清理过期的检测结果"""
        current_time = time.time()
        max_age = 0.5  # 500ms
        
        self.mutex.lock()
        
        # 清理buffer1
        while (self.detection_buffer1 and 
               current_time - self.detection_buffer1[0]['timestamp'] > max_age):
            self.detection_buffer1.popleft()
        
        # 清理buffer2
        while (self.detection_buffer2 and 
               current_time - self.detection_buffer2[0]['timestamp'] > max_age):
            self.detection_buffer2.popleft()
        
        self.mutex.unlock()
    
    def stop(self):
        """停止处理"""
        self.processing = False
        self.stats_timer.stop()