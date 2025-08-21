"""
预测处理线程
负责物理模型计算和落点预测
"""
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QTimer
from collections import deque
from predictor_core import TrajectoryPredictor, CourtBoundaryAnalyzer
from config import config


class PredictionWorker(QThread):
    """预测处理工作线程"""
    
    # 信号定义
    prediction_ready = pyqtSignal(dict)  # 预测结果
    prediction_progress = pyqtSignal(int)  # 预测进度 (0-100)
    boundary_analysis = pyqtSignal(dict)  # 边界分析结果
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    performance_stats = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        
        # 预测器
        self.trajectory_predictor = TrajectoryPredictor()
        self.boundary_analyzer = CourtBoundaryAnalyzer()
        
        # 处理队列
        self.prediction_queue = deque(maxlen=5)
        
        # 处理状态
        self.processing = False
        self.current_prediction = None
        
        # 线程同步
        self.mutex = QMutex()
        
        # 配置参数
        self.prediction_method = 'ensemble'
        self.min_trajectory_points = 5
        self.court_type = 'singles'
        
        # 性能统计
        self.prediction_count = 0
        self.successful_predictions = 0
        self.start_time = 0
        self.processing_times = deque(maxlen=10)
        
        # 预测历史
        self.prediction_history = deque(maxlen=20)
        
        # 定时器
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.emit_performance_stats)
        self.stats_timer.start(2000)  # 每2秒更新
    
    def update_parameters(self, parameters):
        """更新预测参数"""
        if 'prediction_method' in parameters:
            self.prediction_method = parameters['prediction_method']
        
        if 'min_trajectory_points' in parameters:
            self.min_trajectory_points = parameters['min_trajectory_points']
        
        if 'court_type' in parameters:
            self.court_type = parameters['court_type']
        
        # 更新物理参数
        physics_params = ['shuttlecock_mass', 'drag_coefficient', 'air_density']
        for param in physics_params:
            if param in parameters:
                setattr(config, param, parameters[param])
    
    def request_prediction(self, trajectory_3d, timestamps, priority='normal'):
        """请求预测"""
        if len(trajectory_3d) < self.min_trajectory_points:
            self.error_occurred.emit(f"Insufficient trajectory points: {len(trajectory_3d)} < {self.min_trajectory_points}")
            return False
        
        prediction_request = {
            'trajectory_3d': trajectory_3d.copy(),
            'timestamps': timestamps.copy(),
            'priority': priority,
            'request_time': time.time(),
            'method': self.prediction_method
        }
        
        self.mutex.lock()
        
        # 根据优先级插入队列
        if priority == 'high':
            # 高优先级插入队列前端
            if len(self.prediction_queue) == self.prediction_queue.maxlen:
                self.prediction_queue.pop()  # 移除最旧的
            self.prediction_queue.appendleft(prediction_request)
        else:
            # 普通优先级添加到队列末尾
            self.prediction_queue.append(prediction_request)
        
        self.mutex.unlock()
        
        self.status_changed.emit(f"Prediction requested (Queue: {len(self.prediction_queue)})")
        return True
    
    def run(self):
        """主线程循环"""
        self.processing = True
        self.start_time = time.time()
        
        self.status_changed.emit("Prediction worker started")
        
        try:
            while self.processing:
                self.mutex.lock()
                
                # 检查是否有待处理的预测请求
                if self.prediction_queue:
                    request = self.prediction_queue.popleft()
                    self.mutex.unlock()
                    
                    # 处理预测请求
                    self.process_prediction_request(request)
                    
                else:
                    self.mutex.unlock()
                    # 没有请求时短暂休眠
                    self.msleep(100)
        
        except Exception as e:
            self.error_occurred.emit(f"Prediction worker error: {str(e)}")
        
        finally:
            self.status_changed.emit("Prediction worker stopped")
    
    def process_prediction_request(self, request):
        """处理预测请求"""
        start_time = time.time()
        
        try:
            self.current_prediction = request
            self.prediction_count += 1
            
            # 更新状态
            self.status_changed.emit("Processing prediction...")
            self.prediction_progress.emit(10)
            
            # 提取轨迹数据
            trajectory_3d = request['trajectory_3d']
            timestamps = request['timestamps']
            method = request['method']
            
            # 数据预处理
            self.prediction_progress.emit(20)
            filtered_trajectory, filtered_timestamps = self.preprocess_trajectory(
                trajectory_3d, timestamps
            )
            
            # 执行预测
            self.prediction_progress.emit(40)
            prediction_result = self.trajectory_predictor.predict_landing_point(
                filtered_trajectory, filtered_timestamps, method=method
            )
            
            self.prediction_progress.emit(70)
            
            if prediction_result:
                # 边界分析
                boundary_result = self.boundary_analyzer.analyze_landing(
                    prediction_result['landing_point'], self.court_type
                )
                
                self.prediction_progress.emit(90)
                
                # 组合结果
                final_result = self.create_final_result(
                    prediction_result, boundary_result, request
                )
                
                # 记录成功
                self.successful_predictions += 1
                
                # 添加到历史
                self.prediction_history.append(final_result)
                
                # 发送结果
                self.prediction_ready.emit(final_result)
                if boundary_result:
                    self.boundary_analysis.emit(boundary_result)
                
                self.status_changed.emit("Prediction completed successfully")
                
            else:
                self.error_occurred.emit("Prediction failed - insufficient data or model error")
                self.status_changed.emit("Prediction failed")
            
            self.prediction_progress.emit(100)
            
        except Exception as e:
            self.error_occurred.emit(f"Prediction processing error: {str(e)}")
            self.status_changed.emit("Prediction error")
        
        finally:
            # 记录处理时间
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            self.current_prediction = None
            
            # 清除进度
            QTimer.singleShot(2000, lambda: self.prediction_progress.emit(0))
    
    def preprocess_trajectory(self, trajectory_3d, timestamps):
        """预处理轨迹数据"""
        if not trajectory_3d or not timestamps:
            return [], []
        
        # 转换为numpy数组
        trajectory = np.array(trajectory_3d)
        times = np.array(timestamps)
        
        # 数据验证
        valid_indices = []
        for i, (point, ts) in enumerate(zip(trajectory, times)):
            # 检查3D点有效性
            if (len(point) == 3 and 
                all(np.isfinite(point)) and
                -100 <= point[0] <= 710 and  # X范围
                -100 <= point[1] <= 770 and  # Y范围
                0 <= point[2] <= 500):       # Z范围
                valid_indices.append(i)
        
        if len(valid_indices) < self.min_trajectory_points:
            return [], []
        
        # 提取有效数据
        filtered_trajectory = trajectory[valid_indices].tolist()
        filtered_timestamps = times[valid_indices].tolist()
        
        # 时间排序
        sorted_pairs = sorted(zip(filtered_timestamps, filtered_trajectory))
        filtered_timestamps, filtered_trajectory = zip(*sorted_pairs)
        
        # 轨迹平滑（可选）
        if len(filtered_trajectory) > 5:
            filtered_trajectory = self.smooth_trajectory(list(filtered_trajectory))
        
        return list(filtered_trajectory), list(filtered_timestamps)
    
    def smooth_trajectory(self, trajectory, window_size=3):
        """轨迹平滑"""
        if len(trajectory) < window_size:
            return trajectory
        
        smoothed = []
        trajectory = np.array(trajectory)
        
        for i in range(len(trajectory)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(trajectory), i + window_size // 2 + 1)
            
            # 计算窗口内点的平均值
            window_points = trajectory[start_idx:end_idx]
            smoothed_point = np.mean(window_points, axis=0)
            smoothed.append(smoothed_point.tolist())
        
        return smoothed
    
    def create_final_result(self, prediction_result, boundary_result, request):
        """创建最终结果"""
        result = {
            'prediction': prediction_result,
            'boundary_analysis': boundary_result,
            'request_info': {
                'method': request['method'],
                'trajectory_points': len(request['trajectory_3d']),
                'time_span': request['timestamps'][-1] - request['timestamps'][0] if request['timestamps'] else 0,
                'processing_time': time.time() - request['request_time']
            },
            'quality_metrics': self.calculate_prediction_quality(prediction_result),
            'timestamp': time.time()
        }
        
        return result
    
    def calculate_prediction_quality(self, prediction_result):
        """计算预测质量"""
        if not prediction_result:
            return {'overall': 0.0}
        
        metrics = {}
        
        # 置信度
        confidence = prediction_result.get('confidence', 0.0)
        metrics['confidence'] = confidence
        
        # 轨迹完整性
        trajectory = prediction_result.get('trajectory', [])
        trajectory_completeness = min(1.0, len(trajectory) / 50)  # 假设50个点为完整轨迹
        metrics['trajectory_completeness'] = trajectory_completeness
        
        # 方法一致性（如果有多种方法的结果）
        if 'sub_predictions' in prediction_result:
            sub_predictions = prediction_result['sub_predictions']
            consistency = self.calculate_method_consistency(sub_predictions)
            metrics['method_consistency'] = consistency
        else:
            metrics['method_consistency'] = 1.0
        
        # 物理合理性
        physics_score = self.evaluate_physics_reasonableness(prediction_result)
        metrics['physics_reasonableness'] = physics_score
        
        # 综合质量分数
        weights = {
            'confidence': 0.3,
            'trajectory_completeness': 0.2,
            'method_consistency': 0.2,
            'physics_reasonableness': 0.3
        }
        
        overall = sum(metrics[key] * weights[key] for key in weights)
        metrics['overall'] = overall
        
        return metrics
    
    def calculate_method_consistency(self, sub_predictions):
        """计算不同方法的一致性"""
        if len(sub_predictions) < 2:
            return 1.0
        
        landing_points = []
        for pred in sub_predictions.values():
            if pred and pred.get('landing_point'):
                landing_points.append(pred['landing_point'])
        
        if len(landing_points) < 2:
            return 0.5
        
        # 计算落点之间的距离
        distances = []
        for i in range(len(landing_points)):
            for j in range(i + 1, len(landing_points)):
                dist = np.linalg.norm(np.array(landing_points[i][:2]) - np.array(landing_points[j][:2]))
                distances.append(dist)
        
        # 一致性分数（距离越小一致性越高）
        avg_distance = np.mean(distances) if distances else 0
        consistency = 1.0 / (1.0 + avg_distance / 50)  # 50cm为参考距离
        
        return consistency
    
    def evaluate_physics_reasonableness(self, prediction_result):
        """评估物理合理性"""
        landing_point = prediction_result.get('landing_point')
        trajectory = prediction_result.get('trajectory', [])
        
        if not landing_point or not trajectory:
            return 0.5
        
        scores = []
        
        # 落点高度合理性（应该接近0）
        height_score = 1.0 - abs(landing_point[2]) / 100  # 100cm为参考
        scores.append(max(0, height_score))
        
        # 轨迹单调性（Z坐标应该大致递减）
        if len(trajectory) > 5:
            z_coords = [p[2] for p in trajectory if len(p) > 2]
            if len(z_coords) > 1:
                decreasing_count = sum(1 for i in range(1, len(z_coords)) if z_coords[i] <= z_coords[i-1])
                monotonic_score = decreasing_count / (len(z_coords) - 1)
                scores.append(monotonic_score)
        
        # 速度合理性（不应该过快或过慢）
        if len(trajectory) > 2:
            distances = []
            for i in range(1, len(trajectory)):
                if len(trajectory[i]) > 2 and len(trajectory[i-1]) > 2:
                    dist = np.linalg.norm(np.array(trajectory[i][:3]) - np.array(trajectory[i-1][:3]))
                    distances.append(dist)
            
            if distances:
                avg_speed = np.mean(distances) * 30  # 假设30fps，转换为速度
                # 羽毛球合理速度范围：5-70 m/s
                if 5 <= avg_speed <= 70:
                    speed_score = 1.0
                else:
                    speed_score = max(0, 1.0 - abs(avg_speed - 37.5) / 37.5)  # 37.5为中间值
                scores.append(speed_score)
        
        return np.mean(scores) if scores else 0.5
    
    def get_prediction_statistics(self):
        """获取预测统计信息"""
        success_rate = 0
        if self.prediction_count > 0:
            success_rate = self.successful_predictions / self.prediction_count
        
        avg_processing_time = 0
        if self.processing_times:
            avg_processing_time = np.mean(self.processing_times)
        
        return {
            'total_predictions': self.prediction_count,
            'successful_predictions': self.successful_predictions,
            'success_rate': success_rate,
            'avg_processing_time': avg_processing_time,
            'queue_size': len(self.prediction_queue),
            'prediction_history_size': len(self.prediction_history)
        }
    
    def get_recent_predictions(self, count=5):
        """获取最近的预测结果"""
        recent = list(self.prediction_history)[-count:]
        return recent
    
    def emit_performance_stats(self):
        """发送性能统计"""
        stats = self.get_prediction_statistics()
        
        # 添加实时状态
        stats['current_queue_size'] = len(self.prediction_queue)
        stats['is_processing'] = self.current_prediction is not None
        
        if self.current_prediction:
            stats['current_processing_time'] = time.time() - self.current_prediction['request_time']
        
        self.performance_stats.emit(stats)
    
    def clear_history(self):
        """清除预测历史"""
        self.prediction_history.clear()
    
    def clear_queue(self):
        """清除预测队列"""
        self.mutex.lock()
        self.prediction_queue.clear()
        self.mutex.unlock()
        self.status_changed.emit("Prediction queue cleared")
    
    def stop(self):
        """停止处理"""
        self.processing = False
        self.stats_timer.stop()
        self.clear_queue()