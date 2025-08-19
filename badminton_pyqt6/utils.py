"""
工具函数模块
提供系统辅助功能
"""
import os
import time
import json
import numpy as np
import cv2
from PyQt6.QtWidgets import QMessageBox, QProgressDialog, QApplication
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import psutil


class SystemUtils:
    """系统工具类"""
    
    @staticmethod
    def get_system_info():
        """获取系统信息"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
        }
        return info
    
    @staticmethod
    def format_bytes(bytes_value):
        """格式化字节数"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"
    
    @staticmethod
    def check_dependencies():
        """检查依赖项"""
        dependencies = {
            'cv2': False,
            'numpy': False,
            'PyQt6': False,
            'ultralytics': False,
            'open3d': False,
            'filterpy': False,
            'scipy': False
        }
        
        for dep in dependencies:
            try:
                __import__(dep)
                dependencies[dep] = True
            except ImportError:
                dependencies[dep] = False
        
        return dependencies
    
    @staticmethod
    def create_results_directory():
        """创建结果目录"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        results_dir = f"./results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        return results_dir
    
    @staticmethod
    def save_configuration(config_dict, file_path):
        """保存配置到文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Failed to save configuration: {e}")
            return False
    
    @staticmethod
    def load_configuration(file_path):
        """从文件加载配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            return None


class ImageUtils:
    """图像处理工具类"""
    
    @staticmethod
    def cv2_to_qimage(cv_img):
        """OpenCV图像转换为QImage"""
        if cv_img is None:
            return None
        
        height, width = cv_img.shape[:2]
        
        if len(cv_img.shape) == 3:
            # 彩色图像
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * width
            qt_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            # 灰度图像
            bytes_per_line = width
            qt_image = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        
        return qt_image
    
    @staticmethod
    def qimage_to_cv2(qt_img):
        """QImage转换为OpenCV图像"""
        if qt_img is None:
            return None
        
        width = qt_img.width()
        height = qt_img.height()
        
        ptr = qt_img.bits()
        ptr.setsize(qt_img.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # RGBA
        
        # 转换为BGR
        cv_img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return cv_img
    
    @staticmethod
    def resize_image_keep_ratio(image, target_size):
        """保持比例缩放图像"""
        if image is None:
            return None
        
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        
        # 计算新尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return resized
    
    @staticmethod
    def add_text_with_background(image, text, position, font_scale=0.7, 
                                color=(255, 255, 255), bg_color=(0, 0, 0), 
                                thickness=1, padding=5):
        """添加带背景的文本"""
        if image is None:
            return image
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 获取文本尺寸
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 计算背景矩形
        x, y = position
        rect_x1 = x - padding
        rect_y1 = y - text_h - padding
        rect_x2 = x + text_w + padding
        rect_y2 = y + baseline + padding
        
        # 绘制背景矩形
        cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
        
        # 绘制文本
        cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
        
        return image
    
    @staticmethod
    def draw_trajectory_on_image(image, trajectory, color=(255, 0, 0), thickness=2):
        """在图像上绘制轨迹"""
        if image is None or not trajectory or len(trajectory) < 2:
            return image
        
        result = image.copy()
        
        # 绘制轨迹线
        for i in range(1, len(trajectory)):
            pt1 = tuple(map(int, trajectory[i-1]))
            pt2 = tuple(map(int, trajectory[i]))
            cv2.line(result, pt1, pt2, color, thickness)
        
        # 绘制轨迹点
        for i, point in enumerate(trajectory):
            center = tuple(map(int, point))
            radius = max(2, thickness)
            
            # 颜色渐变
            alpha = i / max(1, len(trajectory) - 1)
            point_color = (
                int(color[0] * alpha + 128 * (1 - alpha)),
                int(color[1] * alpha + 128 * (1 - alpha)),
                int(color[2] * alpha + 128 * (1 - alpha))
            )
            
            cv2.circle(result, center, radius, point_color, -1)
        
        return result


class MathUtils:
    """数学工具类"""
    
    @staticmethod
    def calculate_distance_2d(point1, point2):
        """计算2D距离"""
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    @staticmethod
    def calculate_distance_3d(point1, point2):
        """计算3D距离"""
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    @staticmethod
    def calculate_velocity(positions, timestamps):
        """计算速度序列"""
        if len(positions) < 2 or len(timestamps) < 2:
            return []
        
        velocities = []
        for i in range(1, len(positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                displacement = np.array(positions[i]) - np.array(positions[i-1])
                velocity = np.linalg.norm(displacement) / dt
                velocities.append(velocity)
            else:
                velocities.append(0)
        
        return velocities
    
    @staticmethod
    def calculate_acceleration(velocities, timestamps):
        """计算加速度序列"""
        if len(velocities) < 2 or len(timestamps) < 2:
            return []
        
        accelerations = []
        for i in range(1, len(velocities)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                dv = velocities[i] - velocities[i-1]
                acceleration = dv / dt
                accelerations.append(acceleration)
            else:
                accelerations.append(0)
        
        return accelerations
    
    @staticmethod
    def smooth_trajectory(trajectory, window_size=5):
        """轨迹平滑"""
        if len(trajectory) < window_size:
            return trajectory
        
        smoothed = []
        trajectory = np.array(trajectory)
        
        for i in range(len(trajectory)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(trajectory), i + window_size // 2 + 1)
            
            window_points = trajectory[start_idx:end_idx]
            smoothed_point = np.mean(window_points, axis=0)
            smoothed.append(smoothed_point)
        
        return smoothed
    
    @staticmethod
    def interpolate_trajectory(trajectory, timestamps, target_fps=30):
        """轨迹插值"""
        if len(trajectory) < 2 or len(timestamps) < 2:
            return trajectory, timestamps
        
        # 创建目标时间序列
        start_time = timestamps[0]
        end_time = timestamps[-1]
        dt = 1.0 / target_fps
        target_times = np.arange(start_time, end_time + dt, dt)
        
        # 插值每个维度
        trajectory = np.array(trajectory)
        interpolated = []
        
        for dim in range(trajectory.shape[1]):
            interp_values = np.interp(target_times, timestamps, trajectory[:, dim])
            interpolated.append(interp_values)
        
        interpolated_trajectory = np.array(interpolated).T.tolist()
        
        return interpolated_trajectory, target_times.tolist()


class ValidationUtils:
    """数据验证工具类"""
    
    @staticmethod
    def validate_trajectory_3d(trajectory):
        """验证3D轨迹数据"""
        if not trajectory:
            return False, "Empty trajectory"
        
        for i, point in enumerate(trajectory):
            if len(point) != 3:
                return False, f"Point {i} is not 3D"
            
            if not all(np.isfinite(point)):
                return False, f"Point {i} contains invalid values"
            
            x, y, z = point
            
            # 基本范围检查
            if not (-1000 <= x <= 1000):
                return False, f"Point {i} X coordinate out of range"
            
            if not (-1000 <= y <= 1000):
                return False, f"Point {i} Y coordinate out of range"
            
            if not (0 <= z <= 1000):
                return False, f"Point {i} Z coordinate out of range"
        
        return True, "Valid trajectory"
    
    @staticmethod
    def validate_timestamps(timestamps):
        """验证时间戳数据"""
        if not timestamps:
            return False, "Empty timestamps"
        
        for i, ts in enumerate(timestamps):
            if not isinstance(ts, (int, float)):
                return False, f"Timestamp {i} is not numeric"
            
            if not np.isfinite(ts):
                return False, f"Timestamp {i} is invalid"
            
            if i > 0 and ts <= timestamps[i-1]:
                return False, f"Timestamp {i} is not monotonic"
        
        return True, "Valid timestamps"
    
    @staticmethod
    def validate_camera_parameters(camera_matrix, dist_coeffs):
        """验证相机参数"""
        if camera_matrix is None:
            return False, "Camera matrix is None"
        
        if camera_matrix.shape != (3, 3):
            return False, "Camera matrix shape is not 3x3"
        
        if dist_coeffs is None:
            return False, "Distortion coefficients are None"
        
        if len(dist_coeffs) < 4:
            return False, "Insufficient distortion coefficients"
        
        if not np.all(np.isfinite(camera_matrix)):
            return False, "Camera matrix contains invalid values"
        
        if not np.all(np.isfinite(dist_coeffs)):
            return False, "Distortion coefficients contain invalid values"
        
        return True, "Valid camera parameters"


class PerformanceMonitor:
    """性能监控工具"""
    
    def __init__(self):
        self.start_time = time.time()
        self.checkpoints = {}
    
    def start_timer(self, name):
        """开始计时"""
        self.checkpoints[name] = time.time()
    
    def end_timer(self, name):
        """结束计时"""
        if name in self.checkpoints:
            elapsed = time.time() - self.checkpoints[name]
            return elapsed
        return 0
    
    def get_elapsed_time(self):
        """获取总耗时"""
        return time.time() - self.start_time
    
    def get_memory_usage(self):
        """获取内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss,  # 物理内存
            'vms': memory_info.vms,  # 虚拟内存
            'percent': process.memory_percent()
        }
    
    def get_cpu_usage(self):
        """获取CPU使用率"""
        return psutil.cpu_percent(interval=0.1)


class DialogUtils:
    """对话框工具类"""
    
    @staticmethod
    def show_info(parent, title, message):
        """显示信息对话框"""
        QMessageBox.information(parent, title, message)
    
    @staticmethod
    def show_warning(parent, title, message):
        """显示警告对话框"""
        QMessageBox.warning(parent, title, message)
    
    @staticmethod
    def show_error(parent, title, message):
        """显示错误对话框"""
        QMessageBox.critical(parent, title, message)
    
    @staticmethod
    def show_question(parent, title, message):
        """显示询问对话框"""
        reply = QMessageBox.question(parent, title, message,
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return reply == QMessageBox.StandardButton.Yes
    
    @staticmethod
    def show_progress_dialog(parent, title, maximum=100):
        """创建进度对话框"""
        progress = QProgressDialog(title, "Cancel", 0, maximum, parent)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        return progress


class DataExporter:
    """数据导出工具"""
    
    @staticmethod
    def export_trajectory_csv(trajectory_3d, timestamps, file_path):
        """导出轨迹为CSV文件"""
        try:
            import csv
            
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # 写入头部
                writer.writerow(['Timestamp', 'X', 'Y', 'Z'])
                
                # 写入数据
                for i, (point, ts) in enumerate(zip(trajectory_3d, timestamps)):
                    writer.writerow([ts, point[0], point[1], point[2]])
            
            return True
        
        except Exception as e:
            print(f"Failed to export trajectory CSV: {e}")
            return False
    
    @staticmethod
    def export_prediction_json(prediction_result, file_path):
        """导出预测结果为JSON文件"""
        try:
            # 转换numpy数组为列表
            serializable_result = DataExporter._make_json_serializable(prediction_result)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=4, ensure_ascii=False)
            
            return True
        
        except Exception as e:
            print(f"Failed to export prediction JSON: {e}")
            return False
    
    @staticmethod
    def _make_json_serializable(obj):
        """使对象可以JSON序列化"""
        if isinstance(obj, dict):
            return {key: DataExporter._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [DataExporter._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj


class LogManager:
    """日志管理器"""
    
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.log_entries = []
    
    def log(self, level, message):
        """记录日志"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        entry = f"[{timestamp}] {level}: {message}"
        
        self.log_entries.append(entry)
        
        # 控制台输出
        print(entry)
        
        # 文件输出
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(entry + '\n')
            except Exception as e:
                print(f"Failed to write log file: {e}")
    
    def info(self, message):
        """信息日志"""
        self.log("INFO", message)
    
    def warning(self, message):
        """警告日志"""
        self.log("WARNING", message)
    
    def error(self, message):
        """错误日志"""
        self.log("ERROR", message)
    
    def debug(self, message):
        """调试日志"""
        self.log("DEBUG", message)
    
    def get_recent_logs(self, count=50):
        """获取最近的日志"""
        return self.log_entries[-count:]
    
    def clear_logs(self):
        """清除日志"""
        self.log_entries.clear()


# 全局实例
logger = LogManager()
performance_monitor = PerformanceMonitor()