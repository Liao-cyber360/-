"""
配置管理模块
统一管理系统参数和用户偏好设置
"""
import os
import time
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

try:
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication
except ImportError:
    # 如果PyQt6不可用，提供备用实现
    class QSettings:
        def __init__(self, *args):
            self._data = {}
        
        def value(self, key, default, type=None):
            return self._data.get(key, default)
        
        def setValue(self, key, value):
            self._data[key] = value


class Config:
    """配置类,集中管理所有系统参数"""
    
    def __init__(self):
        self.settings = QSettings("BadmintonSystem", "PyQt6")
        self._load_default_config()
        self._load_user_config()
    
    def _load_default_config(self):
        """加载默认配置"""
        # 相机参数
        self.camera_params_file_1 = self.get_setting("E:\hawkeye\calibration\calibration_results_2025-08-11_18-06-15.yaml", "")
        self.camera_params_file_2 = self.get_setting("E:\hawkeye\calibration\calibration_results_2025-08-11_18-06-15.yaml", "")
        
        # 模型参数
        self.yolo_ball_model = self.get_setting("E:\\hawkeye\\ball\\best.pt", "")
        self.yolo_court_model = self.get_setting("E:\\hawkeye\\field\\best.pt", "")
        
        # 视频参数
        self.video_width = self.get_setting("video/width", 1280)
        self.video_height = self.get_setting("video/height", 720)
        self.fps = self.get_setting("video/fps", 30)
        
        # 分析参数
        self.trajectory_buffer_size = self.get_setting("analysis/trajectory_buffer_size", 300)
        self.landing_analysis_window = self.get_setting("analysis/landing_window", 10)
        self.prediction_time_window = self.get_setting("analysis/prediction_time_window", 2)
        self.poly_fit_degree = self.get_setting("analysis/poly_fit_degree", 4)
        
        # 物理模型参数
        self.shuttlecock_mass = self.get_setting("physics/mass", 0.005)
        self.shuttlecock_radius = self.get_setting("physics/radius", 0.025)
        self.air_density = self.get_setting("physics/air_density", 1.225)
        self.drag_coefficient = self.get_setting("physics/drag_coefficient", 0.6)
        self.gravity = self.get_setting("physics/gravity", 9.8)
        
        # 计算空气动力学长度
        self.cross_section = np.pi * self.shuttlecock_radius ** 2
        self.aero_length = (2 * self.shuttlecock_mass / 
                           (self.air_density * self.cross_section * self.drag_coefficient))
        
        # EKF参数
        self.ekf_process_noise = self.get_setting("ekf/process_noise", 0.01)
        self.ekf_measurement_noise = self.get_setting("ekf/measurement_noise", 0.1)
        
        # 界面参数
        self.court_view_width = self.get_setting("ui/court_view_width", 610)
        self.court_view_height = self.get_setting("ui/court_view_height", 1340)
        self.display_scale = self.get_setting("ui/display_scale", 0.3)
        
        # 结果目录
        self.results_dir = f"./results_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 主题设置
        self.theme = self.get_setting("ui/theme", "dark")
        self.window_geometry = self.get_setting("ui/window_geometry", None)
    
    def _load_user_config(self):
        """加载用户配置文件"""
        config_file = "config.yaml"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                    self._apply_user_config(user_config)
            except Exception as e:
                print(f"Warning: Failed to load user config: {e}")
    
    def _apply_user_config(self, user_config):
        """应用用户配置"""
        if 'camera' in user_config:
            self.camera_params_file_1 = user_config['camera'].get('params_file_1', self.camera_params_file_1)
            self.camera_params_file_2 = user_config['camera'].get('params_file_2', self.camera_params_file_2)
        
        if 'models' in user_config:
            self.yolo_ball_model = user_config['models'].get('yolo_ball', self.yolo_ball_model)
            self.yolo_court_model = user_config['models'].get('yolo_court', self.yolo_court_model)

    def get_setting(self, key, default_value):
        """获取设置值"""
        if default_value is None:
            return self.settings.value(key, default_value)
        return self.settings.value(key, default_value, type=type(default_value))
    
    def set_setting(self, key, value):
        """设置值"""
        self.settings.setValue(key, value)
    
    def get_aero_params(self):
        """返回空气动力学参数"""
        return {
            'mass': self.shuttlecock_mass,
            'radius': self.shuttlecock_radius,
            'air_density': self.air_density,
            'drag_coefficient': self.drag_coefficient,
            'gravity': self.gravity,
            'aero_length': self.aero_length,
            'cross_section': self.cross_section
        }
    
    def save_config(self):
        """保存配置到文件"""
        config_data = {
            'camera': {
                'params_file_1': self.camera_params_file_1,
                'params_file_2': self.camera_params_file_2
            },
            'models': {
                'yolo_ball': self.yolo_ball_model,
                'yolo_court': self.yolo_court_model
            },
            'video': {
                'width': self.video_width,
                'height': self.video_height,
                'fps': self.fps
            },
            'physics': {
                'mass': self.shuttlecock_mass,
                'radius': self.shuttlecock_radius,
                'air_density': self.air_density,
                'drag_coefficient': self.drag_coefficient,
                'gravity': self.gravity
            },
            'ui': {
                'theme': self.theme,
                'court_view_width': self.court_view_width,
                'court_view_height': self.court_view_height
            }
        }
        
        try:
            with open('config.yaml', 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"Warning: Failed to save config: {e}")


# 全局配置实例
config = Config()