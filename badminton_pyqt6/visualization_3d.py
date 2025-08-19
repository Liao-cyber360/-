"""
3D可视化组件
集成Open3D到PyQt6，支持多种点云类型切换和轨迹展示
"""
import numpy as np
import open3d as o3d
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QComboBox, QCheckBox, QSlider,
                            QGroupBox, QFormLayout, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
import time


class Visualization3DWidget(QWidget):
    """3D可视化主组件"""
    
    # 信号定义
    view_changed = pyqtSignal(dict)  # 视角变化信号
    point_selected = pyqtSignal(np.ndarray)  # 点选择信号
    
    def __init__(self):
        super().__init__()
        
        # Open3D可视化器
        self.vis = None
        self.geometry_dict = {}
        
        # 可视化数据
        self.trajectory_3d = []
        self.prediction_trajectory = []
        self.landing_point = None
        self.court_mesh = None
        
        # 显示状态
        self.show_trajectory = True
        self.show_prediction = True
        self.show_court = True
        self.show_coordinate_frame = True
        
        # 颜色设置
        self.trajectory_color = [0, 0.7, 1.0]  # 蓝色
        self.prediction_color = [1.0, 0.5, 0]  # 橙色
        self.landing_color = [1.0, 0, 0]       # 红色
        self.court_color = [0.2, 0.8, 0.2]    # 绿色
        
        self.setup_ui()
        self.setup_visualization()
        
        # 更新定时器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visualization)
        self.update_timer.start(100)  # 每100ms更新一次
    
    def setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 标题
        title_label = QLabel("3D Trajectory Visualization")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #333;
                padding: 5px;
                background-color: #f0f0f0;
                border-radius: 3px;
            }
        """)
        layout.addWidget(title_label)
        
        # 控制面板
        self.setup_control_panel(layout)
        
        # 3D视图占位符（实际的Open3D窗口将嵌入到这里）
        self.view_placeholder = QFrame()
        self.view_placeholder.setMinimumHeight(400)
        self.view_placeholder.setStyleSheet("""
            QFrame {
                background-color: #000;
                border: 2px solid #ccc;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.view_placeholder, 1)
        
        # 状态信息
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666; font-size: 11px; padding: 2px;")
        layout.addWidget(self.status_label)
    
    def setup_control_panel(self, parent_layout):
        """设置控制面板"""
        control_frame = QFrame()
        control_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        control_layout = QHBoxLayout(control_frame)
        
        # 显示选项组
        display_group = QGroupBox("Display Options")
        display_layout = QFormLayout(display_group)
        
        self.show_trajectory_cb = QCheckBox("Trajectory")
        self.show_trajectory_cb.setChecked(True)
        self.show_trajectory_cb.toggled.connect(self.on_display_options_changed)
        display_layout.addRow(self.show_trajectory_cb)
        
        self.show_prediction_cb = QCheckBox("Prediction")
        self.show_prediction_cb.setChecked(True)
        self.show_prediction_cb.toggled.connect(self.on_display_options_changed)
        display_layout.addRow(self.show_prediction_cb)
        
        self.show_court_cb = QCheckBox("Court")
        self.show_court_cb.setChecked(True)
        self.show_court_cb.toggled.connect(self.on_display_options_changed)
        display_layout.addRow(self.show_court_cb)
        
        self.show_frame_cb = QCheckBox("Coordinate Frame")
        self.show_frame_cb.setChecked(True)
        self.show_frame_cb.toggled.connect(self.on_display_options_changed)
        display_layout.addRow(self.show_frame_cb)
        
        control_layout.addWidget(display_group)
        
        # 视图控制组
        view_group = QGroupBox("View Control")
        view_layout = QVBoxLayout(view_group)
        
        # 预设视角
        view_preset_layout = QHBoxLayout()
        
        self.top_view_btn = QPushButton("Top View")
        self.top_view_btn.clicked.connect(lambda: self.set_view_preset("top"))
        view_preset_layout.addWidget(self.top_view_btn)
        
        self.side_view_btn = QPushButton("Side View")
        self.side_view_btn.clicked.connect(lambda: self.set_view_preset("side"))
        view_preset_layout.addWidget(self.side_view_btn)
        
        self.front_view_btn = QPushButton("Front View")
        self.front_view_btn.clicked.connect(lambda: self.set_view_preset("front"))
        view_preset_layout.addWidget(self.front_view_btn)
        
        view_layout.addLayout(view_preset_layout)
        
        # 重置按钮
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_view)
        view_layout.addWidget(self.reset_view_btn)
        
        control_layout.addWidget(view_group)
        
        # 数据控制组
        data_group = QGroupBox("Data Control")
        data_layout = QVBoxLayout(data_group)
        
        self.clear_trajectory_btn = QPushButton("Clear Trajectory")
        self.clear_trajectory_btn.clicked.connect(self.clear_trajectory)
        data_layout.addWidget(self.clear_trajectory_btn)
        
        self.clear_prediction_btn = QPushButton("Clear Prediction")
        self.clear_prediction_btn.clicked.connect(self.clear_prediction)
        data_layout.addWidget(self.clear_prediction_btn)
        
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(self.clear_all)
        data_layout.addWidget(self.clear_all_btn)
        
        control_layout.addWidget(data_group)
        
        control_layout.addStretch()
        parent_layout.addWidget(control_frame)
    
    def setup_visualization(self):
        """设置Open3D可视化"""
        try:
            # 创建可视化器
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="3D Visualization", 
                                 width=800, height=600, visible=False)
            
            # 设置渲染选项
            render_option = self.vis.get_render_option()
            render_option.background_color = np.array([0.1, 0.1, 0.1])
            render_option.point_size = 5.0
            render_option.line_width = 2.0
            
            # 创建坐标系
            self.create_coordinate_frame()
            
            # 创建场地
            self.create_court_mesh()
            
            self.status_label.setText("3D visualization initialized")
            
        except Exception as e:
            self.status_label.setText(f"3D initialization error: {str(e)}")
            print(f"Open3D visualization setup error: {e}")
    
    def create_coordinate_frame(self):
        """创建坐标系"""
        try:
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=100, origin=[0, 0, 0]
            )
            self.geometry_dict['coordinate_frame'] = coordinate_frame
            
            if self.vis:
                self.vis.add_geometry(coordinate_frame)
        
        except Exception as e:
            print(f"Failed to create coordinate frame: {e}")
    
    def create_court_mesh(self):
        """创建场地网格"""
        try:
            # 场地尺寸 (单位: cm)
            court_width = 610
            court_length = 670
            
            # 创建场地平面
            court_vertices = np.array([
                [0, 0, 0],
                [court_width, 0, 0],
                [court_width, court_length, 0],
                [0, court_length, 0]
            ], dtype=np.float64)
            
            court_triangles = np.array([
                [0, 1, 2],
                [0, 2, 3]
            ])
            
            court_mesh = o3d.geometry.TriangleMesh()
            court_mesh.vertices = o3d.utility.Vector3dVector(court_vertices)
            court_mesh.triangles = o3d.utility.Vector3iVector(court_triangles)
            court_mesh.paint_uniform_color(self.court_color)
            
            # 计算法向量
            court_mesh.compute_vertex_normals()
            
            self.geometry_dict['court'] = court_mesh
            
            # 创建场地线框
            self.create_court_lines()
            
            if self.vis:
                self.vis.add_geometry(court_mesh)
        
        except Exception as e:
            print(f"Failed to create court mesh: {e}")
    
    def create_court_lines(self):
        """创建场地线框"""
        try:
            # 场地线定义
            lines = [
                # 外边界
                [[0, 0, 1], [610, 0, 1]],      # 底线
                [[0, 670, 1], [610, 670, 1]],  # 网线
                [[0, 0, 1], [0, 670, 1]],      # 左边线
                [[610, 0, 1], [610, 670, 1]],  # 右边线
                
                # 内边界
                [[4, 4, 1], [606, 4, 1]],      # 内底线
                [[4, 666, 1], [606, 666, 1]],  # 内网线
                [[4, 4, 1], [4, 666, 1]],      # 左内线
                [[606, 4, 1], [606, 666, 1]],  # 右内线
                
                # 发球线
                [[76, 198, 1], [534, 198, 1]],
                [[76, 472, 1], [534, 472, 1]],
                
                # 中线
                [[305, 198, 1], [305, 472, 1]],
            ]
            
            # 创建线集合
            line_points = []
            line_indices = []
            
            for line in lines:
                start_idx = len(line_points)
                line_points.extend(line)
                line_indices.append([start_idx, start_idx + 1])
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector(line_indices)
            line_set.paint_uniform_color([1, 1, 1])  # 白色线
            
            self.geometry_dict['court_lines'] = line_set
            
            if self.vis:
                self.vis.add_geometry(line_set)
        
        except Exception as e:
            print(f"Failed to create court lines: {e}")
    
    def update_trajectory(self, trajectory_3d):
        """更新3D轨迹"""
        try:
            if not trajectory_3d:
                return
            
            self.trajectory_3d = trajectory_3d
            
            # 移除旧轨迹
            if 'trajectory_points' in self.geometry_dict:
                if self.vis:
                    self.vis.remove_geometry(self.geometry_dict['trajectory_points'], reset_bounding_box=False)
            
            if 'trajectory_lines' in self.geometry_dict:
                if self.vis:
                    self.vis.remove_geometry(self.geometry_dict['trajectory_lines'], reset_bounding_box=False)
            
            # 创建轨迹点
            points = np.array(trajectory_3d, dtype=np.float64)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            
            # 设置颜色渐变（从蓝到红）
            colors = []
            for i in range(len(points)):
                ratio = i / max(1, len(points) - 1)
                color = [self.trajectory_color[0] * (1 - ratio) + 1.0 * ratio,
                        self.trajectory_color[1] * (1 - ratio),
                        self.trajectory_color[2]]
                colors.append(color)
            
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            self.geometry_dict['trajectory_points'] = point_cloud
            
            # 创建轨迹线
            if len(points) > 1:
                line_indices = [[i, i + 1] for i in range(len(points) - 1)]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(line_indices)
                line_set.paint_uniform_color(self.trajectory_color)
                
                self.geometry_dict['trajectory_lines'] = line_set
                
                if self.vis and self.show_trajectory:
                    self.vis.add_geometry(line_set, reset_bounding_box=False)
            
            if self.vis and self.show_trajectory:
                self.vis.add_geometry(point_cloud, reset_bounding_box=False)
            
            self.status_label.setText(f"Trajectory updated: {len(trajectory_3d)} points")
        
        except Exception as e:
            print(f"Failed to update trajectory: {e}")
    
    def update_prediction(self, prediction_trajectory, landing_point=None):
        """更新预测轨迹"""
        try:
            self.prediction_trajectory = prediction_trajectory
            self.landing_point = landing_point
            
            # 移除旧预测
            for key in ['prediction_points', 'prediction_lines', 'landing_point']:
                if key in self.geometry_dict:
                    if self.vis:
                        self.vis.remove_geometry(self.geometry_dict[key], reset_bounding_box=False)
            
            # 创建预测轨迹
            if prediction_trajectory:
                points = np.array(prediction_trajectory, dtype=np.float64)
                
                # 预测点云
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(points)
                point_cloud.paint_uniform_color(self.prediction_color)
                self.geometry_dict['prediction_points'] = point_cloud
                
                # 预测线
                if len(points) > 1:
                    line_indices = [[i, i + 1] for i in range(len(points) - 1)]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(points)
                    line_set.lines = o3d.utility.Vector2iVector(line_indices)
                    line_set.paint_uniform_color(self.prediction_color)
                    
                    self.geometry_dict['prediction_lines'] = line_set
                    
                    if self.vis and self.show_prediction:
                        self.vis.add_geometry(line_set, reset_bounding_box=False)
                
                if self.vis and self.show_prediction:
                    self.vis.add_geometry(point_cloud, reset_bounding_box=False)
            
            # 创建落点标记
            if landing_point is not None:
                landing_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=20)
                landing_sphere.translate(landing_point)
                landing_sphere.paint_uniform_color(self.landing_color)
                
                self.geometry_dict['landing_point'] = landing_sphere
                
                if self.vis and self.show_prediction:
                    self.vis.add_geometry(landing_sphere, reset_bounding_box=False)
            
            self.status_label.setText(f"Prediction updated: {len(prediction_trajectory) if prediction_trajectory else 0} points")
        
        except Exception as e:
            print(f"Failed to update prediction: {e}")
    
    def on_display_options_changed(self):
        """显示选项变化"""
        self.show_trajectory = self.show_trajectory_cb.isChecked()
        self.show_prediction = self.show_prediction_cb.isChecked()
        self.show_court = self.show_court_cb.isChecked()
        self.show_coordinate_frame = self.show_frame_cb.isChecked()
        
        self.update_visibility()
    
    def update_visibility(self):
        """更新几何体可见性"""
        if not self.vis:
            return
        
        try:
            # 轨迹可见性
            for key in ['trajectory_points', 'trajectory_lines']:
                if key in self.geometry_dict:
                    if self.show_trajectory:
                        self.vis.add_geometry(self.geometry_dict[key], reset_bounding_box=False)
                    else:
                        self.vis.remove_geometry(self.geometry_dict[key], reset_bounding_box=False)
            
            # 预测可见性
            for key in ['prediction_points', 'prediction_lines', 'landing_point']:
                if key in self.geometry_dict:
                    if self.show_prediction:
                        self.vis.add_geometry(self.geometry_dict[key], reset_bounding_box=False)
                    else:
                        self.vis.remove_geometry(self.geometry_dict[key], reset_bounding_box=False)
            
            # 场地可见性
            for key in ['court', 'court_lines']:
                if key in self.geometry_dict:
                    if self.show_court:
                        self.vis.add_geometry(self.geometry_dict[key], reset_bounding_box=False)
                    else:
                        self.vis.remove_geometry(self.geometry_dict[key], reset_bounding_box=False)
            
            # 坐标系可见性
            if 'coordinate_frame' in self.geometry_dict:
                if self.show_coordinate_frame:
                    self.vis.add_geometry(self.geometry_dict['coordinate_frame'], reset_bounding_box=False)
                else:
                    self.vis.remove_geometry(self.geometry_dict['coordinate_frame'], reset_bounding_box=False)
        
        except Exception as e:
            print(f"Failed to update visibility: {e}")
    
    def set_view_preset(self, preset):
        """设置预设视角"""
        if not self.vis:
            return
        
        try:
            view_control = self.vis.get_view_control()
            
            if preset == "top":
                # 俯视图
                view_control.set_lookat([305, 335, 0])  # 场地中心
                view_control.set_up([0, 1, 0])
                view_control.set_front([0, 0, -1])
                view_control.set_zoom(0.3)
            
            elif preset == "side":
                # 侧视图
                view_control.set_lookat([305, 335, 100])
                view_control.set_up([0, 0, 1])
                view_control.set_front([1, 0, 0])
                view_control.set_zoom(0.5)
            
            elif preset == "front":
                # 前视图
                view_control.set_lookat([305, 335, 100])
                view_control.set_up([0, 0, 1])
                view_control.set_front([0, 1, 0])
                view_control.set_zoom(0.5)
        
        except Exception as e:
            print(f"Failed to set view preset: {e}")
    
    def reset_view(self):
        """重置视图"""
        if not self.vis:
            return
        
        try:
            view_control = self.vis.get_view_control()
            view_control.reset_camera_local_rotate()
            self.set_view_preset("top")
        
        except Exception as e:
            print(f"Failed to reset view: {e}")
    
    def clear_trajectory(self):
        """清除轨迹"""
        try:
            for key in ['trajectory_points', 'trajectory_lines']:
                if key in self.geometry_dict:
                    if self.vis:
                        self.vis.remove_geometry(self.geometry_dict[key], reset_bounding_box=False)
                    del self.geometry_dict[key]
            
            self.trajectory_3d.clear()
            self.status_label.setText("Trajectory cleared")
        
        except Exception as e:
            print(f"Failed to clear trajectory: {e}")
    
    def clear_prediction(self):
        """清除预测"""
        try:
            for key in ['prediction_points', 'prediction_lines', 'landing_point']:
                if key in self.geometry_dict:
                    if self.vis:
                        self.vis.remove_geometry(self.geometry_dict[key], reset_bounding_box=False)
                    del self.geometry_dict[key]
            
            self.prediction_trajectory.clear()
            self.landing_point = None
            self.status_label.setText("Prediction cleared")
        
        except Exception as e:
            print(f"Failed to clear prediction: {e}")
    
    def clear_all(self):
        """清除所有数据"""
        self.clear_trajectory()
        self.clear_prediction()
    
    def update_visualization(self):
        """更新可视化（定时调用）"""
        if self.vis:
            try:
                self.vis.poll_events()
                self.vis.update_renderer()
            except Exception as e:
                # 静默处理更新错误
                pass
    
    def closeEvent(self, event):
        """关闭事件"""
        if self.update_timer:
            self.update_timer.stop()
        
        if self.vis:
            try:
                self.vis.destroy_window()
            except:
                pass
        
        event.accept()


class CourtVisualizationWidget(QWidget):
    """2D场地可视化组件（俯视图）"""
    
    def __init__(self):
        super().__init__()
        
        # 场地参数
        self.court_width = 610   # cm
        self.court_length = 670  # cm
        self.scale_factor = 0.5  # 显示缩放因子
        
        # 可视化数据
        self.trajectory_2d = []
        self.prediction_2d = []
        self.landing_point_2d = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 标题
        title_label = QLabel("Court View (Top-down)")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #333;
                padding: 3px;
                background-color: #f0f0f0;
                border-radius: 3px;
            }
        """)
        layout.addWidget(title_label)
        
        # 2D视图（简化实现，实际可以使用QGraphicsView）
        self.court_label = QLabel()
        self.court_label.setMinimumSize(int(self.court_width * self.scale_factor),
                                       int(self.court_length * self.scale_factor))
        self.court_label.setStyleSheet("""
            QLabel {
                background-color: #2E7D32;
                border: 2px solid #fff;
                border-radius: 5px;
            }
        """)
        self.court_label.setText("2D Court View\n(Implementation placeholder)")
        self.court_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.court_label)
    
    def update_trajectory_2d(self, trajectory_3d):
        """更新2D轨迹显示"""
        # 投影3D轨迹到2D
        if trajectory_3d:
            self.trajectory_2d = [(point[0], point[1]) for point in trajectory_3d]
        
        # 这里可以实现具体的2D绘制逻辑
        pass
    
    def update_prediction_2d(self, prediction_3d, landing_point_3d):
        """更新2D预测显示"""
        if prediction_3d:
            self.prediction_2d = [(point[0], point[1]) for point in prediction_3d]
        
        if landing_point_3d:
            self.landing_point_2d = (landing_point_3d[0], landing_point_3d[1])
        
        # 这里可以实现具体的2D绘制逻辑
        pass