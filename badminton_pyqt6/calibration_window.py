"""
标定窗口
QDialog模态窗口，支持图像显示和点选交互
"""
import cv2
import time
import numpy as np
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QGroupBox, QTextEdit, QProgressBar,
                            QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                            QListWidget, QListWidgetItem, QSplitter, QFrame,
                            QFormLayout, QSpinBox, QComboBox, QFileDialog,
                            QMessageBox, QScrollArea)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QRectF, QPointF, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor, QFont
from calibration_core import CalibrationCore
from config import config


class InteractiveImageView(QGraphicsView):
    """交互式图像显示视图"""
    
    point_clicked = pyqtSignal(float, float)  # 点击坐标信号
    zoom_requested = pyqtSignal(float, float, float)  # 缩放请求信号 (x, y, radius)
    
    def __init__(self):
        super().__init__()
        
        # 图形场景
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # 图像项
        self.image_item = None
        self.point_items = []
        
        # 交互状态
        self.zoom_enabled = True
        self.point_selection_enabled = True
        self.current_point_index = 0
        self.point_names = ["Bottom Left", "Bottom Right", "Top Right", "Top Left"]
        
        # 样式设置
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # 背景
        self.setBackgroundBrush(QBrush(QColor(50, 50, 50)))
    
    def set_image(self, image):
        """设置显示图像"""
        if image is None:
            return
        
        # 清除现有内容
        self.scene.clear()
        self.image_item = None
        self.point_items.clear()
        
        # 转换图像格式
        if len(image.shape) == 3:
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qt_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            height, width = image.shape
            bytes_per_line = width
            qt_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        
        # 创建像素图
        pixmap = QPixmap.fromImage(qt_image)
        
        # 添加到场景
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)
        
        # 调整视图
        self.scene.setSceneRect(self.image_item.boundingRect())
        self.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)
    
    def add_point(self, x, y, name="Point", color=QColor(255, 0, 0)):
        """添加标注点"""
        if self.image_item is None:
            return
        
        # 创建点标记
        radius = 8
        pen = QPen(color, 2)
        brush = QBrush(color)
        
        # 外圆
        outer_circle = self.scene.addEllipse(
            x - radius, y - radius, 2 * radius, 2 * radius,
            pen, brush
        )
        
        # 内圆
        inner_pen = QPen(QColor(255, 255, 255), 1)
        inner_brush = QBrush(QColor(255, 255, 255))
        inner_circle = self.scene.addEllipse(
            x - radius/2, y - radius/2, radius, radius,
            inner_pen, inner_brush
        )
        
        # 文本标签
        text_item = self.scene.addText(name, QFont("Arial", 10))
        text_item.setDefaultTextColor(QColor(255, 255, 255))
        text_item.setPos(x + radius + 5, y - radius)
        
        # 添加阴影效果
        text_item.setGraphicsEffect(None)  # 简化实现
        
        # 保存点项
        point_info = {
            'x': x, 'y': y, 'name': name,
            'outer_circle': outer_circle,
            'inner_circle': inner_circle,
            'text': text_item
        }
        self.point_items.append(point_info)
        
        return len(self.point_items) - 1
    
    def clear_points(self):
        """清除所有标注点"""
        for point_info in self.point_items:
            self.scene.removeItem(point_info['outer_circle'])
            self.scene.removeItem(point_info['inner_circle'])
            self.scene.removeItem(point_info['text'])
        self.point_items.clear()
    
    def mousePressEvent(self, event):
        """鼠标点击事件"""
        if (event.button() == Qt.MouseButton.RightButton and 
            self.point_selection_enabled and self.image_item):
            
            # 转换到场景坐标
            scene_pos = self.mapToScene(event.pos())
            
            # 检查是否在图像范围内
            if self.image_item.contains(scene_pos):
                # 转换到图像坐标
                image_pos = self.image_item.mapFromScene(scene_pos)
                
                # 发射点击信号
                self.point_clicked.emit(image_pos.x(), image_pos.y())
        
        else:
            super().mousePressEvent(event)
    
    def wheelEvent(self, event):
        """鼠标滚轮缩放"""
        if self.zoom_enabled:
            # 缩放因子
            factor = 1.2
            if event.angleDelta().y() < 0:
                factor = 1.0 / factor
            
            # 以鼠标位置为中心缩放
            self.scale(factor, factor)
    
    def fit_to_view(self):
        """适应视图大小"""
        if self.image_item:
            self.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)
    
    def reset_view(self):
        """重置视图"""
        self.resetTransform()
        self.fit_to_view()


class CalibrationWorker(QThread):
    """标定处理工作线程"""
    
    # 信号定义
    progress_updated = pyqtSignal(int, str)  # (progress, message)
    result_ready = pyqtSignal(dict)  # 标定结果
    error_occurred = pyqtSignal(str)
    frame_processed = pyqtSignal(np.ndarray)  # 处理后的帧
    
    def __init__(self, calibration_core):
        super().__init__()
        self.calibration_core = calibration_core
        self.manual_points = []
        self.video_path = None
        self.processing = False
    
    def set_manual_points(self, points):
        """设置手动选择的点"""
        self.manual_points = points
    
    def set_video_path(self, path):
        """设置视频路径"""
        self.video_path = path
    
    def run(self):
        """执行标定"""
        self.processing = True
        
        try:
            self.progress_updated.emit(10, "Initializing calibration...")
            
            if not self.manual_points or len(self.manual_points) < 4:
                self.error_occurred.emit("Insufficient manual points (need at least 4)")
                return
            
            self.progress_updated.emit(30, "Processing manual points...")
            
            # 使用手动点进行标定
            rotation_matrix, tvec, rvec, matched_corners = self.calibration_core.calibrate_from_manual_points(
                self.manual_points
            )
            
            self.progress_updated.emit(70, "Saving calibration results...")
            
            # 保存结果
            output_dir = f"./calibration_results_{int(time.time())}"
            result_file = self.calibration_core.save_calibration_results(
                output_dir, rotation_matrix, tvec, rvec
            )
            
            self.progress_updated.emit(100, "Calibration completed successfully!")
            
            # 准备结果
            result = {
                'rotation_matrix': rotation_matrix,
                'translation_vector': tvec,
                'rotation_vector': rvec,
                'matched_corners': matched_corners,
                'output_file': result_file,
                'manual_points': self.manual_points
            }
            
            self.result_ready.emit(result)
            
        except Exception as e:
            self.error_occurred.emit(f"Calibration error: {str(e)}")
        
        finally:
            self.processing = False
    
    def stop(self):
        """停止处理"""
        self.processing = False


class CalibrationWindow(QDialog):
    """标定窗口主界面"""
    
    # 信号定义
    calibration_completed = pyqtSignal(dict)  # 标定完成信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 标定核心
        self.calibration_core = None
        self.calibration_worker = None
        
        # 界面状态
        self.current_image = None
        self.selected_points = []
        self.current_point_index = 0
        self.point_names = ["Bottom Left", "Bottom Right", "Top Right", "Top Left"]
        
        # 设置窗口
        self.setWindowTitle("Camera Calibration")
        self.setMinimumSize(1200, 800)
        self.setModal(True)
        
        self.setup_ui()
        self.setup_connections()
        self.reset_calibration()
    
    def setup_ui(self):
        """设置界面"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：图像显示
        self.setup_image_panel(splitter)
        
        # 右侧：控制面板
        self.setup_control_panel(splitter)
        
        # 设置分割器比例
        splitter.setSizes([800, 400])
        layout.addWidget(splitter)
    
    def setup_image_panel(self, parent):
        """设置图像显示面板"""
        # 图像面板容器
        image_frame = QFrame()
        image_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        image_layout = QVBoxLayout(image_frame)
        
        # 标题
        title_label = QLabel("Image View")
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
        image_layout.addWidget(title_label)
        
        # 图像视图
        self.image_view = InteractiveImageView()
        image_layout.addWidget(self.image_view)
        
        # 图像控制按钮
        image_control_layout = QHBoxLayout()
        
        self.frame_select_btn = QPushButton("从视频帧选择")
        self.frame_select_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        image_control_layout.addWidget(self.frame_select_btn)
        
        self.fit_view_btn = QPushButton("Fit to View")
        self.fit_view_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        image_control_layout.addWidget(self.fit_view_btn)
        
        self.clear_points_btn = QPushButton("Clear Points")
        self.clear_points_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        image_control_layout.addWidget(self.clear_points_btn)
        
        image_control_layout.addStretch()
        image_layout.addLayout(image_control_layout)
        
        parent.addWidget(image_frame)
    
    def setup_control_panel(self, parent):
        """设置控制面板"""
        # 控制面板容器
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        control_layout = QVBoxLayout(control_frame)
        
        # 配置组
        self.setup_config_group(control_layout)
        
        # 点选择组
        self.setup_point_selection_group(control_layout)
        
        # 标定控制组
        self.setup_calibration_control_group(control_layout)
        
        # 进度显示
        self.setup_progress_group(control_layout)
        
        # 结果显示
        self.setup_result_group(control_layout)
        
        control_layout.addStretch()
        parent.addWidget(control_frame)
    
    def setup_config_group(self, parent_layout):
        """设置配置组"""
        config_group = QGroupBox("Configuration")
        config_layout = QFormLayout(config_group)
        
        # 相机参数文件
        self.camera_params_path = QLabel("Not selected")
        self.camera_params_path.setStyleSheet("color: #666; font-style: italic;")
        self.camera_params_btn = QPushButton("Browse...")
        self.camera_params_btn.clicked.connect(self.select_camera_params_file)
        
        params_layout = QHBoxLayout()
        params_layout.addWidget(self.camera_params_path)
        params_layout.addWidget(self.camera_params_btn)
        config_layout.addRow("Camera Params:", params_layout)
        
        # YOLO模型文件
        self.yolo_model_path = QLabel("Not selected")
        self.yolo_model_path.setStyleSheet("color: #666; font-style: italic;")
        self.yolo_model_btn = QPushButton("Browse...")
        self.yolo_model_btn.clicked.connect(self.select_yolo_model_file)
        
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.yolo_model_path)
        model_layout.addWidget(self.yolo_model_btn)
        config_layout.addRow("YOLO Model:", model_layout)
        
        parent_layout.addWidget(config_group)
    
    def setup_point_selection_group(self, parent_layout):
        """设置点选择组"""
        point_group = QGroupBox("Point Selection")
        point_layout = QVBoxLayout(point_group)
        
        # 说明文本
        instruction_text = QLabel(
            "Right-click on the image to select court corner points in order:\n"
            "1. Bottom Left\n2. Bottom Right\n3. Top Right\n4. Top Left"
        )
        instruction_text.setStyleSheet("""
            QLabel {
                background-color: #e3f2fd;
                border: 1px solid #bbdefb;
                border-radius: 5px;
                padding: 10px;
                color: #1976d2;
            }
        """)
        point_layout.addWidget(instruction_text)
        
        # 当前点状态
        self.current_point_label = QLabel("Current: Bottom Left (1/4)")
        self.current_point_label.setStyleSheet("font-weight: bold; color: #333;")
        point_layout.addWidget(self.current_point_label)
        
        # 已选择点列表
        self.points_list = QListWidget()
        self.points_list.setMaximumHeight(120)
        point_layout.addWidget(self.points_list)
        
        parent_layout.addWidget(point_group)
    
    def setup_calibration_control_group(self, parent_layout):
        """设置标定控制组"""
        control_group = QGroupBox("Calibration Control")
        control_layout = QVBoxLayout(control_group)
        
        # 开始标定按钮
        self.start_calibration_btn = QPushButton("Start Calibration")
        self.start_calibration_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #999;
            }
        """)
        self.start_calibration_btn.setEnabled(False)
        control_layout.addWidget(self.start_calibration_btn)
        
        # 重置按钮
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #9E9E9E;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #757575;
            }
        """)
        control_layout.addWidget(self.reset_btn)
        
        parent_layout.addWidget(control_group)
    
    def setup_progress_group(self, parent_layout):
        """设置进度显示组"""
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666; font-size: 12px;")
        progress_layout.addWidget(self.status_label)
        
        parent_layout.addWidget(progress_group)
    
    def setup_result_group(self, parent_layout):
        """设置结果显示组"""
        result_group = QGroupBox("Calibration Results")
        result_layout = QVBoxLayout(result_group)
        
        # 结果文本
        self.result_text = QTextEdit()
        self.result_text.setMaximumHeight(150)
        self.result_text.setFont(QFont("Consolas", 9))
        self.result_text.setReadOnly(True)
        self.result_text.setPlainText("No calibration performed yet.")
        result_layout.addWidget(self.result_text)
        
        parent_layout.addWidget(result_group)
    
    def setup_connections(self):
        """设置信号连接"""
        # 图像视图
        self.image_view.point_clicked.connect(self.on_point_clicked)
        
        # 按钮
        self.frame_select_btn.clicked.connect(self.select_from_video_frame)
        self.fit_view_btn.clicked.connect(self.image_view.fit_to_view)
        self.clear_points_btn.clicked.connect(self.clear_all_points)
        self.start_calibration_btn.clicked.connect(self.start_calibration)
        self.reset_btn.clicked.connect(self.reset_calibration)
    
    def select_camera_params_file(self):
        """选择相机参数文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Camera Parameters File", "", "YAML Files (*.yaml *.yml)"
        )
        
        if file_path:
            self.camera_params_path.setText(file_path)
            self.camera_params_path.setStyleSheet("color: #333;")
            self.check_ready_state()
    
    def select_yolo_model_file(self):
        """选择YOLO模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model File", "", "PyTorch Files (*.pt)"
        )
        
        if file_path:
            self.yolo_model_path.setText(file_path)
            self.yolo_model_path.setStyleSheet("color: #333;")
            self.check_ready_state()
    
    def select_from_video_frame(self):
        """从视频帧选择标定图像 - 替代文件加载"""
        try:
            # 这里需要从主窗口获取当前视频帧
            # 注意：确保视频帧是原画质量
            from PyQt6.QtWidgets import QInputDialog
            
            # 简单的帧选择对话框（实际实现中应该连接到视频缓冲区）
            items = ["Camera 1 - 当前帧", "Camera 2 - 当前帧", "从缓冲区选择..."]
            item, ok = QInputDialog.getItem(
                self, "选择视频帧", "选择要用于标定的视频帧:", items, 0, False
            )
            
            if ok and item:
                # 这里应该实现从视频工作线程获取原画帧的逻辑
                # 暂时显示提示信息
                self.status_label.setText(f"从视频帧选择: {item} (需要连接视频缓冲区)")
                
                # 模拟加载一个帧的逻辑
                # 在实际实现中，这里应该：
                # 1. 从video_worker获取当前帧或缓冲帧
                # 2. 确保帧是原画质量（未压缩）
                # 3. 设置到image_view中
                
                # 临时提示
                QMessageBox.information(
                    self, "提示", 
                    "视频帧选择功能已启用。\n"
                    "请确保：\n"
                    "1. 视频正在播放或已暂停\n"
                    "2. 选择的帧质量为原画\n"
                    "3. 帧中包含清晰的球场边界"
                )
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to select video frame: {str(e)}")
    
    def on_point_clicked(self, x, y):
        """处理点击事件"""
        if self.current_point_index >= 4:
            QMessageBox.information(self, "Info", "All 4 points have been selected")
            return
        
        # 添加点到图像
        point_name = self.point_names[self.current_point_index]
        colors = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255), QColor(255, 255, 0)]
        color = colors[self.current_point_index]
        
        self.image_view.add_point(x, y, point_name, color)
        
        # 添加到选择列表
        self.selected_points.append((x, y))
        
        # 更新列表显示
        item_text = f"{self.current_point_index + 1}. {point_name}: ({x:.1f}, {y:.1f})"
        item = QListWidgetItem(item_text)
        item.setForeground(color)
        self.points_list.addItem(item)
        
        # 更新当前点
        self.current_point_index += 1
        
        if self.current_point_index < 4:
            next_point = self.point_names[self.current_point_index]
            self.current_point_label.setText(f"Current: {next_point} ({self.current_point_index + 1}/4)")
        else:
            self.current_point_label.setText("All points selected!")
            self.current_point_label.setStyleSheet("font-weight: bold; color: green;")
        
        self.check_ready_state()
    
    def clear_all_points(self):
        """清除所有点"""
        self.image_view.clear_points()
        self.selected_points.clear()
        self.points_list.clear()
        self.current_point_index = 0
        self.current_point_label.setText("Current: Bottom Left (1/4)")
        self.current_point_label.setStyleSheet("font-weight: bold; color: #333;")
        self.check_ready_state()
    
    def check_ready_state(self):
        """检查是否准备好开始标定"""
        camera_params_ready = (self.camera_params_path.text() != "Not selected" and 
                              self.camera_params_path.text() != "")
        yolo_model_ready = (self.yolo_model_path.text() != "Not selected" and 
                           self.yolo_model_path.text() != "")
        image_ready = self.current_image is not None
        points_ready = len(self.selected_points) >= 4
        
        self.start_calibration_btn.setEnabled(
            camera_params_ready and yolo_model_ready and image_ready and points_ready
        )
    
    def start_calibration(self):
        """开始标定"""
        try:
            # 创建标定核心
            camera_params_file = self.camera_params_path.text()
            yolo_model_path = self.yolo_model_path.text()
            
            self.calibration_core = CalibrationCore(camera_params_file, yolo_model_path)
            
            # 创建工作线程
            self.calibration_worker = CalibrationWorker(self.calibration_core)
            self.calibration_worker.set_manual_points(self.selected_points)
            
            # 连接信号
            self.calibration_worker.progress_updated.connect(self.on_progress_updated)
            self.calibration_worker.result_ready.connect(self.on_calibration_completed)
            self.calibration_worker.error_occurred.connect(self.on_calibration_error)
            
            # 禁用界面
            self.start_calibration_btn.setEnabled(False)
            self.reset_btn.setEnabled(False)
            
            # 开始处理
            self.calibration_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start calibration: {str(e)}")
    
    def on_progress_updated(self, progress, message):
        """更新进度"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
    
    def on_calibration_completed(self, result):
        """标定完成"""
        # 恢复界面
        self.start_calibration_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        
        # 显示结果
        result_text = "Calibration completed successfully!\n\n"
        result_text += f"Output file: {result['output_file']}\n"
        result_text += f"Manual points used: {len(result['manual_points'])}\n"
        result_text += f"Matched corners: {len(result['matched_corners'])}\n\n"
        
        # 旋转矩阵
        rotation_matrix = result['rotation_matrix']
        result_text += "Rotation Matrix:\n"
        for row in rotation_matrix:
            result_text += "  " + "  ".join(f"{val:8.4f}" for val in row) + "\n"
        
        # 平移向量
        translation_vector = result['translation_vector'].flatten()
        result_text += f"\nTranslation Vector:\n"
        result_text += "  " + "  ".join(f"{val:8.4f}" for val in translation_vector) + "\n"
        
        self.result_text.setPlainText(result_text)
        
        # 在图像上绘制场地线
        if self.current_image is not None:
            try:
                court_image = self.calibration_core.draw_court_lines(
                    self.current_image, rotation_matrix, result['translation_vector']
                )
                self.image_view.set_image(court_image)
            except Exception as e:
                print(f"Failed to draw court lines: {e}")
        
        # 发射完成信号
        self.calibration_completed.emit(result)
        
        QMessageBox.information(self, "Success", "Calibration completed successfully!")
    
    def on_calibration_error(self, error_message):
        """标定错误"""
        # 恢复界面
        self.start_calibration_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        
        # 重置进度
        self.progress_bar.setValue(0)
        self.status_label.setText("Error occurred")
        
        # 显示错误
        QMessageBox.critical(self, "Calibration Error", error_message)
    
    def reset_calibration(self):
        """重置标定"""
        # 停止工作线程
        if self.calibration_worker and self.calibration_worker.isRunning():
            self.calibration_worker.stop()
            self.calibration_worker.wait(3000)
        
        # 重置界面
        self.clear_all_points()
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
        self.result_text.setPlainText("No calibration performed yet.")
        
        # 重置图像
        if self.current_image is not None:
            self.image_view.set_image(self.current_image)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止工作线程
        if self.calibration_worker and self.calibration_worker.isRunning():
            self.calibration_worker.stop()
            self.calibration_worker.wait(3000)
        
        event.accept()