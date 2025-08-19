"""
控制面板组件
系统状态显示、参数调整、调试信息展示
"""
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QPushButton, QGroupBox, QTextEdit, QSlider,
                            QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
                            QProgressBar, QTabWidget, QFormLayout, QSplitter)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPalette


class SystemStatusWidget(QWidget):
    """系统状态显示组件"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
        # 状态更新定时器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(500)  # 每500ms更新一次
    
    def setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 相机状态
        camera_group = QGroupBox("Camera Status")
        camera_layout = QFormLayout(camera_group)
        
        self.camera1_status = QLabel("Disconnected")
        self.camera1_status.setStyleSheet("color: red; font-weight: bold;")
        camera_layout.addRow("Camera 1:", self.camera1_status)
        
        self.camera2_status = QLabel("Disconnected")
        self.camera2_status.setStyleSheet("color: red; font-weight: bold;")
        camera_layout.addRow("Camera 2:", self.camera2_status)
        
        layout.addWidget(camera_group)
        
        # 检测状态
        detection_group = QGroupBox("Detection Status")
        detection_layout = QFormLayout(detection_group)
        
        self.shuttlecock_status = QLabel("Not Detected")
        self.shuttlecock_status.setStyleSheet("color: red; font-weight: bold;")
        detection_layout.addRow("Shuttlecock:", self.shuttlecock_status)
        
        self.detection_count = QLabel("0")
        detection_layout.addRow("Detection Count:", self.detection_count)
        
        self.detection_fps = QLabel("0.0")
        detection_layout.addRow("Detection FPS:", self.detection_fps)
        
        layout.addWidget(detection_group)
        
        # 轨迹状态
        trajectory_group = QGroupBox("Trajectory Status")
        trajectory_layout = QFormLayout(trajectory_group)
        
        self.trajectory_points = QLabel("0")
        trajectory_layout.addRow("Trajectory Points:", self.trajectory_points)
        
        self.trajectory_3d_points = QLabel("0")
        trajectory_layout.addRow("3D Points:", self.trajectory_3d_points)
        
        self.trajectory_quality = QProgressBar()
        self.trajectory_quality.setMaximum(100)
        self.trajectory_quality.setValue(0)
        trajectory_layout.addRow("Quality:", self.trajectory_quality)
        
        layout.addWidget(trajectory_group)
        
        # 预测状态
        prediction_group = QGroupBox("Prediction Status")
        prediction_layout = QFormLayout(prediction_group)
        
        self.prediction_status = QLabel("Idle")
        self.prediction_status.setStyleSheet("color: gray; font-weight: bold;")
        prediction_layout.addRow("Status:", self.prediction_status)
        
        self.prediction_confidence = QProgressBar()
        self.prediction_confidence.setMaximum(100)
        self.prediction_confidence.setValue(0)
        prediction_layout.addRow("Confidence:", self.prediction_confidence)
        
        self.landing_point = QLabel("N/A")
        prediction_layout.addRow("Landing Point:", self.landing_point)
        
        self.in_bounds = QLabel("N/A")
        prediction_layout.addRow("In Bounds:", self.in_bounds)
        
        layout.addWidget(prediction_group)
        
        layout.addStretch()
    
    def update_camera_status(self, camera_id, connected, fps=None):
        """更新相机状态"""
        status_label = self.camera1_status if camera_id == 0 else self.camera2_status
        
        if connected:
            text = "Connected"
            if fps is not None:
                text += f" ({fps:.1f} FPS)"
            status_label.setText(text)
            status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            status_label.setText("Disconnected")
            status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def update_detection_status(self, detected, count=None, fps=None):
        """更新检测状态"""
        if detected:
            self.shuttlecock_status.setText("Detected")
            self.shuttlecock_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.shuttlecock_status.setText("Not Detected")
            self.shuttlecock_status.setStyleSheet("color: red; font-weight: bold;")
        
        if count is not None:
            self.detection_count.setText(str(count))
        
        if fps is not None:
            self.detection_fps.setText(f"{fps:.1f}")
    
    def update_trajectory_status(self, points_2d=0, points_3d=0, quality=0):
        """更新轨迹状态"""
        self.trajectory_points.setText(str(points_2d))
        self.trajectory_3d_points.setText(str(points_3d))
        self.trajectory_quality.setValue(int(quality * 100))
    
    def update_prediction_status(self, status, confidence=0, landing_point=None, in_bounds=None):
        """更新预测状态"""
        status_colors = {
            'idle': 'gray',
            'ready': 'orange',
            'processing': 'blue',
            'completed': 'green',
            'error': 'red'
        }
        
        color = status_colors.get(status.lower(), 'gray')
        self.prediction_status.setText(status.title())
        self.prediction_status.setStyleSheet(f"color: {color}; font-weight: bold;")
        
        self.prediction_confidence.setValue(int(confidence * 100))
        
        if landing_point is not None:
            self.landing_point.setText(f"({landing_point[0]:.1f}, {landing_point[1]:.1f})")
        else:
            self.landing_point.setText("N/A")
        
        if in_bounds is not None:
            text = "Yes" if in_bounds else "No"
            color = "green" if in_bounds else "red"
            self.in_bounds.setText(text)
            self.in_bounds.setStyleSheet(f"color: {color}; font-weight: bold;")
        else:
            self.in_bounds.setText("N/A")
            self.in_bounds.setStyleSheet("color: gray; font-weight: bold;")
    
    def update_display(self):
        """定期更新显示"""
        pass  # 由外部调用具体的更新方法


class ParameterControlWidget(QWidget):
    """参数控制组件"""
    
    # 参数变化信号
    parameter_changed = pyqtSignal(str, object)  # (parameter_name, value)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 检测参数
        detection_group = QGroupBox("Detection Parameters")
        detection_layout = QFormLayout(detection_group)
        
        self.confidence_threshold = QDoubleSpinBox()
        self.confidence_threshold.setRange(0.1, 1.0)
        self.confidence_threshold.setSingleStep(0.05)
        self.confidence_threshold.setValue(0.3)
        self.confidence_threshold.setDecimals(2)
        detection_layout.addRow("Confidence:", self.confidence_threshold)
        
        self.iou_threshold = QDoubleSpinBox()
        self.iou_threshold.setRange(0.1, 1.0)
        self.iou_threshold.setSingleStep(0.05)
        self.iou_threshold.setValue(0.5)
        self.iou_threshold.setDecimals(2)
        detection_layout.addRow("IoU Threshold:", self.iou_threshold)
        
        layout.addWidget(detection_group)
        
        # 轨迹参数
        trajectory_group = QGroupBox("Trajectory Parameters")
        trajectory_layout = QFormLayout(trajectory_group)
        
        self.buffer_size = QSpinBox()
        self.buffer_size.setRange(50, 1000)
        self.buffer_size.setSingleStep(10)
        self.buffer_size.setValue(300)
        trajectory_layout.addRow("Buffer Size:", self.buffer_size)
        
        self.smoothing_window = QSpinBox()
        self.smoothing_window.setRange(1, 20)
        self.smoothing_window.setValue(5)
        trajectory_layout.addRow("Smoothing Window:", self.smoothing_window)
        
        layout.addWidget(trajectory_group)
        
        # 预测参数
        prediction_group = QGroupBox("Prediction Parameters")
        prediction_layout = QFormLayout(prediction_group)
        
        self.prediction_method = QComboBox()
        self.prediction_method.addItems(["Ensemble", "EKF", "Polynomial", "Physics"])
        prediction_layout.addRow("Method:", self.prediction_method)
        
        self.time_window = QDoubleSpinBox()
        self.time_window.setRange(0.5, 5.0)
        self.time_window.setSingleStep(0.1)
        self.time_window.setValue(2.0)
        self.time_window.setDecimals(1)
        prediction_layout.addRow("Time Window (s):", self.time_window)
        
        self.poly_degree = QSpinBox()
        self.poly_degree.setRange(2, 6)
        self.poly_degree.setValue(4)
        prediction_layout.addRow("Polynomial Degree:", self.poly_degree)
        
        layout.addWidget(prediction_group)
        
        # 物理参数
        physics_group = QGroupBox("Physics Parameters")
        physics_layout = QFormLayout(physics_group)
        
        self.shuttlecock_mass = QDoubleSpinBox()
        self.shuttlecock_mass.setRange(0.001, 0.010)
        self.shuttlecock_mass.setSingleStep(0.0001)
        self.shuttlecock_mass.setValue(0.005)
        self.shuttlecock_mass.setDecimals(4)
        physics_layout.addRow("Mass (kg):", self.shuttlecock_mass)
        
        self.drag_coefficient = QDoubleSpinBox()
        self.drag_coefficient.setRange(0.1, 2.0)
        self.drag_coefficient.setSingleStep(0.05)
        self.drag_coefficient.setValue(0.6)
        self.drag_coefficient.setDecimals(2)
        physics_layout.addRow("Drag Coefficient:", self.drag_coefficient)
        
        self.air_density = QDoubleSpinBox()
        self.air_density.setRange(1.0, 1.5)
        self.air_density.setSingleStep(0.01)
        self.air_density.setValue(1.225)
        self.air_density.setDecimals(3)
        physics_layout.addRow("Air Density:", self.air_density)
        
        layout.addWidget(physics_group)
        
        # 重置按钮
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_parameters)
        layout.addWidget(reset_btn)
        
        layout.addStretch()
    
    def setup_connections(self):
        """设置信号连接"""
        self.confidence_threshold.valueChanged.connect(
            lambda v: self.parameter_changed.emit('confidence_threshold', v))
        self.iou_threshold.valueChanged.connect(
            lambda v: self.parameter_changed.emit('iou_threshold', v))
        self.buffer_size.valueChanged.connect(
            lambda v: self.parameter_changed.emit('buffer_size', v))
        self.smoothing_window.valueChanged.connect(
            lambda v: self.parameter_changed.emit('smoothing_window', v))
        self.prediction_method.currentTextChanged.connect(
            lambda v: self.parameter_changed.emit('prediction_method', v.lower()))
        self.time_window.valueChanged.connect(
            lambda v: self.parameter_changed.emit('time_window', v))
        self.poly_degree.valueChanged.connect(
            lambda v: self.parameter_changed.emit('poly_degree', v))
        self.shuttlecock_mass.valueChanged.connect(
            lambda v: self.parameter_changed.emit('shuttlecock_mass', v))
        self.drag_coefficient.valueChanged.connect(
            lambda v: self.parameter_changed.emit('drag_coefficient', v))
        self.air_density.valueChanged.connect(
            lambda v: self.parameter_changed.emit('air_density', v))
    
    def reset_parameters(self):
        """重置参数为默认值"""
        self.confidence_threshold.setValue(0.3)
        self.iou_threshold.setValue(0.5)
        self.buffer_size.setValue(300)
        self.smoothing_window.setValue(5)
        self.prediction_method.setCurrentText("Ensemble")
        self.time_window.setValue(2.0)
        self.poly_degree.setValue(4)
        self.shuttlecock_mass.setValue(0.005)
        self.drag_coefficient.setValue(0.6)
        self.air_density.setValue(1.225)
    
    def get_parameters(self):
        """获取当前参数值"""
        return {
            'confidence_threshold': self.confidence_threshold.value(),
            'iou_threshold': self.iou_threshold.value(),
            'buffer_size': self.buffer_size.value(),
            'smoothing_window': self.smoothing_window.value(),
            'prediction_method': self.prediction_method.currentText().lower(),
            'time_window': self.time_window.value(),
            'poly_degree': self.poly_degree.value(),
            'shuttlecock_mass': self.shuttlecock_mass.value(),
            'drag_coefficient': self.drag_coefficient.value(),
            'air_density': self.air_density.value()
        }


class DebugInfoWidget(QWidget):
    """调试信息显示组件"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 日志显示
        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(200)
        self.log_display.setFont(QFont("Consolas", 9))
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """)
        log_layout.addWidget(self.log_display)
        
        # 日志控制按钮
        log_control_layout = QHBoxLayout()
        
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.clear_log)
        log_control_layout.addWidget(clear_log_btn)
        
        self.auto_scroll = QCheckBox("Auto Scroll")
        self.auto_scroll.setChecked(True)
        log_control_layout.addWidget(self.auto_scroll)
        
        log_control_layout.addStretch()
        
        log_layout.addLayout(log_control_layout)
        layout.addWidget(log_group)
        
        # 性能信息
        performance_group = QGroupBox("Performance")
        performance_layout = QFormLayout(performance_group)
        
        self.cpu_usage = QProgressBar()
        self.cpu_usage.setMaximum(100)
        performance_layout.addRow("CPU Usage:", self.cpu_usage)
        
        self.memory_usage = QProgressBar()
        self.memory_usage.setMaximum(100)
        performance_layout.addRow("Memory Usage:", self.memory_usage)
        
        self.processing_time = QLabel("0 ms")
        performance_layout.addRow("Processing Time:", self.processing_time)
        
        layout.addWidget(performance_group)
        
        layout.addStretch()
    
    def add_log_message(self, message, level="INFO"):
        """添加日志消息"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # 根据级别设置颜色
        colors = {
            "INFO": "black",
            "WARNING": "orange",
            "ERROR": "red",
            "DEBUG": "blue"
        }
        
        color = colors.get(level, "black")
        formatted_message = f'<span style="color: {color};">[{timestamp}] {level}: {message}</span>'
        
        self.log_display.append(formatted_message)
        
        if self.auto_scroll.isChecked():
            scrollbar = self.log_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def clear_log(self):
        """清空日志"""
        self.log_display.clear()
    
    def update_performance(self, cpu_percent=0, memory_percent=0, processing_time_ms=0):
        """更新性能信息"""
        self.cpu_usage.setValue(int(cpu_percent))
        self.memory_usage.setValue(int(memory_percent))
        self.processing_time.setText(f"{processing_time_ms:.1f} ms")


class ControlPanel(QDockWidget):
    """控制面板主窗口组件"""
    
    # 信号定义
    parameter_changed = pyqtSignal(str, object)
    start_calibration = pyqtSignal()
    start_prediction = pyqtSignal()
    reset_trajectory = pyqtSignal()
    
    def __init__(self, title="Control Panel"):
        super().__init__(title)
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        """设置界面"""
        # 创建主widget
        main_widget = QWidget()
        self.setWidget(main_widget)
        
        # 主布局
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 创建选项卡
        tab_widget = QTabWidget()
        
        # 状态标签页
        self.status_widget = SystemStatusWidget()
        tab_widget.addTab(self.status_widget, "Status")
        
        # 参数标签页
        self.parameter_widget = ParameterControlWidget()
        tab_widget.addTab(self.parameter_widget, "Parameters")
        
        # 调试标签页
        self.debug_widget = DebugInfoWidget()
        tab_widget.addTab(self.debug_widget, "Debug")
        
        layout.addWidget(tab_widget)
        
        # 控制按钮
        control_layout = QVBoxLayout()
        control_layout.setSpacing(5)
        
        # 标定按钮
        self.calibration_btn = QPushButton("Start Calibration")
        self.calibration_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        control_layout.addWidget(self.calibration_btn)
        
        # 预测按钮
        self.prediction_btn = QPushButton("Start Prediction")
        self.prediction_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        control_layout.addWidget(self.prediction_btn)
        
        # 重置按钮
        self.reset_btn = QPushButton("Reset Trajectory")
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
        
        layout.addLayout(control_layout)
        
        # 设置停靠属性
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | 
                            Qt.DockWidgetArea.RightDockWidgetArea)
        self.setMinimumWidth(300)
    
    def setup_connections(self):
        """设置信号连接"""
        self.parameter_widget.parameter_changed.connect(self.parameter_changed.emit)
        self.calibration_btn.clicked.connect(self.start_calibration.emit)
        self.prediction_btn.clicked.connect(self.start_prediction.emit)
        self.reset_btn.clicked.connect(self.reset_trajectory.emit)
    
    def update_status(self, **kwargs):
        """更新状态显示"""
        if 'camera_status' in kwargs:
            camera_id, connected, fps = kwargs['camera_status']
            self.status_widget.update_camera_status(camera_id, connected, fps)
        
        if 'detection_status' in kwargs:
            detected, count, fps = kwargs['detection_status']
            self.status_widget.update_detection_status(detected, count, fps)
        
        if 'trajectory_status' in kwargs:
            points_2d, points_3d, quality = kwargs['trajectory_status']
            self.status_widget.update_trajectory_status(points_2d, points_3d, quality)
        
        if 'prediction_status' in kwargs:
            status, confidence, landing_point, in_bounds = kwargs['prediction_status']
            self.status_widget.update_prediction_status(status, confidence, landing_point, in_bounds)
    
    def add_log_message(self, message, level="INFO"):
        """添加日志消息"""
        self.debug_widget.add_log_message(message, level)
    
    def update_performance(self, cpu_percent=0, memory_percent=0, processing_time_ms=0):
        """更新性能信息"""
        self.debug_widget.update_performance(cpu_percent, memory_percent, processing_time_ms)