"""
主窗口界面
QMainWindow为基础，Material Design风格
"""
import sys
import os
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QMenuBar, QMenu, QStatusBar, QSplitter, QLabel,
                            QFileDialog, QMessageBox, QApplication, QToolBar,
                            QProgressBar, QFrame, QDockWidget, QPushButton,
                            QGroupBox, QFormLayout, QDoubleSpinBox, QSlider,
                            QLineEdit)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, QSettings
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QPixmap
import time

# 导入自定义组件
from video_widget import DualVideoWidget
from control_panel import ControlPanel
from calibration_window import CalibrationWindow
from visualization_3d import Visualization3DWidget
from video_worker import DualVideoWorker
from detection_worker import DetectionWorker, StereoDetectionWorker
from prediction_worker import PredictionWorker
from speed_detector import SpeedDetector
from config import config
from utils import logger, SystemUtils, DialogUtils
from styles import MAIN_STYLE, DARK_THEME


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        
        # 窗口基本设置
        self.setWindowTitle("Badminton Landing Prediction System - PyQt6")
        self.setMinimumSize(1400, 900)
        
        # 工作线程
        self.video_worker = None
        self.detection_worker1 = None
        self.detection_worker2 = None
        self.stereo_worker = None
        self.prediction_worker = None
        self.speed_detector = SpeedDetector()
        
        # 状态
        self.system_initialized = False
        self.calibration_completed = False
        self.video_loaded = False
        self.processing_active = False
        self.is_paused = False
        
        # 性能监控
        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self.update_performance_stats)
        self.performance_timer.start(2000)  # 每2秒更新
        
        # 设置界面
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_tool_bar()
        self.setup_status_bar()
        self.setup_connections()
        
        # 应用样式
        self.apply_theme("light")
        
        # 加载设置
        self.load_settings()
        
        # 初始化工作线程
        self.initialize_workers()
        
        logger.info("Main window initialized")
    
    def setup_ui(self):
        """设置主界面 - 改进的多窗口多标签页设计"""
        # 中央窗口
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局 - 使用标签页设计
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # 创建标签页管理器
        from PyQt6.QtWidgets import QTabWidget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # 主监控标签页
        self.setup_main_tab()
        
        # 标定标签页
        self.setup_calibration_tab()
        
        # 分析标签页
        self.setup_analysis_tab()
        
        # 创建停靠面板
        self.setup_dock_widgets()
    
    def setup_main_tab(self):
        """设置主监控标签页 - 优化布局：大视频区域，参数在底部"""
        main_tab = QWidget()
        layout = QVBoxLayout(main_tab)  # 改为垂直布局
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # 主要区域：视频显示 (占绝大部分空间)
        self.setup_video_area_large(layout)
        
        # 底部区域：控制参数 (紧凑布局)
        self.setup_control_parameters_bottom(layout)
        
        self.tab_widget.addTab(main_tab, "主监控")
    
    def setup_calibration_tab(self):
        """设置标定标签页 - 完整的标定功能界面"""
        calibration_tab = QWidget()
        layout = QVBoxLayout(calibration_tab)
        
        # 标定控制面板标题
        calibration_label = QLabel("相机标定与配置")
        calibration_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(calibration_label)
        
        # 创建水平分割器：左侧标定区域，右侧控制面板
        calibration_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：标定操作区域
        self.setup_calibration_work_area(calibration_splitter)
        
        # 右侧：标定控制面板
        self.setup_calibration_control_panel(calibration_splitter)
        
        # 设置分割器比例
        calibration_splitter.setSizes([800, 400])
        layout.addWidget(calibration_splitter)
        
        self.tab_widget.addTab(calibration_tab, "相机标定")

    def setup_calibration_work_area(self, parent):
        """设置标定工作区域"""
        work_frame = QFrame()
        work_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        work_layout = QVBoxLayout(work_frame)
        work_layout.setContentsMargins(5, 5, 5, 5)
        
        # 说明文字
        info_label = QLabel("标定说明：只能从缓冲视频帧中选择图片进行标定，确保视频帧为原画质量。")
        info_label.setStyleSheet("color: #666; font-style: italic; padding: 10px; background-color: #f8f8f8; border-radius: 5px;")
        work_layout.addWidget(info_label)
        
        # 帧选择区域
        frame_selection_group = QGroupBox("视频帧选择")
        frame_selection_layout = QVBoxLayout(frame_selection_group)
        
        # 帧缓冲显示
        self.frame_buffer_label = QLabel("当前缓冲帧数: 0")
        frame_selection_layout.addWidget(self.frame_buffer_label)
        
        # 帧选择控件
        frame_control_layout = QHBoxLayout()
        
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        frame_control_layout.addWidget(self.frame_slider)
        
        self.frame_number_label = QLabel("0 / 0")
        frame_control_layout.addWidget(self.frame_number_label)
        
        frame_selection_layout.addLayout(frame_control_layout)
        
        # 选择帧按钮
        select_frame_layout = QHBoxLayout()
        
        self.select_frame1_btn = QPushButton("选择作为Camera1标定帧")
        self.select_frame1_btn.setEnabled(False)
        select_frame_layout.addWidget(self.select_frame1_btn)
        
        self.select_frame2_btn = QPushButton("选择作为Camera2标定帧")
        self.select_frame2_btn.setEnabled(False)
        select_frame_layout.addWidget(self.select_frame2_btn)
        
        frame_selection_layout.addLayout(select_frame_layout)
        
        work_layout.addWidget(frame_selection_group)
        
        # 当前标定帧显示
        current_frame_group = QGroupBox("当前标定帧")
        current_frame_layout = QHBoxLayout(current_frame_group)
        
        # Camera 1标定帧
        cam1_frame = QFrame()
        cam1_layout = QVBoxLayout(cam1_frame)
        cam1_layout.addWidget(QLabel("Camera 1"))
        self.calib_frame1_label = QLabel("未选择")
        self.calib_frame1_label.setMinimumSize(300, 200)
        self.calib_frame1_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        self.calib_frame1_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cam1_layout.addWidget(self.calib_frame1_label)
        current_frame_layout.addWidget(cam1_frame)
        
        # Camera 2标定帧
        cam2_frame = QFrame()
        cam2_layout = QVBoxLayout(cam2_frame)
        cam2_layout.addWidget(QLabel("Camera 2"))
        self.calib_frame2_label = QLabel("未选择")
        self.calib_frame2_label.setMinimumSize(300, 200)
        self.calib_frame2_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        self.calib_frame2_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cam2_layout.addWidget(self.calib_frame2_label)
        current_frame_layout.addWidget(cam2_frame)
        
        work_layout.addWidget(current_frame_group)
        
        parent.addWidget(work_frame)

    def setup_calibration_control_panel(self, parent):
        """设置标定控制面板"""
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        control_layout = QVBoxLayout(control_frame)
        control_layout.setContentsMargins(5, 5, 5, 5)
        
        # 标定参数设置
        params_group = QGroupBox("标定参数")
        params_layout = QFormLayout(params_group)
        
        # 相机参数文件路径
        self.camera_params_path_edit = QLineEdit()
        self.camera_params_path_edit.setPlaceholderText("选择相机参数文件...")
        params_layout.addRow("相机参数:", self.camera_params_path_edit)
        
        browse_params_btn = QPushButton("浏览...")
        browse_params_btn.clicked.connect(self.browse_camera_params_file)
        params_layout.addRow("", browse_params_btn)
        
        # YOLO模型文件路径
        self.yolo_model_path_edit = QLineEdit()
        self.yolo_model_path_edit.setPlaceholderText("选择YOLO模型文件...")
        params_layout.addRow("YOLO模型:", self.yolo_model_path_edit)
        
        browse_yolo_btn = QPushButton("浏览...")
        browse_yolo_btn.clicked.connect(self.browse_yolo_model_file)
        params_layout.addRow("", browse_yolo_btn)
        
        control_layout.addWidget(params_group)
        
        # 标定操作
        action_group = QGroupBox("标定操作")
        action_layout = QVBoxLayout(action_group)
        
        # 自动检测角点
        self.auto_detect_btn = QPushButton("自动检测角点")
        self.auto_detect_btn.setEnabled(False)
        action_layout.addWidget(self.auto_detect_btn)
        
        # 手动标定
        self.manual_calib_btn = QPushButton("手动标定")
        self.manual_calib_btn.setEnabled(False)
        action_layout.addWidget(self.manual_calib_btn)
        
        # 开始标定
        self.start_calibration_btn = QPushButton("开始标定")
        self.start_calibration_btn.setEnabled(False)
        self.start_calibration_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        action_layout.addWidget(self.start_calibration_btn)
        
        # 重置标定
        self.reset_calibration_btn = QPushButton("重置标定")
        action_layout.addWidget(self.reset_calibration_btn)
        
        control_layout.addWidget(action_group)
        
        # 标定状态
        status_group = QGroupBox("标定状态")
        status_layout = QVBoxLayout(status_group)
        
        self.calibration_status_label = QLabel("未开始")
        self.calibration_status_label.setStyleSheet("color: #666;")
        status_layout.addWidget(self.calibration_status_label)
        
        self.calibration_progress = QProgressBar()
        self.calibration_progress.setVisible(False)
        status_layout.addWidget(self.calibration_progress)
        
        control_layout.addWidget(status_group)
        
        control_layout.addStretch()
        
        parent.addWidget(control_frame)
    
    def setup_analysis_tab(self):
        """设置分析标签页 - 包含3D可视化和轨迹分析"""
        analysis_tab = QWidget()
        layout = QVBoxLayout(analysis_tab)
        
        # 分析控制面板标题
        analysis_label = QLabel("三维轨迹可视化与分析")
        analysis_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(analysis_label)
        
        # 创建水平分割器：左侧3D可视化，右侧控制面板
        analysis_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：3D可视化区域 (主要空间)
        self.setup_3d_visualization_area(analysis_splitter)
        
        # 右侧：分析控制面板
        self.setup_analysis_control_panel(analysis_splitter)
        
        # 设置分割器比例
        analysis_splitter.setSizes([1000, 300])
        layout.addWidget(analysis_splitter)
        
        self.tab_widget.addTab(analysis_tab, "轨迹分析")

    def setup_3d_visualization_area(self, parent):
        """设置3D可视化区域"""
        viz_frame = QFrame()
        viz_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        viz_layout = QVBoxLayout(viz_frame)
        viz_layout.setContentsMargins(5, 5, 5, 5)
        
        # 3D可视化标题
        viz_title = QLabel("3D轨迹可视化")
        viz_title.setStyleSheet("font-weight: bold; padding: 5px; font-size: 14px;")
        viz_layout.addWidget(viz_title)
        
        # 3D可视化组件 - 占用大部分空间
        self.viz_3d_widget = Visualization3DWidget()
        viz_layout.addWidget(self.viz_3d_widget)
        
        parent.addWidget(viz_frame)

    def setup_analysis_control_panel(self, parent):
        """设置分析控制面板"""
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        control_layout = QVBoxLayout(control_frame)
        control_layout.setContentsMargins(5, 5, 5, 5)
        
        # 3D可视化控制
        viz_control_group = QGroupBox("可视化控制")
        viz_control_layout = QVBoxLayout(viz_control_group)
        
        # 显示设置
        self.show_trajectory_btn = QPushButton("显示轨迹")
        self.show_trajectory_btn.setCheckable(True)
        self.show_trajectory_btn.setChecked(True)
        viz_control_layout.addWidget(self.show_trajectory_btn)
        
        self.show_prediction_btn = QPushButton("显示预测")
        self.show_prediction_btn.setCheckable(True)
        viz_control_layout.addWidget(self.show_prediction_btn)
        
        self.show_court_btn = QPushButton("显示场地")
        self.show_court_btn.setCheckable(True)
        self.show_court_btn.setChecked(True)
        viz_control_layout.addWidget(self.show_court_btn)
        
        # 重置视图
        self.reset_view_btn = QPushButton("重置视图")
        viz_control_layout.addWidget(self.reset_view_btn)
        
        control_layout.addWidget(viz_control_group)
        
        # 轨迹分析
        analysis_group = QGroupBox("轨迹分析")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.max_speed_label = QLabel("最大速度: -- km/h")
        analysis_layout.addWidget(self.max_speed_label)
        
        self.flight_time_label = QLabel("飞行时间: -- s")
        analysis_layout.addWidget(self.flight_time_label)
        
        self.trajectory_points_label = QLabel("轨迹点数: 0")
        analysis_layout.addWidget(self.trajectory_points_label)
        
        # 清除轨迹按钮
        self.clear_trajectory_btn = QPushButton("清除轨迹")
        analysis_layout.addWidget(self.clear_trajectory_btn)
        
        control_layout.addWidget(analysis_group)
        
        control_layout.addStretch()
        
        parent.addWidget(control_frame)
    
    def setup_video_area_large(self, parent_layout):
        """设置大视频显示区域 - 占据主要空间"""
        # 视频区域容器
        video_frame = QFrame()
        video_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(5, 5, 5, 5)
        
        # 视频标题
        video_title = QLabel("双路视频监控")
        video_title.setStyleSheet("font-weight: bold; padding: 5px; font-size: 14px;")
        video_layout.addWidget(video_title)
        
        # 视频显示组件 - 设置为更大尺寸
        self.video_widget = DualVideoWidget()
        # 设置更大的尺寸让视频占据更多空间
        self.video_widget.setMinimumSize(1600, 600)  # 增大视频显示区域
        video_layout.addWidget(self.video_widget)
        
        parent_layout.addWidget(video_frame, 4)  # 给视频区域分配更多比重 (4/5)

    def setup_control_parameters_bottom(self, parent_layout):
        """设置底部控制参数区域 - 紧凑布局"""
        # 控制参数容器
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        control_frame.setMaximumHeight(150)  # 限制高度，保持紧凑
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(5, 5, 5, 5)
        
        # 视频控制区域
        video_control_group = QGroupBox("视频控制")
        video_control_layout = QHBoxLayout(video_control_group)
        
        # 播放控制按钮
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.on_pause_clicked)
        self.pause_btn.setFixedSize(80, 30)
        video_control_layout.addWidget(self.pause_btn)
        
        self.predict_btn = QPushButton("开始预测")
        self.predict_btn.setEnabled(False)
        self.predict_btn.clicked.connect(self.on_predict_clicked)
        self.predict_btn.setFixedSize(80, 30)
        video_control_layout.addWidget(self.predict_btn)
        
        self.speed_detect_btn = QPushButton("速度检测")
        self.speed_detect_btn.setEnabled(False)
        self.speed_detect_btn.clicked.connect(self.on_speed_detect_clicked)
        self.speed_detect_btn.setFixedSize(80, 30)
        video_control_layout.addWidget(self.speed_detect_btn)
        
        control_layout.addWidget(video_control_group)
        
        # 检测参数区域
        detection_group = QGroupBox("检测参数")
        detection_layout = QFormLayout(detection_group)
        detection_layout.setVerticalSpacing(2)  # 紧凑间距
        
        # 置信度设置
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.1, 1.0)
        self.confidence_spinbox.setSingleStep(0.1)
        self.confidence_spinbox.setValue(0.5)
        self.confidence_spinbox.setFixedWidth(80)
        detection_layout.addRow("置信度:", self.confidence_spinbox)
        
        # NMS阈值
        self.nms_spinbox = QDoubleSpinBox()
        self.nms_spinbox.setRange(0.1, 1.0)
        self.nms_spinbox.setSingleStep(0.1)
        self.nms_spinbox.setValue(0.4)
        self.nms_spinbox.setFixedWidth(80)
        detection_layout.addRow("NMS:", self.nms_spinbox)
        
        control_layout.addWidget(detection_group)
        
        # 状态显示区域
        status_group = QGroupBox("系统状态")
        status_layout = QFormLayout(status_group)
        status_layout.setVerticalSpacing(2)
        
        self.fps_status_label = QLabel("FPS: --")
        self.fps_status_label.setStyleSheet("color: #666;")
        status_layout.addRow("处理速度:", self.fps_status_label)
        
        self.detection_status_label = QLabel("未检测")
        self.detection_status_label.setStyleSheet("color: red;")
        status_layout.addRow("球检测:", self.detection_status_label)
        
        control_layout.addWidget(status_group)
        
        control_layout.addStretch()  # 添加弹性空间
        
        parent_layout.addWidget(control_frame, 1)  # 给控制区域分配较小比重 (1/5)
    
    def setup_dock_widgets(self):
        """设置停靠窗口"""
        # 控制面板
        self.control_panel = ControlPanel("Control Panel")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.control_panel)
    
    def setup_menu_bar(self):
        """设置菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("File")
        
        # 打开视频文件
        open_video_action = QAction("Open Video Files...", self)
        open_video_action.setShortcut(QKeySequence.StandardKey.Open)
        open_video_action.setStatusTip("Open dual video files for analysis")
        open_video_action.triggered.connect(self.open_video_files)
        file_menu.addAction(open_video_action)
        
        # 网络摄像头连接
        open_camera_action = QAction("Connect Network Cameras...", self)
        open_camera_action.setShortcut("Ctrl+Shift+O")
        open_camera_action.setStatusTip("Connect to network cameras via IP (mutually exclusive with video files)")
        open_camera_action.triggered.connect(self.open_network_cameras)
        file_menu.addAction(open_camera_action)
        
        # 加载配置
        load_config_action = QAction("Load Configuration...", self)
        load_config_action.setStatusTip("Load system configuration from file")
        load_config_action.triggered.connect(self.load_configuration)
        file_menu.addAction(load_config_action)
        
        # 保存配置
        save_config_action = QAction("Save Configuration...", self)
        save_config_action.setStatusTip("Save current configuration to file")
        save_config_action.triggered.connect(self.save_configuration)
        file_menu.addAction(save_config_action)
        
        file_menu.addSeparator()
        
        # 导出数据
        export_menu = file_menu.addMenu("Export")
        
        export_trajectory_action = QAction("Export Trajectory...", self)
        export_trajectory_action.triggered.connect(self.export_trajectory)
        export_menu.addAction(export_trajectory_action)
        
        export_prediction_action = QAction("Export Prediction...", self)
        export_prediction_action.triggered.connect(self.export_prediction)
        export_menu.addAction(export_prediction_action)
        
        file_menu.addSeparator()
        
        # 退出
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 标定菜单
        calibration_menu = menubar.addMenu("Calibration")
        
        # 相机标定
        camera_calibration_action = QAction("Camera Calibration...", self)
        camera_calibration_action.setShortcut("Ctrl+C")
        camera_calibration_action.setStatusTip("Open camera calibration window")
        camera_calibration_action.triggered.connect(self.open_calibration_window)
        calibration_menu.addAction(camera_calibration_action)
        
        # 加载标定数据
        load_calibration_action = QAction("Load Calibration Data...", self)
        load_calibration_action.setStatusTip("Load existing calibration data")
        load_calibration_action.triggered.connect(self.load_calibration_data)
        calibration_menu.addAction(load_calibration_action)
        
        # 处理菜单
        processing_menu = menubar.addMenu("Processing")
        
        # 开始处理
        self.start_processing_action = QAction("Start Processing", self)
        self.start_processing_action.setShortcut("F5")
        self.start_processing_action.setStatusTip("Start video processing and detection")
        self.start_processing_action.triggered.connect(self.start_processing)
        processing_menu.addAction(self.start_processing_action)
        
        # 停止处理
        self.stop_processing_action = QAction("Stop Processing", self)
        self.stop_processing_action.setShortcut("F6")
        self.stop_processing_action.setStatusTip("Stop video processing")
        self.stop_processing_action.setEnabled(False)
        self.stop_processing_action.triggered.connect(self.stop_processing)
        processing_menu.addAction(self.stop_processing_action)
        
        processing_menu.addSeparator()
        
        # 预测
        predict_action = QAction("Predict Landing Point", self)
        predict_action.setShortcut("Space")
        predict_action.setStatusTip("Predict shuttlecock landing point")
        predict_action.triggered.connect(self.predict_landing)
        processing_menu.addAction(predict_action)
        
        # 重置
        reset_action = QAction("Reset Trajectory", self)
        reset_action.setShortcut("Ctrl+R")
        reset_action.setStatusTip("Reset trajectory data")
        reset_action.triggered.connect(self.reset_trajectory)
        processing_menu.addAction(reset_action)
        
        # 视图菜单\n
        view_menu = menubar.addMenu("View")
        
        # 主题切换
        theme_menu = view_menu.addMenu("Theme")
        
        light_theme_action = QAction("Light Theme", self)
        light_theme_action.triggered.connect(lambda: self.apply_theme("light"))
        theme_menu.addAction(light_theme_action)
        
        dark_theme_action = QAction("Dark Theme", self)
        dark_theme_action.triggered.connect(lambda: self.apply_theme("dark"))
        theme_menu.addAction(dark_theme_action)
        
        # 窗口布局
        view_menu.addSeparator()
        
        reset_layout_action = QAction("Reset Layout", self)
        reset_layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout_action)
        
        # 全屏
        fullscreen_action = QAction("Full Screen", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("Help")
        
        # 系统信息
        system_info_action = QAction("System Information", self)
        system_info_action.triggered.connect(self.show_system_info)
        help_menu.addAction(system_info_action)
        
        # 关于
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_tool_bar(self):
        """设置工具栏"""
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        
        # 视频控制
        toolbar.addAction("⏵", self.start_processing).setStatusTip("Start Processing")
        toolbar.addAction("⏸", self.pause_processing).setStatusTip("Pause Processing")
        toolbar.addAction("⏹", self.stop_processing).setStatusTip("Stop Processing")
        
        toolbar.addSeparator()
        
        # 标定
        toolbar.addAction("📷", self.open_calibration_window).setStatusTip("Camera Calibration")
        
        toolbar.addSeparator()
        
        # 预测
        toolbar.addAction("🎯", self.predict_landing).setStatusTip("Predict Landing")
        toolbar.addAction("🔄", self.reset_trajectory).setStatusTip("Reset Trajectory")
    
    def setup_status_bar(self):
        """设置状态栏"""
        self.statusBar().showMessage("Ready")
        
        # 添加永久状态信息
        self.fps_label = QLabel("FPS: 0")
        self.statusBar().addPermanentWidget(self.fps_label)
        
        self.memory_label = QLabel("Memory: 0 MB")
        self.statusBar().addPermanentWidget(self.memory_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.statusBar().addPermanentWidget(self.progress_bar)

    def setup_connections(self):
        """设置信号连接 - 确保所有信号都被正确处理"""
        # 视频组件
        if hasattr(self, 'video_widget'):
            self.video_widget.play_pause_clicked.connect(self.toggle_video_playback)
            self.video_widget.stop_clicked.connect(self.stop_processing)
            self.video_widget.seek_requested.connect(self.seek_video)
            self.video_widget.frame1_clicked.connect(self.on_frame_clicked)
            self.video_widget.frame2_clicked.connect(self.on_frame_clicked)

        # 控制面板
        self.control_panel.parameter_changed.connect(self.on_parameter_changed)
        self.control_panel.start_calibration.connect(self.open_calibration_window)
        self.control_panel.start_prediction.connect(self.predict_landing)
        self.control_panel.reset_trajectory.connect(self.reset_trajectory)

        # 调试：打印所有连接
        print("Signal connections setup completed")

    def initialize_workers(self):
        """初始化工作线程"""
        try:
            # 预测工作线程
            self.prediction_worker = PredictionWorker()
            self.prediction_worker.prediction_ready.connect(self.on_prediction_ready)
            self.prediction_worker.error_occurred.connect(self.on_worker_error)
            self.prediction_worker.status_changed.connect(self.on_prediction_status_changed)
            self.prediction_worker.start()

            # 双目检测工作线程
            self.stereo_worker = StereoDetectionWorker()
            self.stereo_worker.stereo_result.connect(self.on_stereo_result)
            self.stereo_worker.trajectory_3d_updated.connect(self.on_trajectory_3d_updated)
            self.stereo_worker.error_occurred.connect(self.on_worker_error)
            self.stereo_worker.start()

            logger.info("Worker threads initialized")

        except Exception as e:
            logger.error(f"Failed to initialize workers: {e}")

    def open_video_files(self):
        """打开视频文件 - 添加调试信息"""
        try:
            file_dialog = QFileDialog()
            file_paths, _ = file_dialog.getOpenFileNames(
                self, "Select Video Files", "",
                "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv)"
            )

            print(f"Selected files: {file_paths}")  # 调试输出

            if len(file_paths) >= 2:
                # 检查文件是否存在
                for path in file_paths[:2]:
                    if not os.path.exists(path):
                        print(f"File not found: {path}")
                        DialogUtils.show_error(self, "Error", f"File not found: {path}")
                        return

                # 创建双路视频工作线程
                self.video_worker = DualVideoWorker(file_paths[0], file_paths[1])

                # 连接信号
                self.video_worker.frames_ready.connect(self.on_frames_ready)
                self.video_worker.frame_ready.connect(self.on_single_frame_ready)
                self.video_worker.error_occurred.connect(self.on_worker_error)
                self.video_worker.status_changed.connect(self.on_video_status_changed)

                print("Starting video worker...")  # 调试输出
                self.video_worker.start_processing()
                
                # 自动开始播放视频
                self.video_worker.play()

                self.video_loaded = True
                self.statusBar().showMessage(f"Video files loaded: {len(file_paths)} files")

            elif len(file_paths) == 1:
                DialogUtils.show_warning(self, "Warning",
                                         "Please select two video files for stereo analysis")

        except Exception as e:
            print(f"Error in open_video_files: {e}")
            DialogUtils.show_error(self, "Error", f"Failed to open video files: {e}")

    def open_network_cameras(self):
        """连接网络摄像头 - 与本地视频互斥"""
        try:
            # 创建IP输入对话框
            from PyQt6.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QLabel, QVBoxLayout
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Connect Network Cameras")
            dialog.setModal(True)
            dialog.setMinimumWidth(400)
            
            layout = QVBoxLayout(dialog)
            
            # 说明文字
            info_label = QLabel("注意：网络摄像头和本地视频是互相替代关系，只能选择其中一种。")
            info_label.setStyleSheet("color: #666; font-style: italic; margin-bottom: 10px;")
            layout.addWidget(info_label)
            
            form_layout = QFormLayout()
            
            # 摄像头1 IP输入
            camera1_ip = QLineEdit()
            camera1_ip.setPlaceholderText("例如: rtsp://192.168.1.100:554/stream1")
            camera1_ip.setText("rtsp://192.168.1.100:554/stream1")  # 默认值
            form_layout.addRow("Camera 1 IP/URL:", camera1_ip)
            
            # 摄像头2 IP输入  
            camera2_ip = QLineEdit()
            camera2_ip.setPlaceholderText("例如: rtsp://192.168.1.101:554/stream1")
            camera2_ip.setText("rtsp://192.168.1.101:554/stream1")  # 默认值
            form_layout.addRow("Camera 2 IP/URL:", camera2_ip)
            
            layout.addLayout(form_layout)
            
            # 按钮
            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)
            
            # 显示对话框
            if dialog.exec() == QDialog.DialogCode.Accepted:
                ip1 = camera1_ip.text().strip()
                ip2 = camera2_ip.text().strip()
                
                if not ip1 or not ip2:
                    DialogUtils.show_warning(self, "Warning", "Please enter both camera IP addresses/URLs")
                    return
                
                # 停止现有的视频处理
                if self.video_worker:
                    self.video_worker.stop()
                    self.video_worker = None
                
                # 创建网络摄像头工作线程
                from video_worker import DualVideoWorker
                self.video_worker = DualVideoWorker(ip1, ip2)
                
                # 连接信号
                self.video_worker.frames_ready.connect(self.on_frames_ready)
                self.video_worker.frame_ready.connect(self.on_single_frame_ready)
                self.video_worker.error_occurred.connect(self.on_worker_error)
                self.video_worker.status_changed.connect(self.on_video_status_changed)
                
                print(f"Starting network cameras: {ip1}, {ip2}")
                self.video_worker.start_processing()
                
                # 自动开始播放视频
                self.video_worker.play()
                
                self.video_loaded = True
                self.statusBar().showMessage(f"Network cameras connected: {ip1}, {ip2}")
                
        except Exception as e:
            print(f"Error in open_network_cameras: {e}")
            DialogUtils.show_error(self, "Error", f"Failed to connect network cameras: {e}")

    def open_calibration_window(self):
        """打开标定窗口"""
        try:
            calibration_window = CalibrationWindow(self)
            calibration_window.calibration_completed.connect(self.on_calibration_completed)
            
            result = calibration_window.exec()
            
            if result == calibration_window.DialogCode.Accepted:
                logger.info("Calibration window closed successfully")
            
        except Exception as e:
            logger.error(f"Failed to open calibration window: {e}")
            DialogUtils.show_error(self, "Error", f"Failed to open calibration window: {e}")
    
    def start_processing(self):
        """开始处理"""
        if not self.video_loaded:
            DialogUtils.show_warning(self, "Warning", "Please load video files first")
            return
        
        try:
            # 启动检测工作线程
            if config.yolo_ball_model:
                self.detection_worker1 = DetectionWorker(0, config.yolo_ball_model)
                self.detection_worker2 = DetectionWorker(1, config.yolo_ball_model)
                
                # 连接信号
                self.detection_worker1.detection_result.connect(self.on_detection_result)
                self.detection_worker2.detection_result.connect(self.on_detection_result)
                self.detection_worker1.error_occurred.connect(self.on_worker_error)
                self.detection_worker2.error_occurred.connect(self.on_worker_error)
                
                # 启动线程
                self.detection_worker1.start()
                self.detection_worker2.start()
            
            # 启动视频播放
            if self.video_worker:
                self.video_worker.play()
            
            self.processing_active = True
            self.start_processing_action.setEnabled(False)
            self.stop_processing_action.setEnabled(True)
            
            self.statusBar().showMessage("Processing started")
            logger.info("Processing started")
            
        except Exception as e:
            logger.error(f"Failed to start processing: {e}")
            DialogUtils.show_error(self, "Error", f"Failed to start processing: {e}")
    
    def stop_processing(self):
        """停止处理"""
        try:
            # 停止视频
            if self.video_worker:
                self.video_worker.stop_processing()
            
            # 停止检测
            if self.detection_worker1:
                self.detection_worker1.stop()
                self.detection_worker1.wait(3000)
            
            if self.detection_worker2:
                self.detection_worker2.stop()
                self.detection_worker2.wait(3000)
            
            self.processing_active = False
            self.start_processing_action.setEnabled(True)
            self.stop_processing_action.setEnabled(False)
            
            self.statusBar().showMessage("Processing stopped")
            logger.info("Processing stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop processing: {e}")
    
    def pause_processing(self):
        """暂停处理"""
        if self.video_worker:
            self.video_worker.pause()
        
        self.statusBar().showMessage("Processing paused")
    
    def toggle_video_playback(self):
        """切换视频播放状态"""
        if self.video_worker:
            if self.processing_active:
                self.pause_processing()
            else:
                if self.video_worker:
                    self.video_worker.play()
                self.statusBar().showMessage("Processing resumed")
    
    def seek_video(self, frame_number):
        """跳转视频"""
        if self.video_worker:
            self.video_worker.seek(frame_number)
    
    def predict_landing(self):
        """预测落点"""
        try:
            if not self.stereo_worker:
                DialogUtils.show_warning(self, "Warning", "Stereo system not initialized")
                return
            
            # 获取3D轨迹数据
            trajectory_data = self.stereo_worker.get_trajectory_data()
            trajectory_3d = trajectory_data.get('trajectory_3d', [])
            timestamps = trajectory_data.get('timestamps', [])
            
            if len(trajectory_3d) < 5:
                DialogUtils.show_warning(self, "Warning", 
                                       f"Insufficient trajectory data: {len(trajectory_3d)} points")
                return
            
            # 请求预测
            success = self.prediction_worker.request_prediction(
                trajectory_3d, timestamps, priority='high'
            )
            
            if success:
                self.statusBar().showMessage("Prediction requested...")
                self.progress_bar.setVisible(True)
                logger.info("Prediction requested")
            else:
                DialogUtils.show_error(self, "Error", "Failed to request prediction")
            
        except Exception as e:
            logger.error(f"Failed to predict landing: {e}")
            DialogUtils.show_error(self, "Error", f"Failed to predict landing: {e}")
    
    def reset_trajectory(self):
        """重置轨迹"""
        try:
            # 重置检测器
            if self.detection_worker1:
                self.detection_worker1.reset_trajectory()
            if self.detection_worker2:
                self.detection_worker2.reset_trajectory()
            
            # 重置双目处理器
            if self.stereo_worker:
                self.stereo_worker.reset_trajectory()
            
            # 清除可视化
            self.viz_3d_widget.clear_all()
            
            self.statusBar().showMessage("Trajectory reset")
            logger.info("Trajectory reset")
            
        except Exception as e:
            logger.error(f"Failed to reset trajectory: {e}")

    def on_frames_ready(self, frame1, frame2, timestamp):
        """双路帧就绪"""
        if hasattr(self, 'video_widget'):
            self.video_widget.update_frames(frame1, frame2)

    def on_single_frame_ready(self, camera_id, frame, timestamp):
        """单路帧就绪"""
        # 发送给检测线程
        if camera_id == 0 and self.detection_worker1:
            self.detection_worker1.add_frame(frame, timestamp)
        elif camera_id == 1 and self.detection_worker2:
            self.detection_worker2.add_frame(frame, timestamp)
    
    def on_detection_result(self, camera_id, result):
        """检测结果处理"""
        # 更新视频显示
        if camera_id == 0:
            self.video_widget.update_frame1(result['frame'])
        elif camera_id == 1:
            self.video_widget.update_frame2(result['frame'])
        
        # 发送给双目处理器
        if self.stereo_worker:
            self.stereo_worker.add_detection_result(camera_id, result)
        
        # 更新控制面板状态
        self.control_panel.update_status(
            detection_status=(result['position'] is not None, 0, 0)
        )
    
    def on_stereo_result(self, result):
        """双目处理结果"""
        # 更新3D可视化
        if 'point_3d' in result:
            # 这里可以实时更新3D点
            pass
    
    def on_trajectory_3d_updated(self, trajectory_data):
        """3D轨迹更新"""
        trajectory_3d = trajectory_data.get('trajectory_3d', [])
        
        # 更新3D可视化
        self.viz_3d_widget.update_trajectory(trajectory_3d)
        
        # 更新控制面板状态
        quality = trajectory_data.get('quality', 0)
        self.control_panel.update_status(
            trajectory_status=(len(trajectory_3d), len(trajectory_3d), quality)
        )
    
    def on_pause_clicked(self):
        """暂停按钮点击处理"""
        if self.video_worker:
            self.video_worker.pause()
            self.is_paused = True
            self.pause_btn.setText("播放")
            
            # 启用预测和速度检测按钮
            self.predict_btn.setEnabled(True)
            self.speed_detect_btn.setEnabled(True)
            
            self.statusBar().showMessage("视频已暂停，可以开始预测")
        else:
            self.video_worker.play()
            self.is_paused = False
            self.pause_btn.setText("暂停")
            
            # 禁用预测和速度检测按钮
            self.predict_btn.setEnabled(False)
            self.speed_detect_btn.setEnabled(False)
    
    def on_predict_clicked(self):
        """预测按钮点击处理"""
        if not self.is_paused:
            DialogUtils.show_warning(self, "警告", "请先暂停视频再进行预测")
            return
            
        if self.video_worker:
            # 触发预测处理
            self.video_worker.trigger_prediction()
            self.statusBar().showMessage("正在进行轨迹预测...")
    
    def on_speed_detect_clicked(self):
        """速度检测按钮点击处理"""
        if not self.is_paused:
            DialogUtils.show_warning(self, "警告", "请先暂停视频再进行速度检测")
            return
            
        try:
            # 获取3D轨迹数据
            if self.stereo_worker:
                trajectory_data = self.stereo_worker.get_trajectory_data()
                trajectory_3d = trajectory_data.get('trajectory_3d', [])
                timestamps = trajectory_data.get('timestamps', [])
                
                if len(trajectory_3d) < 5:
                    DialogUtils.show_warning(self, "警告", "轨迹数据不足，需要至少5个点")
                    return
                
                # 进行速度检测
                result = self.speed_detector.detect_max_speed_before_landing(trajectory_3d, timestamps)
                
                if result:
                    # 显示速度检测结果
                    max_speed = result['max_speed']
                    max_speed_kmh = result['max_speed_kmh']
                    time_to_landing = result['time_to_landing']
                    
                    message = f"最大速度检测结果:\n"
                    message += f"最大速度: {max_speed:.1f} cm/s ({max_speed_kmh:.1f} km/h)\n"
                    message += f"距落地时间: {time_to_landing:.2f} 秒\n"
                    message += f"轨迹点数: {result['trajectory_length']}"
                    
                    DialogUtils.show_info(self, "速度检测结果", message)
                    self.statusBar().showMessage(f"最大速度: {max_speed_kmh:.1f} km/h")
                else:
                    DialogUtils.show_error(self, "错误", "速度检测失败")
            else:
                DialogUtils.show_warning(self, "警告", "立体视觉系统未初始化")
                
        except Exception as e:
            logger.error(f"Speed detection failed: {e}")
            DialogUtils.show_error(self, "错误", f"速度检测失败: {e}")
    
    def on_prediction_ready(self, result):
        """预测结果就绪"""
        try:
            self.progress_bar.setVisible(False)
            
            prediction = result.get('prediction', {})
            boundary_analysis = result.get('boundary_analysis', {})
            
            if prediction:
                landing_point = prediction.get('landing_point')
                trajectory = prediction.get('trajectory', [])
                confidence = prediction.get('confidence', 0)
                
                # 更新3D可视化
                self.viz_3d_widget.update_prediction(trajectory, landing_point)
                
                # 更新控制面板状态
                in_bounds = None
                if boundary_analysis:
                    in_bounds = boundary_analysis.get('in_bounds')
                
                self.control_panel.update_status(
                    prediction_status=("completed", confidence, landing_point, in_bounds)
                )
                
                # 状态栏消息
                if landing_point:
                    bounds_text = "IN" if in_bounds else "OUT" if in_bounds is not None else "UNKNOWN"
                    self.statusBar().showMessage(
                        f"Prediction: ({landing_point[0]:.1f}, {landing_point[1]:.1f}) - {bounds_text}"
                    )
                
                logger.info(f"Prediction completed: confidence={confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to process prediction result: {e}")
    
    def on_prediction_status_changed(self, status):
        """预测状态变化"""
        self.control_panel.add_log_message(f"Prediction: {status}")
    
    def on_worker_error(self, error_message):
        """工作线程错误"""
        logger.error(f"Worker error: {error_message}")
        self.control_panel.add_log_message(error_message, "ERROR")
    
    def on_video_status_changed(self, status):
        """视频状态变化"""
        self.statusBar().showMessage(f"Video: {status}")
    
    def on_frame_clicked(self, x, y):
        """帧点击处理"""
        logger.debug(f"Frame clicked at ({x}, {y})")
    
    def on_parameter_changed(self, parameter_name, value):
        """参数变化处理"""
        try:
            # 更新配置
            if hasattr(config, parameter_name):
                setattr(config, parameter_name, value)
            
            # 更新工作线程参数
            parameters = {parameter_name: value}
            
            if self.detection_worker1:
                self.detection_worker1.update_parameters(parameters)
            if self.detection_worker2:
                self.detection_worker2.update_parameters(parameters)
            if self.prediction_worker:
                self.prediction_worker.update_parameters(parameters)
            
            logger.debug(f"Parameter updated: {parameter_name} = {value}")
            
        except Exception as e:
            logger.error(f"Failed to update parameter {parameter_name}: {e}")
    
    def on_calibration_completed(self, result):
        """标定完成"""
        try:
            self.calibration_completed = True
            
            # 加载标定数据到双目处理器
            if self.stereo_worker and 'output_file' in result:
                output_file = result['output_file']
                success = self.stereo_worker.load_calibration(
                    output_file, output_file  # 假设使用相同的标定文件
                )
                
                if success:
                    self.control_panel.add_log_message("Calibration data loaded successfully")
                else:
                    self.control_panel.add_log_message("Failed to load calibration data", "ERROR")
            
            self.statusBar().showMessage("Calibration completed")
            logger.info("Calibration completed")
            
        except Exception as e:
            logger.error(f"Failed to process calibration result: {e}")
    
    def update_performance_stats(self):
        """更新性能统计"""
        try:
            system_info = SystemUtils.get_system_info()
            
            # 更新状态栏
            memory_mb = system_info['memory_total'] * system_info['memory_percent'] / 100 / (1024**2)
            self.memory_label.setText(f"Memory: {memory_mb:.0f} MB")
            
            # 更新控制面板
            self.control_panel.update_performance(
                cpu_percent=system_info['cpu_percent'],
                memory_percent=system_info['memory_percent']
            )
            
        except Exception as e:
            logger.debug(f"Failed to update performance stats: {e}")
    
    def apply_theme(self, theme_name):
        """应用主题"""
        try:
            if theme_name == "light":
                self.setStyleSheet(MAIN_STYLE)
            elif theme_name == "dark":
                self.setStyleSheet(MAIN_STYLE + DARK_THEME)
            
            # 保存主题设置
            config.set_setting("ui/theme", theme_name)
            
            logger.info(f"Theme applied: {theme_name}")
            
        except Exception as e:
            logger.error(f"Failed to apply theme: {e}")
    
    def reset_layout(self):
        """重置布局"""
        try:
            # 重置停靠窗口
            self.control_panel.setFloating(False)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.control_panel)
            
            # 重置窗口大小
            self.resize(1400, 900)
            
            logger.info("Layout reset")
            
        except Exception as e:
            logger.error(f"Failed to reset layout: {e}")
    
    def toggle_fullscreen(self):
        """切换全屏"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def load_configuration(self):
        """加载配置"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Configuration", "", "JSON Files (*.json)"
            )
            
            if file_path:
                # 这里可以实现配置加载逻辑
                DialogUtils.show_info(self, "Info", "Configuration loading not implemented yet")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            DialogUtils.show_error(self, "Error", f"Failed to load configuration: {e}")
    
    def save_configuration(self):
        """保存配置"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Configuration", "", "JSON Files (*.json)"
            )
            
            if file_path:
                # 这里可以实现配置保存逻辑
                DialogUtils.show_info(self, "Info", "Configuration saving not implemented yet")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            DialogUtils.show_error(self, "Error", f"Failed to save configuration: {e}")
    
    def export_trajectory(self):
        """导出轨迹"""
        try:
            if not self.stereo_worker:
                DialogUtils.show_warning(self, "Warning", "No trajectory data available")
                return
            
            trajectory_data = self.stereo_worker.get_trajectory_data()
            
            if not trajectory_data['trajectory_3d']:
                DialogUtils.show_warning(self, "Warning", "No trajectory data to export")
                return
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Trajectory", "", "CSV Files (*.csv)"
            )
            
            if file_path:
                # 这里可以实现轨迹导出逻辑
                DialogUtils.show_info(self, "Info", "Trajectory export not implemented yet")
            
        except Exception as e:
            logger.error(f"Failed to export trajectory: {e}")
            DialogUtils.show_error(self, "Error", f"Failed to export trajectory: {e}")
    
    def export_prediction(self):
        """导出预测结果"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Prediction", "", "JSON Files (*.json)"
            )
            
            if file_path:
                # 这里可以实现预测结果导出逻辑
                DialogUtils.show_info(self, "Info", "Prediction export not implemented yet")
            
        except Exception as e:
            logger.error(f"Failed to export prediction: {e}")
            DialogUtils.show_error(self, "Error", f"Failed to export prediction: {e}")
    
    def load_calibration_data(self):
        """加载标定数据"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Calibration Data", "", "YAML Files (*.yaml *.yml)"
            )
            
            if file_path and self.stereo_worker:
                success = self.stereo_worker.load_calibration(file_path, file_path)
                
                if success:
                    self.calibration_completed = True
                    self.control_panel.add_log_message("Calibration data loaded successfully")
                    self.statusBar().showMessage("Calibration data loaded")
                else:
                    DialogUtils.show_error(self, "Error", "Failed to load calibration data")
            
        except Exception as e:
            logger.error(f"Failed to load calibration data: {e}")
            DialogUtils.show_error(self, "Error", f"Failed to load calibration data: {e}")
    
    def show_system_info(self):
        """显示系统信息"""
        try:
            system_info = SystemUtils.get_system_info()
            dependencies = SystemUtils.check_dependencies()
            
            info_text = "System Information\n\n"
            info_text += f"CPU Cores: {system_info['cpu_count']}\n"
            info_text += f"CPU Usage: {system_info['cpu_percent']:.1f}%\n"
            info_text += f"Memory Total: {SystemUtils.format_bytes(system_info['memory_total'])}\n"
            info_text += f"Memory Available: {SystemUtils.format_bytes(system_info['memory_available'])}\n"
            info_text += f"Memory Usage: {system_info['memory_percent']:.1f}%\n"
            info_text += f"Disk Usage: {system_info['disk_usage']:.1f}%\n\n"
            
            info_text += "Dependencies:\n"
            for dep, available in dependencies.items():
                status = "✓" if available else "✗"
                info_text += f"{status} {dep}\n"
            
            DialogUtils.show_info(self, "System Information", info_text)
            
        except Exception as e:
            logger.error(f"Failed to show system info: {e}")
            DialogUtils.show_error(self, "Error", f"Failed to get system information: {e}")
    
    def show_about(self):
        """显示关于信息"""
        about_text = """
Badminton Landing Prediction System - PyQt6

Version: 1.0.0
Build Date: 2024

Features:
• Real-time shuttlecock detection and tracking
• Stereo vision 3D trajectory reconstruction  
• Advanced aerodynamic trajectory prediction
• Accurate landing point estimation
• Automated in/out boundary judgment

Developed with PyQt6, OpenCV, YOLO, and Open3D.
        """
        
        DialogUtils.show_info(self, "About", about_text.strip())
    
    def load_settings(self):
        """加载设置"""
        try:
            # 恢复窗口几何
            geometry = config.get_setting("ui/window_geometry", None)
            if geometry:
                self.restoreGeometry(geometry)
            
            # 恢复主题
            theme = config.get_setting("ui/theme", "light")
            self.apply_theme(theme)
            
            logger.info("Settings loaded")
            
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
    
    def save_settings(self):
        """保存设置"""
        try:
            # 保存窗口几何
            config.set_setting("ui/window_geometry", self.saveGeometry())
            
            # 保存配置
            config.save_config()
            
            logger.info("Settings saved")
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
    
    def closeEvent(self, event):
        """关闭事件"""
        try:
            # 停止所有处理
            self.stop_processing()
            
            # 停止工作线程
            if self.prediction_worker:
                self.prediction_worker.stop()
                self.prediction_worker.wait(3000)
            
            if self.stereo_worker:
                self.stereo_worker.stop()
                self.stereo_worker.wait(3000)
            
            # 保存设置
            self.save_settings()
            
            # 停止定时器
            if self.performance_timer:
                self.performance_timer.stop()
            
            logger.info("Application closing")
            event.accept()
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            event.accept()  # 强制关闭

    # 标定相关方法
    def browse_camera_params_file(self):
        """浏览相机参数文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Camera Parameters File", "", 
            "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            self.camera_params_path_edit.setText(file_path)

    def browse_yolo_model_file(self):
        """浏览YOLO模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model File", "", 
            "PyTorch Files (*.pt)"
        )
        if file_path:
            self.yolo_model_path_edit.setText(file_path)