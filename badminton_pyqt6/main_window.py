"""
主窗口界面
QMainWindow为基础，Material Design风格
"""
import sys
import os
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QMenuBar, QMenu, QStatusBar, QSplitter, QLabel,
                            QFileDialog, QMessageBox, QApplication, QToolBar,
                            QProgressBar, QFrame, QDockWidget)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, QSettings
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QPixmap
import time

# 导入自定义组件
from video_widget import DualVideoWidget
from control_panel import ControlPanel
from calibration_window import CalibrationWindow
from visualization_3d import Visualization3DWidget, CourtVisualizationWidget
from video_worker import DualVideoWorker
from detection_worker import DetectionWorker, StereoDetectionWorker
from prediction_worker import PredictionWorker
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
        
        # 状态
        self.system_initialized = False
        self.calibration_completed = False
        self.video_loaded = False
        self.processing_active = False
        
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
        """设置主界面"""
        # 中央窗口
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # 创建分割器
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：视频显示区域
        self.setup_video_area(main_splitter)
        
        # 右侧：3D可视化区域
        self.setup_visualization_area(main_splitter)
        
        # 设置分割器比例
        main_splitter.setSizes([900, 500])
        main_layout.addWidget(main_splitter)
        
        # 创建停靠面板
        self.setup_dock_widgets()
    
    def setup_video_area(self, parent):
        """设置视频显示区域"""
        # 视频区域容器
        video_frame = QFrame()
        video_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(5, 5, 5, 5)
        
        # 视频显示组件
        self.video_widget = DualVideoWidget()
        video_layout.addWidget(self.video_widget)
        
        parent.addWidget(video_frame)
    
    def setup_visualization_area(self, parent):
        """设置可视化区域"""
        # 可视化区域容器
        viz_frame = QFrame()
        viz_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        viz_layout = QVBoxLayout(viz_frame)
        viz_layout.setContentsMargins(5, 5, 5, 5)
        
        # 创建可视化选项卡分割器
        viz_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 3D可视化
        self.viz_3d_widget = Visualization3DWidget()
        viz_splitter.addWidget(self.viz_3d_widget)
        
        # 2D场地可视化
        self.court_viz_widget = CourtVisualizationWidget()
        viz_splitter.addWidget(self.court_viz_widget)
        
        # 设置分割器比例
        viz_splitter.setSizes([400, 200])
        viz_layout.addWidget(viz_splitter)
        
        parent.addWidget(viz_frame)
    
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
        """设置信号连接"""
        # 视频组件
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
            DialogUtils.show_error(self, "Initialization Error", 
                                 f"Failed to initialize worker threads: {e}")
    
    def open_video_files(self):
        """打开视频文件"""
        try:
            file_dialog = QFileDialog()
            file_paths, _ = file_dialog.getOpenFileNames(
                self, "Select Video Files", "",
                "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv)"
            )
            
            if len(file_paths) >= 2:
                # 创建双路视频工作线程
                self.video_worker = DualVideoWorker(file_paths[0], file_paths[1])
                
                # 连接信号
                self.video_worker.frames_ready.connect(self.on_frames_ready)
                self.video_worker.frame_ready.connect(self.on_single_frame_ready)
                self.video_worker.error_occurred.connect(self.on_worker_error)
                self.video_worker.status_changed.connect(self.on_video_status_changed)
                
                # 开始处理
                self.video_worker.start_processing()
                
                self.video_loaded = True
                self.statusBar().showMessage(f"Video files loaded: {len(file_paths)} files")
                logger.info(f"Video files loaded: {file_paths}")
                
                # 更新界面状态
                self.start_processing_action.setEnabled(True)
                
            elif len(file_paths) == 1:
                DialogUtils.show_warning(self, "Warning", 
                                       "Please select two video files for stereo analysis")
            
        except Exception as e:
            logger.error(f"Failed to open video files: {e}")
            DialogUtils.show_error(self, "Error", f"Failed to open video files: {e}")
    
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
        
        # 更新2D可视化
        self.court_viz_widget.update_trajectory_2d(trajectory_3d)
        
        # 更新控制面板状态
        quality = trajectory_data.get('quality', 0)
        self.control_panel.update_status(
            trajectory_status=(len(trajectory_3d), len(trajectory_3d), quality)
        )
    
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
                
                # 更新2D可视化
                self.court_viz_widget.update_prediction_2d(trajectory, landing_point)
                
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