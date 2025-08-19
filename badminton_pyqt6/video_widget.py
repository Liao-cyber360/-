"""
视频显示组件
支持双路视频同步显示和播放控制
"""
import cv2
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QSlider, QFrame, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QFont


class VideoDisplayWidget(QWidget):
    """单路视频显示组件"""
    
    frame_clicked = pyqtSignal(int, int)  # 点击位置信号
    
    def __init__(self, camera_id=0, title="Camera"):
        super().__init__()
        self.camera_id = camera_id
        self.title = title
        
        # 显示状态
        self.current_frame = None
        self.frame_size = QSize(640, 480)
        self.aspect_ratio = 4/3
        
        self.setup_ui()
    
    def setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 标题
        self.title_label = QLabel(self.title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                color: #333;
                font-weight: bold;
                font-size: 14px;
                padding: 5px;
                background-color: #f0f0f0;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.title_label)
        
        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #000;
                border: 2px solid #ccc;
                border-radius: 5px;
            }
        """)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("No Video")
        self.video_label.mousePressEvent = self.on_frame_clicked
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.video_label)
        
        # 状态信息
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 11px;
                padding: 2px;
            }
        """)
        layout.addWidget(self.status_label)
    
    def update_frame(self, frame):
        """更新显示帧"""
        if frame is None:
            return
        
        self.current_frame = frame.copy()
        
        # 转换为Qt图像格式
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        
        # 确保是RGB格式
        if channel == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        qt_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # 缩放到适合的大小
        label_size = self.video_label.size()
        if label_size.width() > 0 and label_size.height() > 0:
            scaled_image = qt_image.scaled(
                label_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            pixmap = QPixmap.fromImage(scaled_image)
            self.video_label.setPixmap(pixmap)
        
        # 更新状态
        self.status_label.setText(f"Frame: {width}x{height}")
    
    def on_frame_clicked(self, event):
        """处理帧点击事件"""
        if self.current_frame is None:
            return
        
        # 获取点击位置
        click_x = event.pos().x()
        click_y = event.pos().y()
        
        # 计算在原始帧中的坐标
        label_size = self.video_label.size()
        pixmap = self.video_label.pixmap()
        
        if pixmap and label_size.width() > 0 and label_size.height() > 0:
            # 计算缩放比例
            scale_x = self.current_frame.shape[1] / pixmap.width()
            scale_y = self.current_frame.shape[0] / pixmap.height()
            
            # 计算偏移（居中显示）
            offset_x = (label_size.width() - pixmap.width()) // 2
            offset_y = (label_size.height() - pixmap.height()) // 2
            
            # 转换坐标
            frame_x = int((click_x - offset_x) * scale_x)
            frame_y = int((click_y - offset_y) * scale_y)
            
            # 发射信号
            self.frame_clicked.emit(frame_x, frame_y)
    
    def clear_display(self):
        """清空显示"""
        self.video_label.clear()
        self.video_label.setText("No Video")
        self.current_frame = None
        self.status_label.setText("Ready")
    
    def set_status(self, status):
        """设置状态文本"""
        self.status_label.setText(status)


class DualVideoWidget(QWidget):
    """双路视频同步显示组件"""
    
    # 信号定义
    frame1_clicked = pyqtSignal(int, int)  # 相机1点击
    frame2_clicked = pyqtSignal(int, int)  # 相机2点击
    play_pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    seek_requested = pyqtSignal(int)  # 跳转到指定帧
    
    def __init__(self):
        super().__init__()
        
        # 播放状态
        self.is_playing = False
        self.current_frame_number = 0
        self.total_frames = 0
        self.fps = 30
        
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 双路视频显示区域
        video_layout = QHBoxLayout()
        video_layout.setSpacing(10)
        
        # 相机1
        self.camera1_widget = VideoDisplayWidget(camera_id=0, title="Camera 1")
        video_layout.addWidget(self.camera1_widget)
        
        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("color: #ccc;")
        video_layout.addWidget(separator)
        
        # 相机2
        self.camera2_widget = VideoDisplayWidget(camera_id=1, title="Camera 2")
        video_layout.addWidget(self.camera2_widget)
        
        layout.addLayout(video_layout, 1)  # 视频区域占主要空间
        
        # 播放控制区域
        control_frame = QFrame()
        control_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        control_layout = QVBoxLayout(control_frame)
        
        # 进度滑块
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setMinimum(0)
        self.progress_slider.setMaximum(100)
        self.progress_slider.setValue(0)
        self.progress_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                height: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #fff, stop:1 #ddd);
                margin: 2px 0;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #45a049;
            }
        """)
        control_layout.addWidget(self.progress_slider)
        
        # 播放控制按钮
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 播放/暂停按钮
        self.play_pause_btn = QPushButton("▶ Play")
        self.play_pause_btn.setFixedSize(80, 35)
        self.play_pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        button_layout.addWidget(self.play_pause_btn)
        
        # 停止按钮
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setFixedSize(80, 35)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        button_layout.addWidget(self.stop_btn)
        
        control_layout.addLayout(button_layout)
        
        # 状态信息
        info_layout = QHBoxLayout()
        
        self.frame_info_label = QLabel("Frame: 0 / 0")
        self.frame_info_label.setStyleSheet("color: #666; font-size: 11px;")
        info_layout.addWidget(self.frame_info_label)
        
        info_layout.addStretch()
        
        self.fps_label = QLabel("FPS: 30")
        self.fps_label.setStyleSheet("color: #666; font-size: 11px;")
        info_layout.addWidget(self.fps_label)
        
        control_layout.addLayout(info_layout)
        
        layout.addWidget(control_frame)
    
    def setup_connections(self):
        """设置信号连接"""
        # 视频点击
        self.camera1_widget.frame_clicked.connect(self.frame1_clicked.emit)
        self.camera2_widget.frame_clicked.connect(self.frame2_clicked.emit)
        
        # 播放控制
        self.play_pause_btn.clicked.connect(self.on_play_pause_clicked)
        self.stop_btn.clicked.connect(self.on_stop_clicked)
        self.progress_slider.sliderPressed.connect(self.on_slider_pressed)
        self.progress_slider.sliderReleased.connect(self.on_slider_released)
    
    def update_frames(self, frame1=None, frame2=None):
        """更新双路视频帧"""
        if frame1 is not None:
            self.camera1_widget.update_frame(frame1)
        
        if frame2 is not None:
            self.camera2_widget.update_frame(frame2)
    
    def update_frame1(self, frame):
        """更新相机1帧"""
        self.camera1_widget.update_frame(frame)
    
    def update_frame2(self, frame):
        """更新相机2帧"""
        self.camera2_widget.update_frame(frame)
    
    def set_video_info(self, total_frames, fps):
        """设置视频信息"""
        self.total_frames = total_frames
        self.fps = fps
        self.progress_slider.setMaximum(total_frames - 1)
        self.fps_label.setText(f"FPS: {fps}")
        self.update_frame_info()
    
    def set_current_frame(self, frame_number):
        """设置当前帧号"""
        self.current_frame_number = frame_number
        self.progress_slider.setValue(frame_number)
        self.update_frame_info()
    
    def update_frame_info(self):
        """更新帧信息显示"""
        self.frame_info_label.setText(f"Frame: {self.current_frame_number} / {self.total_frames}")
    
    def set_playing_state(self, is_playing):
        """设置播放状态"""
        self.is_playing = is_playing
        if is_playing:
            self.play_pause_btn.setText("⏸ Pause")
        else:
            self.play_pause_btn.setText("▶ Play")
    
    def on_play_pause_clicked(self):
        """播放/暂停按钮点击"""
        self.play_pause_clicked.emit()
    
    def on_stop_clicked(self):
        """停止按钮点击"""
        self.stop_clicked.emit()
    
    def on_slider_pressed(self):
        """进度条按下"""
        pass
    
    def on_slider_released(self):
        """进度条释放"""
        frame_number = self.progress_slider.value()
        self.seek_requested.emit(frame_number)
    
    def clear_display(self):
        """清空显示"""
        self.camera1_widget.clear_display()
        self.camera2_widget.clear_display()
        self.current_frame_number = 0
        self.total_frames = 0
        self.set_playing_state(False)
        self.update_frame_info()
    
    def set_camera_status(self, camera_id, status):
        """设置相机状态"""
        if camera_id == 0:
            self.camera1_widget.set_status(status)
        elif camera_id == 1:
            self.camera2_widget.set_status(status)