"""
视频处理线程 - 简化版本
负责视频帧读取和缓冲，支持网络摄像头，5秒缓冲，暂停-预测工作流
"""
import cv2
import time
import numpy as np
import os
import requests
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
from collections import deque


class VideoWorker(QThread):
    """视频处理工作线程 - 简化版本"""
    
    # 信号定义
    frame_ready = pyqtSignal(int, np.ndarray, float)  # (camera_id, frame, timestamp)
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(int, str)  # (camera_id, status)
    video_info = pyqtSignal(int, int, float)  # (camera_id, total_frames, fps)
    
    def __init__(self, camera_id=0, video_source=None):
        super().__init__()
        
        self.camera_id = camera_id
        self.video_source = video_source  # 文件路径、相机ID或网络URL
        
        # 状态控制
        self.is_running = False
        self.is_paused = False
        self.seek_frame = -1
        
        # 视频捕获
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30.0
        
        # 网络摄像头支持
        self.is_network_camera = False
        self.network_session = None
        
        # 线程同步
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        
        # 5秒帧缓冲
        self.buffer_duration = 5.0  # 5秒缓冲
        self.frame_buffer = deque()
        self.timestamp_buffer = deque()
        
        # 性能统计
        self.frame_count = 0
        self.start_time = 0
        self.actual_fps = 0
    
    def set_video_source(self, source):
        """设置视频源"""
        self.mutex.lock()
        self.video_source = source
        self.mutex.unlock()
    
    def play(self):
        """开始播放"""
        self.mutex.lock()
        self.is_paused = False
        self.condition.wakeAll()
        self.mutex.unlock()
    
    def pause(self):
        """暂停播放"""
        self.mutex.lock()
        self.is_paused = True
        self.mutex.unlock()
    
    def stop(self):
        """停止播放"""
        self.mutex.lock()
        self.is_running = False
        self.condition.wakeAll()
        self.mutex.unlock()
    
    def seek(self, frame_number):
        """跳转到指定帧"""
        self.mutex.lock()
        self.seek_frame = frame_number
        self.condition.wakeAll()
        self.mutex.unlock()

    def initialize_capture(self):
        """初始化视频捕获 - 添加调试信息"""
        try:
            print(f"Initializing capture for camera {self.camera_id}, source: {self.video_source}")

            if self.cap is not None:
                self.cap.release()

            # 判断视频源类型
            if isinstance(self.video_source, str):
                print(f"Opening video file: {self.video_source}")

                # 检查文件是否存在
                if not os.path.exists(self.video_source):
                    raise FileNotFoundError(f"Video file not found: {self.video_source}")

                self.cap = cv2.VideoCapture(self.video_source)
                if not self.cap.isOpened():
                    raise ValueError(f"Cannot open video file: {self.video_source}")

                # 获取视频信息
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

                print(f"Video info - Total frames: {self.total_frames}, FPS: {self.fps}")

            elif isinstance(self.video_source, int):
                print(f"Opening camera: {self.video_source}")
                self.cap = cv2.VideoCapture(self.video_source)
                if not self.cap.isOpened():
                    raise ValueError(f"Cannot open camera {self.video_source}")

                # 设置摄像头参数
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

                self.total_frames = 0
                self.fps = 30.0

            # 发送视频信息
            self.video_info.emit(self.camera_id, self.total_frames, self.fps)
            self.status_changed.emit(self.camera_id, "Connected")

            print(f"Camera {self.camera_id} initialized successfully")
            return True

        except Exception as e:
            print(f"Camera {self.camera_id} initialization error: {str(e)}")
            self.error_occurred.emit(f"Camera {self.camera_id} initialization error: {str(e)}")
            self.status_changed.emit(self.camera_id, "Error")
            return False
    
    def extract_timestamp_from_headers(self):
        """从HTTP头提取X-Timestamp时间戳"""
        if not self.is_network_camera or not self.network_session:
            return None
        
        try:
            if self.video_source.startswith(('http://', 'https://')):
                response = self.network_session.head(self.video_source, timeout=0.1)
                x_timestamp = response.headers.get('X-Timestamp')
                if x_timestamp:
                    return float(x_timestamp)
        except:
            pass
        
        return None
    
    def cleanup_buffer(self):
        """清理过期的缓冲帧"""
        current_time = time.time()
        
        # 移除超过5秒的帧
        while (self.frame_buffer and self.timestamp_buffer and 
               current_time - self.timestamp_buffer[0] > self.buffer_duration):
            self.frame_buffer.popleft()
            self.timestamp_buffer.popleft()
    
    def get_buffered_frames(self):
        """获取缓冲的帧用于处理"""
        return list(self.frame_buffer), list(self.timestamp_buffer)
    
    def run(self):
        """主线程循环"""
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # 初始化视频捕获
        if not self.initialize_capture():
            return
        
        try:
            while self.is_running:
                self.mutex.lock()
                
                # 检查是否需要跳转
                if self.seek_frame >= 0:
                    if self.cap and isinstance(self.video_source, str):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_frame)
                        self.current_frame = self.seek_frame
                    self.seek_frame = -1
                
                # 检查是否暂停
                if self.is_paused:
                    self.condition.wait(self.mutex)
                    self.mutex.unlock()
                    continue
                
                self.mutex.unlock()
                
                # 读取帧
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    
                    if ret:
                        # 获取时间戳 - 优先使用网络摄像头的X-Timestamp
                        network_timestamp = self.extract_timestamp_from_headers()
                        timestamp = network_timestamp if network_timestamp else time.time()
                        
                        # 更新帧计数
                        self.current_frame += 1
                        self.frame_count += 1
                        
                        # 添加到5秒缓冲区
                        self.frame_buffer.append(frame.copy())
                        self.timestamp_buffer.append(timestamp)
                        
                        # 清理过期缓冲
                        self.cleanup_buffer()
                        
                        # 发送帧数据
                        self.frame_ready.emit(self.camera_id, frame, timestamp)
                        
                        # 控制帧率
                        if isinstance(self.video_source, str) and not self.is_network_camera:
                            time.sleep(1.0 / self.fps)
                        else:
                            time.sleep(0.01)  # 网络摄像头和本地摄像头稍微延迟
                    
                    else:
                        # 文件结束或读取错误
                        if isinstance(self.video_source, str) and not self.is_network_camera:
                            self.status_changed.emit(self.camera_id, "Completed")
                            break
                        else:
                            self.error_occurred.emit(f"Camera {self.camera_id} read error")
                            break
                
                else:
                    self.error_occurred.emit(f"Camera {self.camera_id} not available")
                    break
        
        except Exception as e:
            self.error_occurred.emit(f"Camera {self.camera_id} runtime error: {str(e)}")
        
        finally:
            # 清理资源
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.network_session:
                self.network_session.close()
            self.status_changed.emit(self.camera_id, "Disconnected")


class DualVideoWorker(QThread):
    """双路视频处理工作线程 - 简化版本，支持暂停-预测工作流"""

    # 信号定义 - 统一名称
    frames_ready = pyqtSignal(np.ndarray, np.ndarray, float)  # (frame1, frame2, timestamp)
    frame_ready = pyqtSignal(int, np.ndarray, float)  # (camera_id, frame, timestamp)
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    buffered_frames_ready = pyqtSignal(list, list, list, list)  # (frames1, timestamps1, frames2, timestamps2)

    def __init__(self, video_source1=None, video_source2=None):
        super().__init__()
        
        self.video_source1 = video_source1
        self.video_source2 = video_source2
        
        # 创建子工作线程
        self.worker1 = VideoWorker(camera_id=0, video_source=video_source1)
        self.worker2 = VideoWorker(camera_id=1, video_source=video_source2)
        
        # 状态控制
        self.is_running = False
        self.is_paused = False
        
        self.setup_connections()
    
    def setup_connections(self):
        """设置信号连接"""
        # 子线程帧信号
        self.worker1.frame_ready.connect(self.on_frame_ready)
        self.worker2.frame_ready.connect(self.on_frame_ready)
        
        # 错误信号
        self.worker1.error_occurred.connect(self.error_occurred.emit)
        self.worker2.error_occurred.connect(self.error_occurred.emit)
        
        # 状态信号
        self.worker1.status_changed.connect(self.on_status_changed)
        self.worker2.status_changed.connect(self.on_status_changed)
    
    def set_video_sources(self, source1, source2):
        """设置视频源"""
        self.video_source1 = source1
        self.video_source2 = source2
        self.worker1.set_video_source(source1)
        self.worker2.set_video_source(source2)
    
    def start_processing(self):
        """开始处理"""
        self.is_running = True
        self.worker1.start()
        self.worker2.start()
        self.start()
    
    def stop_processing(self):
        """停止处理"""
        self.is_running = False
        self.worker1.stop()
        self.worker2.stop()
        
        # 等待线程结束
        self.worker1.wait(3000)
        self.worker2.wait(3000)
        self.wait(1000)
    
    def play(self):
        """播放"""
        self.is_paused = False
        self.worker1.play()
        self.worker2.play()
    
    def pause(self):
        """暂停"""
        self.is_paused = True
        self.worker1.pause()
        self.worker2.pause()
    
    def trigger_prediction(self):
        """触发预测处理 - 仅在暂停后可用"""
        if not self.is_paused:
            self.error_occurred.emit("Must pause video before triggering prediction")
            return
        
        # 获取缓冲的帧
        frames1, timestamps1 = self.worker1.get_buffered_frames()
        frames2, timestamps2 = self.worker2.get_buffered_frames()
        
        # 发送缓冲的帧用于处理
        self.buffered_frames_ready.emit(frames1, timestamps1, frames2, timestamps2)
    
    def seek(self, frame_number):
        """跳转"""
        self.worker1.seek(frame_number)
        self.worker2.seek(frame_number)
    
    def on_frame_ready(self, camera_id, frame, timestamp):
        """帧就绪 - 直接转发"""
        self.frame_ready.emit(camera_id, frame, timestamp)
    
    def on_status_changed(self, camera_id, status):
        """处理状态变化"""
        self.status_changed.emit(f"Camera {camera_id}: {status}")
    
    def run(self):
        """主线程运行"""
        while self.is_running:
            time.sleep(0.1)