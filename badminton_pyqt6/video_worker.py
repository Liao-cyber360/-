"""
视频处理线程
负责视频帧读取和缓冲
"""
import cv2
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
from collections import deque


class VideoWorker(QThread):
    """视频处理工作线程"""
    
    # 信号定义
    frame_ready = pyqtSignal(int, np.ndarray, float)  # (camera_id, frame, timestamp)
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(int, str)  # (camera_id, status)
    video_info = pyqtSignal(int, int, float)  # (camera_id, total_frames, fps)
    
    def __init__(self, camera_id=0, video_source=None):
        super().__init__()
        
        self.camera_id = camera_id
        self.video_source = video_source  # 文件路径或相机ID
        
        # 状态控制
        self.is_running = False
        self.is_paused = False
        self.seek_frame = -1
        
        # 视频捕获
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30.0
        
        # 线程同步
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        
        # 帧缓冲
        self.frame_buffer = deque(maxlen=10)
        self.buffer_size = 10
        
        # 性能统计
        self.frame_count = 0
        self.start_time = 0
        self.last_fps_update = 0
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
        """初始化视频捕获"""
        try:
            if self.cap is not None:
                self.cap.release()
            
            # 判断是文件还是相机
            if isinstance(self.video_source, str):
                # 视频文件
                self.cap = cv2.VideoCapture(self.video_source)
                if not self.cap.isOpened():
                    raise ValueError(f"Cannot open video file: {self.video_source}")
                
                # 获取视频信息
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
                
            elif isinstance(self.video_source, int):
                # 摄像头
                self.cap = cv2.VideoCapture(self.video_source)
                if not self.cap.isOpened():
                    raise ValueError(f"Cannot open camera {self.video_source}")
                
                # 设置摄像头参数
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                self.total_frames = 0  # 实时摄像头没有总帧数
                self.fps = 30.0
            
            else:
                raise ValueError("Invalid video source")
            
            # 发送视频信息
            self.video_info.emit(self.camera_id, self.total_frames, self.fps)
            self.status_changed.emit(self.camera_id, "Connected")
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Camera {self.camera_id} initialization error: {str(e)}")
            self.status_changed.emit(self.camera_id, "Error")
            return False
    
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
                        timestamp = time.time()
                        
                        # 更新帧计数
                        self.current_frame += 1
                        self.frame_count += 1
                        
                        # 发送帧数据
                        self.frame_ready.emit(self.camera_id, frame, timestamp)
                        
                        # 更新FPS统计
                        self.update_fps_stats()
                        
                        # 控制帧率
                        if isinstance(self.video_source, str):
                            # 文件播放需要控制帧率
                            time.sleep(1.0 / self.fps)
                        else:
                            # 实时摄像头，稍微延迟避免过于频繁
                            time.sleep(0.001)
                    
                    else:
                        # 文件结束或读取错误
                        if isinstance(self.video_source, str):
                            # 视频文件播放完毕
                            self.status_changed.emit(self.camera_id, "Completed")
                            break
                        else:
                            # 摄像头读取错误
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
            self.status_changed.emit(self.camera_id, "Disconnected")
    
    def update_fps_stats(self):
        """更新FPS统计"""
        current_time = time.time()
        
        if current_time - self.last_fps_update >= 1.0:  # 每秒更新一次
            elapsed = current_time - self.start_time
            if elapsed > 0:
                self.actual_fps = self.frame_count / elapsed
            self.last_fps_update = current_time
    
    def get_current_frame_info(self):
        """获取当前帧信息"""
        return {
            'current_frame': self.current_frame,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'actual_fps': self.actual_fps
        }


class DualVideoWorker(QThread):
    """双路视频同步处理工作线程"""
    
    # 信号定义
    frames_ready = pyqtSignal(np.ndarray, np.ndarray, float)  # (frame1, frame2, timestamp)
    frame_ready = pyqtSignal(int, np.ndarray, float)  # (camera_id, frame, timestamp)
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    sync_status = pyqtSignal(bool, float)  # (is_synced, time_diff)
    
    def __init__(self, video_source1=None, video_source2=None):
        super().__init__()
        
        self.video_source1 = video_source1
        self.video_source2 = video_source2
        
        # 创建子工作线程
        self.worker1 = VideoWorker(camera_id=0, video_source=video_source1)
        self.worker2 = VideoWorker(camera_id=1, video_source=video_source2)
        
        # 状态控制
        self.is_running = False
        self.sync_enabled = True
        self.sync_threshold = 0.05  # 50ms同步阈值
        
        # 帧缓冲用于同步
        self.buffer1 = deque(maxlen=30)
        self.buffer2 = deque(maxlen=30)
        
        # 同步统计
        self.sync_errors = 0
        self.total_frames = 0
        
        self.setup_connections()
    
    def setup_connections(self):
        """设置信号连接"""
        # 子线程帧信号
        self.worker1.frame_ready.connect(self.on_frame1_ready)
        self.worker2.frame_ready.connect(self.on_frame2_ready)
        
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
        self.worker1.wait(3000)  # 最多等待3秒
        self.worker2.wait(3000)
        self.wait(1000)
    
    def play(self):
        """播放"""
        self.worker1.play()
        self.worker2.play()
    
    def pause(self):
        """暂停"""
        self.worker1.pause()
        self.worker2.pause()
    
    def seek(self, frame_number):
        """跳转"""
        self.worker1.seek(frame_number)
        self.worker2.seek(frame_number)
    
    def on_frame1_ready(self, camera_id, frame, timestamp):
        """相机1帧就绪"""
        self.buffer1.append((frame, timestamp))
        self.frame_ready.emit(camera_id, frame, timestamp)
        self.try_sync_frames()
    
    def on_frame2_ready(self, camera_id, frame, timestamp):
        """相机2帧就绪"""
        self.buffer2.append((frame, timestamp))
        self.frame_ready.emit(camera_id, frame, timestamp)
        self.try_sync_frames()
    
    def try_sync_frames(self):
        """尝试同步帧"""
        if not self.sync_enabled or len(self.buffer1) == 0 or len(self.buffer2) == 0:
            return
        
        # 寻找时间戳最接近的帧对
        best_match = None
        min_time_diff = float('inf')
        
        for i, (frame1, ts1) in enumerate(self.buffer1):
            for j, (frame2, ts2) in enumerate(self.buffer2):
                time_diff = abs(ts1 - ts2)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_match = (i, j, frame1, frame2, ts1, ts2)
        
        if best_match and min_time_diff <= self.sync_threshold:
            i, j, frame1, frame2, ts1, ts2 = best_match
            
            # 移除已使用的帧
            self.remove_used_frames(i, j)
            
            # 发送同步帧
            avg_timestamp = (ts1 + ts2) / 2
            self.frames_ready.emit(frame1, frame2, avg_timestamp)
            
            # 更新同步状态
            self.total_frames += 1
            self.sync_status.emit(True, min_time_diff)
        
        elif min_time_diff > self.sync_threshold:
            # 同步失败
            self.sync_errors += 1
            self.sync_status.emit(False, min_time_diff)
            
            # 清理过旧的帧
            self.cleanup_old_frames()
    
    def remove_used_frames(self, index1, index2):
        """移除已使用的帧"""
        # 移除已匹配的帧以及之前的帧
        for _ in range(index1 + 1):
            if self.buffer1:
                self.buffer1.popleft()
        
        for _ in range(index2 + 1):
            if self.buffer2:
                self.buffer2.popleft()
    
    def cleanup_old_frames(self):
        """清理过旧的帧"""
        current_time = time.time()
        max_age = 0.1  # 100ms
        
        # 清理buffer1中的旧帧
        while self.buffer1:
            _, timestamp = self.buffer1[0]
            if current_time - timestamp > max_age:
                self.buffer1.popleft()
            else:
                break
        
        # 清理buffer2中的旧帧
        while self.buffer2:
            _, timestamp = self.buffer2[0]
            if current_time - timestamp > max_age:
                self.buffer2.popleft()
            else:
                break
    
    def on_status_changed(self, camera_id, status):
        """处理状态变化"""
        if camera_id == 0:
            status1 = status
            status2 = "Unknown"
        else:
            status1 = "Unknown"
            status2 = status
        
        combined_status = f"Cam1: {status1}, Cam2: {status2}"
        self.status_changed.emit(combined_status)
    
    def set_sync_enabled(self, enabled):
        """设置是否启用同步"""
        self.sync_enabled = enabled
    
    def set_sync_threshold(self, threshold):
        """设置同步阈值"""
        self.sync_threshold = threshold
    
    def get_sync_stats(self):
        """获取同步统计信息"""
        sync_rate = 0
        if self.total_frames > 0:
            sync_rate = (self.total_frames - self.sync_errors) / self.total_frames
        
        return {
            'total_frames': self.total_frames,
            'sync_errors': self.sync_errors,
            'sync_rate': sync_rate,
            'buffer1_size': len(self.buffer1),
            'buffer2_size': len(self.buffer2)
        }
    
    def run(self):
        """主线程运行（监控和维护）"""
        while self.is_running:
            self.cleanup_old_frames()
            self.msleep(100)  # 每100ms清理一次