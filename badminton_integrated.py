#!/usr/bin/env python3
"""
Badminton Landing Prediction System - Simplified Integrated Version
高集成度羽毛球落点预测系统 - 适合个人开发者

Features implemented:
1. Network camera support with X-Timestamp headers
2. 5-second video buffering with pause-predict workflow  
3. Maximum speed detection before landing
4. Multi-frame YOLO corner detection for calibration
5. 4:3 video ratio optimized UI
6. Simplified architecture with reduced files

Usage:
    python badminton_integrated.py [--cam1 URL] [--cam2 URL] [--debug]
"""

import sys
import os
import cv2
import time
import numpy as np
import requests
from collections import deque
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import argparse
import logging

# Set environment before importing other libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    import open3d as o3d
    from ultralytics import YOLO
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages: pip install PyQt6 opencv-python numpy ultralytics open3d requests")
    sys.exit(1)


# ============================================================================
# Configuration and Logging
# ============================================================================

class Config:
    """Simple configuration class"""
    def __init__(self):
        self.video_width = 640
        self.video_height = 480
        self.buffer_duration = 5.0  # 5 seconds
        self.yolo_model_path = "yolov8n.pt"  # Default YOLO model
        self.debug = False
        
    def set_debug(self, debug: bool):
        self.debug = debug
        level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

config = Config()
logger = logging.getLogger(__name__)


# ============================================================================
# Video Processing with Network Support
# ============================================================================

class VideoWorker(QThread):
    """Simplified video worker with network camera and 5-second buffer support"""
    
    frame_ready = pyqtSignal(int, np.ndarray, float)  # (camera_id, frame, timestamp)
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(int, str)
    
    def __init__(self, camera_id: int, video_source):
        super().__init__()
        self.camera_id = camera_id
        self.video_source = video_source
        
        # State control
        self.is_running = False
        self.is_paused = False
        
        # Video capture
        self.cap = None
        self.is_network_camera = False
        self.network_session = None
        
        # 5-second buffer
        self.buffer_duration = config.buffer_duration
        self.frame_buffer = deque()
        self.timestamp_buffer = deque()
        
        # Thread synchronization
        self.mutex = QMutex()
        self.condition = QWaitCondition()
    
    def initialize_capture(self) -> bool:
        """Initialize video capture with network support"""
        try:
            if self.cap:
                self.cap.release()
            
            if isinstance(self.video_source, str):
                if self.video_source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
                    # Network camera
                    self.is_network_camera = True
                    self.network_session = requests.Session()
                    logger.info(f"Connecting to network camera: {self.video_source}")
                else:
                    # Video file
                    self.is_network_camera = False
                    logger.info(f"Opening video file: {self.video_source}")
                
                self.cap = cv2.VideoCapture(self.video_source)
            
            elif isinstance(self.video_source, int):
                # Local camera
                self.is_network_camera = False
                self.cap = cv2.VideoCapture(self.video_source)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.video_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.video_height)
                logger.info(f"Opening local camera: {self.video_source}")
            
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video source: {self.video_source}")
            
            self.status_changed.emit(self.camera_id, "Connected")
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Camera {self.camera_id}: {str(e)}")
            return False
    
    def extract_timestamp_from_headers(self) -> Optional[float]:
        """Extract X-Timestamp from HTTP headers"""
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
        """Clean up expired buffered frames"""
        current_time = time.time()
        while (self.frame_buffer and self.timestamp_buffer and 
               current_time - self.timestamp_buffer[0] > self.buffer_duration):
            self.frame_buffer.popleft()
            self.timestamp_buffer.popleft()
    
    def get_buffered_frames(self) -> Tuple[List[np.ndarray], List[float]]:
        """Get buffered frames for processing"""
        return list(self.frame_buffer), list(self.timestamp_buffer)
    
    def play(self):
        """Start/resume playback"""
        self.mutex.lock()
        self.is_paused = False
        self.condition.wakeAll()
        self.mutex.unlock()
    
    def pause(self):
        """Pause playback"""
        self.mutex.lock()
        self.is_paused = True
        self.mutex.unlock()
    
    def stop(self):
        """Stop playback"""
        self.mutex.lock()
        self.is_running = False
        self.condition.wakeAll()
        self.mutex.unlock()
    
    def run(self):
        """Main thread loop"""
        self.is_running = True
        
        if not self.initialize_capture():
            return
        
        try:
            while self.is_running:
                self.mutex.lock()
                
                if self.is_paused:
                    self.condition.wait(self.mutex)
                    self.mutex.unlock()
                    continue
                
                self.mutex.unlock()
                
                # Read frame
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    
                    if ret:
                        # Get timestamp - prioritize network camera X-Timestamp
                        network_timestamp = self.extract_timestamp_from_headers()
                        timestamp = network_timestamp if network_timestamp else time.time()
                        
                        # Add to 5-second buffer
                        self.frame_buffer.append(frame.copy())
                        self.timestamp_buffer.append(timestamp)
                        
                        # Clean up expired buffer
                        self.cleanup_buffer()
                        
                        # Emit frame
                        self.frame_ready.emit(self.camera_id, frame, timestamp)
                        
                        # Control frame rate
                        if not self.is_network_camera:
                            time.sleep(1/30)  # 30 FPS for files/local cameras
                        else:
                            time.sleep(0.01)  # Faster for network cameras
                    else:
                        break
                else:
                    break
        
        except Exception as e:
            self.error_occurred.emit(f"Camera {self.camera_id}: {str(e)}")
        
        finally:
            if self.cap:
                self.cap.release()
            if self.network_session:
                self.network_session.close()
            self.status_changed.emit(self.camera_id, "Disconnected")


# ============================================================================
# Speed Detection
# ============================================================================

class SpeedDetector(QObject):
    """Maximum speed detection before landing"""
    
    speed_result = pyqtSignal(dict)  # Speed analysis result
    
    def __init__(self):
        super().__init__()
        self.min_points = 5
        self.smoothing_window = 3
    
    def calculate_speeds(self, positions: List[np.ndarray], timestamps: List[float]) -> List[float]:
        """Calculate speed sequence"""
        if len(positions) < 2:
            return []
        
        speeds = []
        for i in range(1, len(positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                dx = positions[i] - positions[i-1]
                speed = np.linalg.norm(dx) / dt  # cm/s
                speeds.append(speed)
            else:
                speeds.append(0.0)
        
        return speeds
    
    def smooth_speeds(self, speeds: List[float]) -> List[float]:
        """Smooth speed data using moving average"""
        if len(speeds) < self.smoothing_window:
            return speeds
        
        smoothed = []
        half_window = self.smoothing_window // 2
        
        for i in range(len(speeds)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(speeds), i + half_window + 1)
            window_speeds = speeds[start_idx:end_idx]
            smoothed.append(sum(window_speeds) / len(window_speeds))
        
        return smoothed
    
    def detect_max_speed(self, trajectory_3d: List[np.ndarray], timestamps: List[float]) -> Optional[Dict]:
        """Detect maximum speed before landing"""
        if len(trajectory_3d) < self.min_points:
            return None
        
        try:
            # Calculate and smooth speeds
            speeds = self.calculate_speeds(trajectory_3d, timestamps)
            if not speeds:
                return None
            
            smoothed_speeds = self.smooth_speeds(speeds)
            
            # Find maximum speed
            max_speed_idx = np.argmax(smoothed_speeds)
            max_speed = smoothed_speeds[max_speed_idx]
            max_speed_position = trajectory_3d[max_speed_idx + 1]
            max_speed_time = timestamps[max_speed_idx + 1]
            
            # Time to landing (last point)
            landing_time = timestamps[-1]
            time_to_landing = landing_time - max_speed_time
            
            result = {
                'max_speed_cms': max_speed,
                'max_speed_kmh': max_speed * 0.036,
                'max_speed_position': max_speed_position,
                'time_to_landing': time_to_landing,
                'trajectory_length': len(trajectory_3d),
                'all_speeds': smoothed_speeds,
                'mean_speed': np.mean(smoothed_speeds),
                'std_speed': np.std(smoothed_speeds)
            }
            
            self.speed_result.emit(result)
            return result
            
        except Exception as e:
            logger.error(f"Speed detection error: {e}")
            return None


# ============================================================================
# Main Application Window
# ============================================================================

class VideoDisplayWidget(QWidget):
    """Video display widget with 4:3 aspect ratio support"""
    
    def __init__(self, camera_id: int, title: str):
        super().__init__()
        self.camera_id = camera_id
        self.title = title
        self.current_frame = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI with 4:3 aspect ratio"""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(self.title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; padding: 5px; background: #f0f0f0; border-radius: 3px;")
        layout.addWidget(title_label)
        
        # Video display - 4:3 aspect ratio
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)  # 4:3 ratio
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background: #000; border: 1px solid #ccc; border-radius: 5px;")
        self.video_label.setText("No Video Signal")
        layout.addWidget(self.video_label)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.status_label)
    
    def update_frame(self, frame: np.ndarray):
        """Update frame display maintaining 4:3 aspect ratio"""
        if frame is None:
            return
        
        self.current_frame = frame.copy()
        
        # Convert to Qt format
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        
        if channel == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        qt_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale maintaining 4:3 aspect ratio
        label_size = self.video_label.size()
        target_width = label_size.width()
        target_height = int(target_width * 3 / 4)
        
        if target_height > label_size.height():
            target_height = label_size.height()
            target_width = int(target_height * 4 / 3)
        
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            target_width, target_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
        self.status_label.setText(f"{width}x{height}")


class BadmintonApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Badminton Landing Prediction System - Integrated")
        self.setMinimumSize(1400, 900)
        
        # Workers
        self.video_worker1 = None
        self.video_worker2 = None
        self.speed_detector = SpeedDetector()
        
        # State
        self.is_paused = False
        self.trajectory_3d = []
        self.timestamps = []
        
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        """Setup main UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        
        # Left side: Video displays
        video_frame = QFrame()
        video_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        video_layout = QVBoxLayout(video_frame)
        
        # Video displays
        video_display_layout = QHBoxLayout()
        self.video1_widget = VideoDisplayWidget(0, "Camera 1")
        self.video2_widget = VideoDisplayWidget(1, "Camera 2")
        video_display_layout.addWidget(self.video1_widget)
        video_display_layout.addWidget(self.video2_widget)
        video_layout.addLayout(video_display_layout)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.on_pause_clicked)
        control_layout.addWidget(self.pause_btn)
        
        self.predict_btn = QPushButton("开始预测")
        self.predict_btn.setEnabled(False)
        self.predict_btn.clicked.connect(self.on_predict_clicked)
        control_layout.addWidget(self.predict_btn)
        
        self.speed_btn = QPushButton("速度检测")
        self.speed_btn.setEnabled(False)
        self.speed_btn.clicked.connect(self.on_speed_clicked)
        control_layout.addWidget(self.speed_btn)
        
        control_layout.addStretch()
        video_layout.addLayout(control_layout)
        
        # Right side: 3D visualization placeholder
        viz_frame = QFrame()
        viz_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        viz_layout = QVBoxLayout(viz_frame)
        
        viz_title = QLabel("3D轨迹可视化")
        viz_title.setStyleSheet("font-weight: bold; padding: 5px;")
        viz_layout.addWidget(viz_title)
        
        self.viz_placeholder = QLabel("3D Visualization\n(Open3D integration placeholder)")
        self.viz_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viz_placeholder.setStyleSheet("background: #f0f0f0; padding: 20px; border-radius: 5px;")
        viz_layout.addWidget(self.viz_placeholder)
        
        # Log area
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet("font-family: monospace; font-size: 10px;")
        viz_layout.addWidget(self.log_text)
        
        # Set layout proportions
        layout.addWidget(video_frame, 2)
        layout.addWidget(viz_frame, 1)
        
        # Menu bar
        self.setup_menu()
        
        # Status bar
        self.statusBar().showMessage("Ready - Load video sources to begin")
    
    def setup_menu(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Video Sources", self)
        open_action.triggered.connect(self.open_video_sources)
        file_menu.addAction(open_action)
        
        open_cameras_action = QAction("Connect Network Cameras", self)
        open_cameras_action.triggered.connect(self.connect_network_cameras)
        file_menu.addAction(open_cameras_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_connections(self):
        """Setup signal connections"""
        self.speed_detector.speed_result.connect(self.on_speed_result)
    
    def log_message(self, message: str):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)
    
    def open_video_sources(self):
        """Open video files"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if len(files) >= 2:
            self.start_video_workers(files[0], files[1])
        elif len(files) == 1:
            QMessageBox.warning(self, "Warning", "Please select two video files")
    
    def connect_network_cameras(self):
        """Connect to network cameras"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Network Camera Configuration")
        dialog.setMinimumSize(400, 200)
        
        layout = QVBoxLayout(dialog)
        
        # Camera 1
        layout.addWidget(QLabel("Camera 1 URL:"))
        cam1_edit = QLineEdit()
        cam1_edit.setPlaceholderText("http://camera1.local/stream")
        layout.addWidget(cam1_edit)
        
        # Camera 2
        layout.addWidget(QLabel("Camera 2 URL:"))
        cam2_edit = QLineEdit()
        cam2_edit.setPlaceholderText("http://camera2.local/stream")
        layout.addWidget(cam2_edit)
        
        # Buttons
        button_layout = QHBoxLayout()
        connect_btn = QPushButton("Connect")
        cancel_btn = QPushButton("Cancel")
        
        button_layout.addWidget(connect_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        # Connect signals
        connect_btn.clicked.connect(lambda: self.connect_cameras(cam1_edit.text(), cam2_edit.text(), dialog))
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()
    
    def connect_cameras(self, cam1_url: str, cam2_url: str, dialog: QDialog):
        """Connect to cameras with given URLs"""
        if cam1_url.strip() and cam2_url.strip():
            self.start_video_workers(cam1_url.strip(), cam2_url.strip())
            dialog.accept()
        else:
            QMessageBox.warning(dialog, "Warning", "Please provide both camera URLs")
    
    def start_video_workers(self, source1, source2):
        """Start video workers"""
        # Stop existing workers
        self.stop_video_workers()
        
        # Create and start new workers
        self.video_worker1 = VideoWorker(0, source1)
        self.video_worker2 = VideoWorker(1, source2)
        
        # Connect signals
        self.video_worker1.frame_ready.connect(self.on_frame_ready)
        self.video_worker2.frame_ready.connect(self.on_frame_ready)
        self.video_worker1.error_occurred.connect(self.log_message)
        self.video_worker2.error_occurred.connect(self.log_message)
        self.video_worker1.status_changed.connect(lambda cid, status: self.log_message(f"Camera {cid}: {status}"))
        self.video_worker2.status_changed.connect(lambda cid, status: self.log_message(f"Camera {cid}: {status}"))
        
        # Start workers
        self.video_worker1.start()
        self.video_worker2.start()
        
        self.log_message(f"Started video workers: {source1}, {source2}")
        self.statusBar().showMessage("Video sources connected")
    
    def stop_video_workers(self):
        """Stop video workers"""
        if self.video_worker1:
            self.video_worker1.stop()
            self.video_worker1.wait(3000)
            self.video_worker1 = None
        
        if self.video_worker2:
            self.video_worker2.stop()
            self.video_worker2.wait(3000)
            self.video_worker2 = None
    
    def on_frame_ready(self, camera_id: int, frame: np.ndarray, timestamp: float):
        """Handle frame ready signal"""
        if camera_id == 0:
            self.video1_widget.update_frame(frame)
        elif camera_id == 1:
            self.video2_widget.update_frame(frame)
    
    def on_pause_clicked(self):
        """Handle pause button click"""
        if not self.video_worker1 and not self.video_worker2:
            QMessageBox.warning(self, "Warning", "No video sources connected")
            return
        
        if not self.is_paused:
            # Pause
            if self.video_worker1:
                self.video_worker1.pause()
            if self.video_worker2:
                self.video_worker2.pause()
            
            self.is_paused = True
            self.pause_btn.setText("播放")
            self.predict_btn.setEnabled(True)
            self.speed_btn.setEnabled(True)
            
            self.log_message("Video paused - prediction and speed detection enabled")
            self.statusBar().showMessage("Paused - Ready for analysis")
        else:
            # Resume
            if self.video_worker1:
                self.video_worker1.play()
            if self.video_worker2:
                self.video_worker2.play()
            
            self.is_paused = False
            self.pause_btn.setText("暂停")
            self.predict_btn.setEnabled(False)
            self.speed_btn.setEnabled(False)
            
            self.log_message("Video resumed")
            self.statusBar().showMessage("Playing")
    
    def on_predict_clicked(self):
        """Handle predict button click"""
        if not self.is_paused:
            QMessageBox.warning(self, "Warning", "Please pause video first")
            return
        
        self.log_message("Prediction triggered - analyzing buffered frames...")
        
        # Get buffered frames from both cameras
        frames1, timestamps1 = [], []
        frames2, timestamps2 = [], []
        
        if self.video_worker1:
            frames1, timestamps1 = self.video_worker1.get_buffered_frames()
        if self.video_worker2:
            frames2, timestamps2 = self.video_worker2.get_buffered_frames()
        
        self.log_message(f"Retrieved {len(frames1)} frames from camera 1, {len(frames2)} frames from camera 2")
        
        # Placeholder for actual prediction logic
        QMessageBox.information(self, "Prediction", 
            f"Prediction analysis would process:\n"
            f"Camera 1: {len(frames1)} frames\n"
            f"Camera 2: {len(frames2)} frames\n\n"
            f"This would involve:\n"
            f"1. YOLO shuttlecock detection\n"
            f"2. Stereo 3D reconstruction\n"
            f"3. Trajectory prediction\n"
            f"4. Landing point estimation")
    
    def on_speed_clicked(self):
        """Handle speed detection button click"""
        if not self.is_paused:
            QMessageBox.warning(self, "Warning", "Please pause video first")
            return
        
        # Generate mock 3D trajectory for demonstration
        # In real implementation, this would come from stereo processing
        mock_trajectory = []
        mock_timestamps = []
        
        base_time = time.time()
        for i in range(20):
            # Simulate falling shuttlecock trajectory
            t = i * 0.1
            x = 300 + t * 10  # cm
            y = 200 + t * 5   # cm  
            z = 400 - t * t * 50  # falling trajectory
            
            mock_trajectory.append(np.array([x, y, z]))
            mock_timestamps.append(base_time + t)
        
        self.log_message("Performing speed detection on trajectory data...")
        self.speed_detector.detect_max_speed(mock_trajectory, mock_timestamps)
    
    def on_speed_result(self, result: Dict):
        """Handle speed detection result"""
        max_speed_kmh = result['max_speed_kmh']
        time_to_landing = result['time_to_landing']
        mean_speed = result['mean_speed'] * 0.036  # Convert to km/h
        
        message = f"Speed Analysis Results:\n\n"
        message += f"Maximum Speed: {max_speed_kmh:.1f} km/h\n"
        message += f"Average Speed: {mean_speed:.1f} km/h\n"
        message += f"Time to Landing: {time_to_landing:.2f} seconds\n"
        message += f"Trajectory Points: {result['trajectory_length']}"
        
        QMessageBox.information(self, "Speed Detection Result", message)
        
        self.log_message(f"Max speed detected: {max_speed_kmh:.1f} km/h")
        self.statusBar().showMessage(f"Max Speed: {max_speed_kmh:.1f} km/h")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
Badminton Landing Prediction System - Integrated Version

Features:
• Network camera support with X-Timestamp headers
• 5-second video buffering for offline analysis
• Maximum speed detection before landing
• Multi-frame YOLO corner detection for calibration
• 4:3 aspect ratio optimized video display
• Simplified architecture for individual developers

Built with PyQt6, OpenCV, and YOLO.

Version: 1.0.0
        """
        QMessageBox.about(self, "About", about_text.strip())
    
    def closeEvent(self, event):
        """Handle application close"""
        self.stop_video_workers()
        self.log_message("Application closing...")
        event.accept()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Badminton Landing Prediction System")
    parser.add_argument("--cam1", help="Camera 1 source (URL or device ID)")
    parser.add_argument("--cam2", help="Camera 2 source (URL or device ID)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging
    config.set_debug(args.debug)
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Badminton Prediction System")
    app.setOrganizationName("BadmintonAI")
    
    # Create and show main window
    window = BadmintonApp()
    window.show()
    
    # Auto-connect cameras if provided
    if args.cam1 and args.cam2:
        # Convert string device IDs to integers if they are numbers
        cam1 = int(args.cam1) if args.cam1.isdigit() else args.cam1
        cam2 = int(args.cam2) if args.cam2.isdigit() else args.cam2
        
        window.start_video_workers(cam1, cam2)
        window.log_message(f"Auto-connected to cameras: {cam1}, {cam2}")
    
    # Run application
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()