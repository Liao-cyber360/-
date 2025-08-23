import os
import time
import cv2
import numpy as np


class Config:
    """配置类,集中管理所有系统参数"""

    def __init__(self):
        # 相机参数
        self.camera_params_file_1 = "E:\hawkeye\calibration\calibration_results_2025-08-11_18-06-15.yaml"
        self.camera_params_file_2 = "E:\hawkeye\calibration\calibration_results_2025-08-11_18-06-15.yaml"

        # 模型参数
        self.yolo_ball_model = "E:\\hawkeye\\ball\\best.pt"
        self.yolo_court_model = "E:\\hawkeye\\field\\best.pt"

        # 视频参数
        self.video_width = 1280
        self.video_height = 720
        self.fps = 30

        # 分析参数
        self.trajectory_buffer_size = 30  # 增加缓存大小
        self.landing_analysis_window = 10
        self.prediction_time_window = 1.5  # 增加预测时间窗口
        self.poly_fit_degree = 4

        # 物理模型参数
        self.shuttlecock_mass = 0.0048
        self.shuttlecock_radius = 0.025
        self.air_density = 1.225
        self.drag_coefficient = 0.6
        self.gravity = 9.8

        # 计算空气动力学长度
        self.cross_section = np.pi * self.shuttlecock_radius ** 2
        self.aero_length = 2 * self.shuttlecock_mass / (self.air_density * self.cross_section * self.drag_coefficient)

        # EKF参数
        self.ekf_process_noise = 0.01
        self.ekf_measurement_noise = 0.1

        # 系统参数
        self.results_dir = f"./results_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)

        # 界面参数
        self.court_view_width = 610
        self.court_view_height = 1340
        self.display_scale = 0.5

        # 判定参数 - 优化参数
        self.landing_detection_threshold = 5  # 稍微提高阈值
        self.landing_confirmation_frames = 3  # 减少确认帧数，提高响应速度
        self.landing_height_threshold = 15.0  # 稍微提高高度阈值

    def get_aero_params(self):
        """返回空气动力学参数"""
        return {
            'mass': self.shuttlecock_mass,
            'gravity': self.gravity,
            'aero_length': self.aero_length
        }


class UIHelper:
    """用户界面辅助类 - 优化版"""

    @staticmethod
    def create_status_bar(frame, fps, frame_count, trajectory, system_state):
        """创建优化的状态栏"""
        h, w = frame.shape[:2]

        # 创建状态栏
        status_bar = frame.copy()
        cv2.rectangle(status_bar, (0, h - 50), (w, h), (0, 0, 0), -1)

        # FPS信息
        cv2.putText(status_bar, f"FPS: {fps:.1f}", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 帧计数
        cv2.putText(status_bar, f"Frame: {frame_count}", (120, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 检测状态
        detection_status = len(trajectory) > 0 if trajectory else False
        color = (0, 255, 0) if detection_status else (0, 0, 255)
        status = "Detected" if detection_status else "No Detection"
        cv2.putText(status_bar, f"Ball: {status}", (250, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 轨迹点数
        trajectory_count = len(trajectory) if trajectory else 0
        cv2.putText(status_bar, f"Points: {trajectory_count}", (400, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 系统状态
        state_colors = {
            "idle": (150, 150, 150),
            "detecting": (255, 255, 0),
            "landing_detected": (255, 165, 0),
            "prediction_ready": (0, 255, 255),
            "predicting": (255, 0, 255),
            "prediction_complete": (0, 255, 0)
        }

        color = state_colors.get(system_state, (255, 255, 255))
        cv2.putText(status_bar, f"State: {system_state.upper()}", (520, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 控制提示
        cv2.putText(status_bar, "SPACE:Predict | H:Help | R:Reset | F:FP Stats | M:Mark FP | ESC:Exit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        return status_bar

    @staticmethod
    def display_splash_screen(duration=3):
        """显示启动画面"""
        splash = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 标题
        cv2.putText(splash, "Badminton Shuttlecock Landing Prediction System v5.0",
                    (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # 分割线
        cv2.line(splash, (150, 120), (1130, 120), (0, 120, 255), 2)

        # 功能介绍
        features = [
            "- Enhanced real-time shuttlecock detection and tracking",
            "- Optimized stereo vision 3D trajectory reconstruction",
            "- Advanced aerodynamic trajectory prediction with EKF",
            "- Accurate landing point estimation and visualization",
            "- Automated in/out boundary judgment with court analysis"
        ]

        for i, feature in enumerate(features):
            cv2.putText(splash, feature, (200, 200 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 255), 2)

        # 新增功能说明
        cv2.putText(splash, "NEW FEATURES v5.0:",
                    (200, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        new_features = [
            "- Non-blocking 3D visualization with smooth integration",
            "- Enhanced keyboard control and window management",
            "- Optimized rendering performance and memory usage",
            "- Improved trajectory quality assessment and filtering",
            "- Real-time multi-threaded processing architecture"
        ]

        for i, feature in enumerate(new_features):
            cv2.putText(splash, feature, (200, 460 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

        # 底部提示
        cv2.putText(splash, "Press any key to continue...",
                    (480, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

        cv2.namedWindow("Badminton Analysis System v5.0", cv2.WINDOW_NORMAL)
        cv2.imshow("Badminton Analysis System v5.0", splash)

        start_time = time.time()
        while time.time() - start_time < duration:
            if cv2.waitKey(100) != -1:
                break

        cv2.destroyWindow("Badminton Analysis System v5.0")

    @staticmethod
    def display_help_screen():
        """显示帮助界面 - 更新空格键功能说明"""
        help_screen = np.zeros((800, 1280, 3), dtype=np.uint8)

        # 标题
        cv2.putText(help_screen, "HELP & INSTRUCTIONS v5.0 (Pause-to-Predict Mode)",
                    (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.line(help_screen, (180, 70), (1100, 70), (0, 120, 255), 2)

        # 基本控制
        basic_shortcuts = [
            "BASIC CONTROLS:",
            "",
            "SPACE - Pause video playback (press when you see the ball)",
            "T     - Trigger trajectory analysis (only when paused)",
            "P     - Resume video playback (when paused)",
            "V     - Toggle 3D visualization window (open/close)",
            "Q     - Close 3D visualization window",
            "H     - Show/Hide this help screen",
            "R     - Reset buffer and system state",
            "+/=   - Increase playback speed",
            "-/_   - Decrease playback speed",
            "0     - Reset playback speed to 1.0x",
            "ESC   - Exit program"
        ]

        y_offset = 100
        for i, shortcut in enumerate(basic_shortcuts):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            font_size = 0.65 if i == 0 else 0.55
            cv2.putText(help_screen, shortcut, (180, y_offset + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2)

        # 工作流程说明
        workflow_info = [
            "RECOMMENDED WORKFLOW:",
            "",
            "1. Watch the video as it plays automatically",
            "2. When you see a shuttlecock trajectory, press SPACE to pause",
            "3. Press T to trigger trajectory analysis and prediction",
            "4. Check console output for prediction results",
            "5. Press V to open 3D visualization for detailed analysis",
            "6. Use keys 1-4 to toggle different point types in 3D view",
            "7. Press D for detailed debug statistics",
            "8. Press P to resume video playback and continue",
            "",
            "This pause-first approach ensures you capture the exact",
            "moment when the shuttlecock trajectory is visible."
        ]

        y_offset = 390
        for i, info in enumerate(workflow_info):
            color = (255, 255, 0) if i == 0 else (200, 255, 200)
            font_size = 0.6 if i == 0 else 0.5
            cv2.putText(help_screen, info, (180, y_offset + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)

        # 调试控制 - 移到右下角
        debug_shortcuts = [
            "DEBUG VISUALIZATION:",
            "D - Debug statistics",
            "1 - Toggle rejected points (red)",
            "2 - Toggle low quality points (orange)",
            "3 - Toggle triangulation failed (gray)",
            "4 - Toggle predicted trajectory (blue)"
        ]

        y_offset = 600
        for i, shortcut in enumerate(debug_shortcuts):
            color = (255, 200, 0) if i == 0 else (180, 180, 255)
            font_size = 0.55 if i == 0 else 0.45
            cv2.putText(help_screen, shortcut, (180, y_offset + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)

        cv2.putText(help_screen, "Press any key to continue...",
                    (520, 770), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.namedWindow("Help & Instructions", cv2.WINDOW_NORMAL)
        cv2.imshow("Help & Instructions", help_screen)
        cv2.waitKey(0)
        cv2.destroyWindow("Help & Instructions")
# 创建全局配置实例
config = Config()