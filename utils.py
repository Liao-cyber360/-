import os
import time
import cv2
import numpy as np


class Config:
    """配置类,集中管理所有系统参数"""
    def __init__(self):
        # 相机参数
        self.camera_params_file_1 = "E:\hawkeye\calibration\calibration_results_2025-08-11_18-06-15.yaml"  # 相机1标定参数
        self.camera_params_file_2 = "E:\hawkeye\calibration\calibration_results_2025-08-11_18-06-15.yaml" # 相机2标定参数
        
        # 模型参数
        self.yolo_ball_model = "E:\\hawkeye\\ball\\best.pt"  # 羽毛球检测模型
        self.yolo_court_model = "E:\\hawkeye\\field\\best.pt"  # 场地检测模型
        
        # 视频参数
        self.video_width = 1280  # 视频宽度
        self.video_height = 720  # 视频高度
        self.fps = 30  # 视频帧率
        
        # 分析参数
        self.trajectory_buffer_size = 300  # 轨迹缓存大小
        self.landing_analysis_window = 10 # 落地分析窗口大小（帧数）
        self.prediction_time_window = 2  # 预测使用的时间窗口（秒）
        self.poly_fit_degree = 4  # 多项式拟合次数
        
        # 物理模型参数
        self.shuttlecock_mass = 0.005  # 羽毛球质量(kg)
        self.shuttlecock_radius = 0.025  # 羽毛球半径(m)
        self.air_density = 1.225  # 空气密度(kg/m³)
        self.drag_coefficient = 0.6  # 阻力系数
        self.gravity = 9.8  # 重力加速度(m/s²)
        
        # 计算空气动力学长度
        self.cross_section = np.pi * self.shuttlecock_radius ** 2
        self.aero_length = 2 * self.shuttlecock_mass / (self.air_density * self.cross_section * self.drag_coefficient)
        
        # EKF参数
        self.ekf_process_noise = 0.01  # 过程噪声
        self.ekf_measurement_noise = 0.1  # 测量噪声
        
        # 系统参数
        self.results_dir = f"./results_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 界面参数
        self.court_view_width = 610  # 场地视图宽度(cm)
        self.court_view_height = 1340  # 场地视图高度(cm)
        self.display_scale = 0.5  # 显示比例
        
        # 判定参数
        self.landing_detection_threshold = 5  # 落地判定速度阈值(像素/帧)
        self.landing_confirmation_frames = 3  # 连续多少帧低于速度阈值判定为落地
        self.landing_height_threshold = 10.0  # 落地判定高度阈值(厘米)
    
    def get_aero_params(self):
        """返回空气动力学参数"""
        return {
            'mass': self.shuttlecock_mass,
            'gravity': self.gravity,
            'aero_length': self.aero_length
        }


class UIHelper:
    """用户界面辅助类,提供各种UI界面生成功能"""
    
    @staticmethod
    def create_status_bar(frame, fps, frame_count, detection_status, prediction_status):
        """创建状态栏"""
        h, w = frame.shape[:2]
        
        # 创建状态栏区域
        status_bar = frame.copy()
        cv2.rectangle(status_bar, (0, h-40), (w, h), (0, 0, 0), -1)
        
        # 添加FPS信息
        cv2.putText(status_bar, f"FPS: {fps:.1f}", (10, h-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 添加帧计数
        cv2.putText(status_bar, f"Frame: {frame_count}", (120, h-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 添加检测状态
        color = (0, 255, 0) if detection_status else (0, 0, 255)
        status = "Detected" if detection_status else "Not Detected"
        cv2.putText(status_bar, f"Shuttlecock: {status}", (250, h-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 添加预测状态
        if prediction_status == "ready":
            color = (255, 255, 0)
            text = "Prediction: Ready (Press SPACE)"
        elif prediction_status == "in_progress":
            color = (0, 255, 255)
            text = "Prediction: Processing..."
        elif prediction_status == "completed":
            color = (0, 255, 0)
            text = "Prediction: Completed"
        else:
            color = (150, 150, 150)
            text = "Prediction: Idle"
            
        cv2.putText(status_bar, text, (450, h-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 添加帮助提示
        cv2.putText(status_bar, "Press H for Help", (w-150, h-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)
        
        return status_bar
    
    @staticmethod
    def display_splash_screen(duration=3):
        """显示启动画面"""
        splash = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # 添加标题
        cv2.putText(splash, "Badminton Shuttlecock Landing Prediction System",
                   (180, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # 添加分割线
        cv2.line(splash, (180, 120), (1100, 120), (0, 120, 255), 2)
        
        # 添加功能介绍
        features = [
            "- Real-time shuttlecock detection and tracking",
            "- Stereo vision 3D trajectory reconstruction",
            "- Advanced aerodynamic trajectory prediction",
            "- Accurate landing point estimation",
            "- Automated in/out boundary judgment"
        ]
        
        for i, feature in enumerate(features):
            cv2.putText(splash, feature, (250, 200 + i * 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 255), 2)
        
        # 添加操作说明
        cv2.putText(splash, "Press any key to continue...", 
                   (480, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        
        cv2.namedWindow("Badminton Analysis System", cv2.WINDOW_NORMAL)
        cv2.imshow("Badminton Analysis System", splash)
        
        # 等待指定时间或按键
        start_time = time.time()
        while time.time() - start_time < duration:
            if cv2.waitKey(100) != -1:
                break
                
        cv2.destroyWindow("Badminton Analysis System")
    
    @staticmethod
    def display_help_screen():
        """显示帮助界面"""
        help_screen = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # 添加标题
        cv2.putText(help_screen, "HELP & INSTRUCTIONS",
                   (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # 添加分割线
        cv2.line(help_screen, (280, 80), (1000, 80), (0, 120, 255), 2)
        
        # 添加快捷键说明
        shortcuts = [
            "KEYBOARD SHORTCUTS:",
            "",
            "SPACE - Trigger landing point prediction",
            "H     - Show/Hide this help screen",
            "R     - Reset trajectory data",
            "ESC   - Exit program"
        ]
        
        for i, shortcut in enumerate(shortcuts):
            cv2.putText(help_screen, shortcut, (280, 130 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 添加底部提示
        cv2.putText(help_screen, "Press any key to continue...",
                   (520, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.namedWindow("Help & Instructions", cv2.WINDOW_NORMAL)
        cv2.imshow("Help & Instructions", help_screen)
        
        # 等待按键
        cv2.waitKey(0)
        cv2.destroyWindow("Help & Instructions")
    
    @staticmethod
    def show_prediction_result(landing_position, in_bounds):
        """显示落点预测结果弹窗"""
        # 创建预测结果窗口
        result_screen = np.ones((400, 600, 3), dtype=np.uint8) * 240  # 浅灰色背景

        # 添加标题栏
        title_color = (0, 150, 0) if in_bounds else (0, 0, 150)  # 绿色=界内，红色=界外
        cv2.rectangle(result_screen, (0, 0), (600, 60), title_color, -1)
        cv2.putText(result_screen, "Shuttlecock Landing Prediction",
                    (120, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # 添加结果信息
        status = "IN" if in_bounds else "OUT"
        status_color = (0, 200, 0) if in_bounds else (0, 0, 200)

        cv2.putText(result_screen, f"Status: {status}",
                    (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        cv2.putText(result_screen, f"X-coordinate: {landing_position[0]:.2f} cm",
                    (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

        cv2.putText(result_screen, f"Y-coordinate: {landing_position[1]:.2f} cm",
                    (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

        # 添加按钮提示
        cv2.rectangle(result_screen, (200, 330), (400, 370), (200, 200, 200), -1)
        cv2.putText(result_screen, "Press any key to close",
                    (220, 358), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        # 显示结果窗口
        cv2.namedWindow("Prediction Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Prediction Result", result_screen)
        cv2.waitKey(0)
        cv2.destroyWindow("Prediction Result")


class CourtVisualizer:
    """羽毛球场地可视化"""
    def __init__(self, width=610, height=1340):
        """初始化场地可视化器"""
        self.width = width
        self.height = height
        self.court_image = self._create_court_image()
        
    def _create_court_image(self):
        """创建球场图像"""
        # 创建空白图像
        court = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255  # 白色背景
        
        # 绘制场地线
        # 外框线
        cv2.rectangle(court, (0, 0), (self.width, self.height), (0, 0, 0), 2)
        
        # 中线(网)
        cv2.line(court, (0, self.height//2), (self.width, self.height//2), (0, 0, 0), 2)
        
        # 单打边线
        cv2.line(court, (46, 0), (46, self.height), (0, 0, 255), 1)  # 左单打线
        cv2.line(court, (564, 0), (564, self.height), (0, 0, 255), 1)  # 右单打线
        
        # 发球线
        cv2.line(court, (0, 76), (self.width, 76), (0, 0, 255), 1)  # 后发球线(下)
        cv2.line(court, (0, 468), (self.width, 468), (0, 0, 255), 1)  # 前发球线(下)
        cv2.line(court, (0, self.height-76), (self.width, self.height-76), (0, 0, 255), 1)  # 后发球线(上)
        cv2.line(court, (0, self.height-468), (self.width, self.height-468), (0, 0, 255), 1)  # 前发球线(上)
        
        # 中央分界线
        cv2.line(court, (self.width//2, 0), (self.width//2, self.height), (0, 0, 255), 1)  
        
        return court
        
    def draw_trajectory_and_landing(self, trajectory_points, landing_point, in_bounds):
        """绘制轨迹和落点"""
        display = self.court_image.copy()
        
        # 绘制轨迹
        if trajectory_points:
            points_2d = [(int(p[0]), int(p[1])) for p in trajectory_points]
            
            # 绘制轨迹线
            for i in range(len(points_2d)-1):
                ratio = i / (len(points_2d)-1)
                # 从蓝色渐变到红色
                b = int(255 * (1 - ratio))
                r = int(255 * ratio)
                g = 0
                
                pt1 = points_2d[i]
                pt2 = points_2d[i+1]
                
                # 确保点在图像范围内
                if (0 <= pt1[0] < self.width and 0 <= pt1[1] < self.height and
                    0 <= pt2[0] < self.width and 0 <= pt2[1] < self.height):
                    cv2.line(display, pt1, pt2, (b, g, r), 2)
        
        # 绘制落点
        if landing_point is not None:
            # 获取x,y坐标
            x, y = int(landing_point[0]), int(landing_point[1])
            
            # 保证坐标在图像范围内
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))
            
            # 判断是否界内
            color = (0, 255, 0) if in_bounds else (0, 0, 255)  # 绿色=界内，红色=界外
            
            # 绘制落点
            cv2.circle(display, (x, y), 10, color, -1)
            cv2.circle(display, (x, y), 12, (0, 0, 0), 2)
            
            # 添加文本标签
            label = "IN" if in_bounds else "OUT"
            cv2.putText(display, label, (x + 15, y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display, f"({x},{y})", (x + 15, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return display


# 创建全局配置实例
config = Config()