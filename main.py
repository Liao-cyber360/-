import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Must be set before importing other libraries!
import cv2
import numpy as np
import time
import argparse
from collections import deque

# Import custom modules
from utils import config, UIHelper, CourtVisualizer
from calibration import calibrate_cameras
from detector import ShuttlecockDetector, StereoProcessor
from predictor import TrajectoryPredictor, CourtBoundaryAnalyzer


class BadmintonSystem:
    """羽毛球落点预测与界内判定系统"""

    def __init__(self):
        """初始化系统"""
        # 视频路径
        self.video_path1 = None
        self.video_path2 = None

        # 相机标定文件
        self.camera1_params_file = None
        self.camera2_params_file = None

        # 检测器
        self.detector1 = None
        self.detector2 = None

        # 双目处理器
        self.stereo_processor = None

        # 轨迹预测器
        self.trajectory_predictor = None

        # 场地边界分析器
        self.court_analyzer = None

        # 场地可视化器
        self.court_visualizer = None

        # 系统状态
        self.running = False
        self.calibration_done = False
        self.prediction_requested = False
        self.prediction_in_progress = False
        self.prediction_result = None

        # 视频捕获
        self.cap1 = None
        self.cap2 = None

        # 视频帧计数
        self.frame_count1 = 0
        self.frame_count2 = 0

        # 帧率控制
        self.fps_counter = 0
        self.fps_prev_time = time.time()
        self.fps = 0

        # 时间戳数据
        self.timestamps1 = []
        self.timestamps2 = []

    def initialize_system(self, video_path1, video_path2, 
                          timestamps_file1=None, timestamps_file2=None):
        """初始化系统"""
        print("Initializing system...")

        # 显示启动画面
        UIHelper.display_splash_screen(3)

        # 存储视频路径
        self.video_path1 = video_path1
        self.video_path2 = video_path2

        # 加载时间戳
        if timestamps_file1 and timestamps_file2:
            self.load_timestamps(timestamps_file1, timestamps_file2)

        # 打开视频捕获
        self.cap1 = cv2.VideoCapture(video_path1)
        self.cap2 = cv2.VideoCapture(video_path2)

        # 检查视频是否成功打开
        if not self.cap1.isOpened():
            raise ValueError(f"Could not open video file: {video_path1}")
        if not self.cap2.isOpened():
            raise ValueError(f"Could not open video file: {video_path2}")

        # 创建检测器
        self.detector1 = ShuttlecockDetector(config.yolo_ball_model, camera_id=0)
        self.detector2 = ShuttlecockDetector(config.yolo_ball_model, camera_id=1)

        # 创建双目处理器
        self.stereo_processor = StereoProcessor()

        # 创建轨迹预测器
        self.trajectory_predictor = TrajectoryPredictor()

        # 创建场地边界分析器
        self.court_analyzer = CourtBoundaryAnalyzer()

        # 创建场地可视化器
        self.court_visualizer = CourtVisualizer(
            width=config.court_view_width,
            height=config.court_view_height
        )

        print("System initialization complete!")
        return True

    def load_timestamps(self, file1, file2):
        """加载时间戳文件"""
        self.timestamps1 = self._read_timestamp_file(file1)
        self.timestamps2 = self._read_timestamp_file(file2)

        print(f"Loaded {len(self.timestamps1)} timestamps for camera 1")
        print(f"Loaded {len(self.timestamps2)} timestamps for camera 2")

    def _read_timestamp_file(self, file_path):
        """读取时间戳文件"""
        timestamps = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    ts = line.strip()
                    if ts:
                        timestamps.append(int(ts))
            return timestamps
        except Exception as e:
            print(f"Error reading timestamp file {file_path}: {e}")
            return []

    def calibration_mode(self):
        """相机标定模式"""
        print("Starting camera calibration mode...")

        # 创建输出目录
        output_dir = os.path.join(config.results_dir, "calibration")
        os.makedirs(output_dir, exist_ok=True)

        # 执行标定
        try:
            extrinsic_file1, extrinsic_file2 = calibrate_cameras(
                self.video_path1, self.video_path2, output_dir
            )

            # 保存标定文件路径
            self.camera1_params_file = extrinsic_file1
            self.camera2_params_file = extrinsic_file2

            # 加载标定结果到检测器
            self.detector1.load_camera_params(extrinsic_file1)
            self.detector2.load_camera_params(extrinsic_file2)

            # 加载标定结果到双目处理器
            self.stereo_processor.load_camera_parameters(extrinsic_file1, extrinsic_file2)

            # 更新标定状态
            self.calibration_done = True

            print("Calibration completed successfully!")
            return True

        except Exception as e:
            print(f"Calibration failed: {e}")
            return False

    # 在 BadmintonSystem 类中的 predict_landing_point 方法中修改

    def predict_landing_point(self):
        """预测羽毛球落点"""
        try:
            # 标记预测状态
            self.prediction_in_progress = True
            print("Starting landing prediction process...")

            # 从两个相机获取最近的二维轨迹
            trajectory1, timestamps1 = self.detector1.get_recent_trajectory(
                config.prediction_time_window
            )

            trajectory2, timestamps2 = self.detector2.get_recent_trajectory(
                config.prediction_time_window
            )

            # 检查是否有足够的点
            if len(trajectory1) < 5 or len(trajectory2) < 5:
                print(
                    f"Not enough trajectory points for prediction: camera1={len(trajectory1)}, camera2={len(trajectory2)} points")
                self.prediction_in_progress = False
                return None, None

            # 1. 筛选出目标场地上方的轨迹点
            print("Filtering trajectories to focus on the target court...")
            filtered_traj1, filtered_ts1, filtered_traj2, filtered_ts2 = self.stereo_processor.filter_trajectory_by_court(
                trajectory1, timestamps1, trajectory2, timestamps2
            )

            # 检查筛选后是否有足够的点
            if len(filtered_traj1) < 5 or len(filtered_traj2) < 5:
                print(
                    f"Not enough valid trajectory points after filtering: camera1={len(filtered_traj1)}, camera2={len(filtered_traj2)} points")
                # 尝试使用原始轨迹
                filtered_traj1, filtered_ts1 = trajectory1, timestamps1
                filtered_traj2, filtered_ts2 = trajectory2, timestamps2

            # 2. 对二维轨迹进行三维重建
            print("Reconstructing 3D trajectory...")
            trajectory_3d, timestamps_3d = self.stereo_processor.reconstruct_3d_trajectory(
                filtered_traj1, filtered_ts1, filtered_traj2, filtered_ts2
            )

            # 检查是否成功重建3D轨迹
            if len(trajectory_3d) < 5:
                print(f"Failed to reconstruct 3D trajectory, only {len(trajectory_3d)} points available")
                self.prediction_in_progress = False
                return None, None

            # 3. 识别自然下落段
            print("Identifying natural falling segment...")
            natural_segment, natural_timestamps = self.detector1.identify_natural_falling_segment(
                trajectory_3d, timestamps_3d
            )

            if len(natural_segment) < 5:
                print("No suitable natural falling segment identified, using entire trajectory")
                natural_segment, natural_timestamps = trajectory_3d, timestamps_3d

            # 4. 使用轨迹预测器预测落点
            print("Predicting landing point using natural falling segment...")
            landing_position, landing_time, trajectory = self.trajectory_predictor.predict_landing_point(
                natural_segment, natural_timestamps
            )

            if landing_position is None:
                print("Landing prediction failed")
                self.prediction_in_progress = False
                return None, None

            print(f"Predicted landing position: {landing_position}")

            # 判断是否界内 (默认单打)
            in_bounds = self.court_analyzer.is_point_in_court(landing_position, 'singles')
            print(f"Landing point is {'IN' if in_bounds else 'OUT'} of bounds")

            # 保存预测结果
            self.prediction_result = {
                'landing_point': landing_position,
                'landing_time': landing_time,
                'trajectory': trajectory,
                'in_bounds': in_bounds
            }

            # 显示预测结果
            UIHelper.show_prediction_result(landing_position, in_bounds)

            # 重置预测状态
            self.prediction_in_progress = False
            self.prediction_requested = False

            # 返回结果
            return landing_position, in_bounds

        except Exception as e:
            import traceback
            print(f"Error in predict_landing_point: {e}")
            print(traceback.format_exc())
            self.prediction_in_progress = False
            self.prediction_requested = False
            return None, None

    def start_processing(self):
        """开始处理视频"""
        if not self.calibration_done:
            print("Warning: Cameras not calibrated. Functionality will be limited.")

        # 创建显示窗口
        cv2.namedWindow('Camera 1', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Camera 2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera 1', 640, 480)
        cv2.resizeWindow('Camera 2', 640, 480)

        # 创建球场俯视窗口
        cv2.namedWindow('Court View', cv2.WINDOW_NORMAL)
        display_width = int(config.court_view_width * config.display_scale)
        display_height = int(config.court_view_height * config.display_scale)
        cv2.resizeWindow('Court View', display_width, display_height)

        # 设置处理状态
        self.running = True

        # 重置帧计数
        self.frame_count1 = 0
        self.frame_count2 = 0

        # 重置检测器
        if self.detector1 and self.detector2:
            self.detector1.reset_trajectory()
            self.detector2.reset_trajectory()

        # 重置双目处理器
        if self.stereo_processor:
            self.stereo_processor.reset()

        # 添加一个标志位，用于防止连续触发预测
        self.last_prediction_time = 0
        self.prediction_cooldown = 2.0  # 2秒冷却时间

        # 主处理循环
        while self.running:
            # 读取帧
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()

            if not ret1 and not ret2:
                print("End of videos reached")
                break

            # 计算FPS
            current_time = time.time()
            self.fps_counter += 1

            if current_time - self.fps_prev_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.fps_prev_time = current_time

            # 获取时间戳
            timestamp1 = current_time
            timestamp2 = current_time

            if self.timestamps1 and self.frame_count1 < len(self.timestamps1):
                timestamp1 = self.timestamps1[self.frame_count1]

            if self.timestamps2 and self.frame_count2 < len(self.timestamps2):
                timestamp2 = self.timestamps2[self.frame_count2]

            # 处理相机1
            shuttlecock_pos1 = None
            landing1 = False

            if ret1:
                self.frame_count1 += 1
                processed1, shuttlecock_pos1, landing1 = self.detector1.detect(frame1, timestamp1)

                # 添加状态栏
                detection_status = (shuttlecock_pos1 is not None)
                prediction_status = "idle"
                if self.prediction_requested:
                    prediction_status = "ready"
                elif self.prediction_in_progress:
                    prediction_status = "in_progress"
                elif self.prediction_result:
                    prediction_status = "completed"

                processed1 = UIHelper.create_status_bar(
                    processed1, self.fps, self.frame_count1,
                    detection_status, prediction_status
                )

                # 显示相机1帧
                cv2.imshow('Camera 1', processed1)

            # 处理相机2
            shuttlecock_pos2 = None
            landing2 = False

            if ret2:
                self.frame_count2 += 1
                processed2, shuttlecock_pos2, landing2 = self.detector2.detect(frame2, timestamp2)

                # 添加状态栏
                detection_status = (shuttlecock_pos2 is not None)
                prediction_status = "idle"
                if self.prediction_requested:
                    prediction_status = "ready"
                elif self.prediction_in_progress:
                    prediction_status = "in_progress"
                elif self.prediction_result:
                    prediction_status = "completed"

                processed2 = UIHelper.create_status_bar(
                    processed2, self.fps, self.frame_count2,
                    detection_status, prediction_status
                )

                # 显示相机2帧
                cv2.imshow('Camera 2', processed2)

            # 双目处理 (如果两个相机都检测到了羽毛球)
            if shuttlecock_pos1 is not None and shuttlecock_pos2 is not None:
                # 使用最小的时间戳
                timestamp = min(timestamp1, timestamp2)
                point_3d = self.stereo_processor.process_stereo_points(shuttlecock_pos1, shuttlecock_pos2, timestamp)

            # 检测落地是否发生
            if landing1 or landing2:
                # 显示提示信息
                self._show_landing_detection_prompt()

            # 显示球场俯视图
            if self.prediction_result:
                court_view = self.court_visualizer.draw_trajectory_and_landing(
                    self.prediction_result['trajectory'],
                    self.prediction_result['landing_point'],
                    self.prediction_result['in_bounds']
                )
            else:
                court_view = self.court_visualizer.court_image.copy()

            # 调整大小并显示
            resized_court = cv2.resize(court_view, (display_width, display_height))
            cv2.imshow('Court View', resized_court)

            # 处理按键事件
            key = cv2.waitKey(1) & 0xFF

            # ESC退出
            if key == 27:
                self.running = False

            # 空格键触发落点预测
            elif key == ord(' '):
                # 检查是否在冷却期
                if current_time - self.last_prediction_time > self.prediction_cooldown:
                    self.prediction_requested = True
                    # 直接调用预测方法
                    landing_point, in_bounds = self.predict_landing_point()
                    # 更新最后预测时间
                    self.last_prediction_time = current_time

            # H键显示帮助
            elif key == ord('h'):
                UIHelper.display_help_screen()

            # R键重置轨迹
            elif key == ord('r'):
                self.detector1.reset_trajectory()
                self.detector2.reset_trajectory()
                self.stereo_processor.reset()
                self.prediction_result = None
                print("Trajectory data reset")

        # 清理资源
        self._cleanup()

    def _show_landing_detection_prompt(self):
        """显示落地检测提示"""
        # 创建提示窗口
        prompt = np.ones((200, 400, 3), dtype=np.uint8) * 240  # 灰色背景

        # 添加标题和提示
        cv2.putText(prompt, "Shuttlecock Landing Detected",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(prompt, "Press SPACE to predict landing point",
                    (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.putText(prompt, "or wait 3 seconds to continue...",
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 显示提示
        cv2.namedWindow("Landing Detection", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Landing Detection", 500, 300)
        cv2.imshow("Landing Detection", prompt)

        # 等待响应
        start_time = time.time()
        while time.time() - start_time < 3.0:  # 最多显示3秒
            key = cv2.waitKey(100)
            if key == ord(' '):
                self.prediction_requested = True
                break

        cv2.destroyWindow("Landing Detection")

    def _cleanup(self):
        """清理资源"""
        # 释放视频捕获
        if self.cap1 is not None:
            self.cap1.release()
        if self.cap2 is not None:
            self.cap2.release()

        # 关闭所有窗口
        cv2.destroyAllWindows()

        print("System shutdown complete")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Badminton Shuttlecock Landing Prediction System")

    parser.add_argument("--video1", type=str, required=True,
                        help="Path to first camera video file")
    parser.add_argument("--video2", type=str, required=True,
                        help="Path to second camera video file")
    parser.add_argument("--timestamps1", type=str, default=None,
                        help="Path to first camera timestamps file")
    parser.add_argument("--timestamps2", type=str, default=None,
                        help="Path to second camera timestamps file")
    parser.add_argument("--calibrated", action="store_true",
                        help="Skip calibration if cameras are already calibrated")
    parser.add_argument("--cam1_params", type=str, default=None,
                        help="Path to camera 1 parameters file (if calibrated)")
    parser.add_argument("--cam2_params", type=str, default=None,
                        help="Path to camera 2 parameters file (if calibrated)")

    return parser.parse_args()


def main():
    """主函数"""
    try:
        print("Badminton Shuttlecock Landing Prediction and Boundary Judgment System")
        print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 解析命令行参数
        args = parse_arguments()

        # 创建系统实例
        system = BadmintonSystem()

        # 初始化系统
        system.initialize_system(
            args.video1, args.video2,
            args.timestamps1, args.timestamps2
        )

        # 如果已有标定参数，直接加载
        if args.calibrated and args.cam1_params and args.cam2_params:
            print("Loading existing calibration parameters...")
            system.camera1_params_file = args.cam1_params
            system.camera2_params_file = args.cam2_params
            system.detector1.load_camera_params(args.cam1_params)
            system.detector2.load_camera_params(args.cam2_params)
            system.stereo_processor.load_camera_parameters(args.cam1_params, args.cam2_params)
            system.calibration_done = True
        else:
            # 否则进入标定模式
            print("Entering calibration mode...")
            print("Please follow the instructions on screen to calibrate the cameras.")
            system.calibration_mode()

        # 开始处理视频
        system.start_processing()

    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()