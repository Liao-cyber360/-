import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
from utils import config


class LandingDetector:
    """羽毛球落地检测器"""

    def __init__(self, threshold=5, confirmation_frames=3, height_threshold=20.0):
        """
        初始化落地检测器

        参数:
            threshold: 速度阈值，低于此值视为可能落地（像素/帧）
            confirmation_frames: 连续多少帧低于阈值判定为落地
            height_threshold: 高度阈值，Z坐标低于此值视为接近地面（单位：厘米）
        """
        self.threshold = threshold
        self.confirmation_frames = confirmation_frames
        self.height_threshold = height_threshold  # 高度阈值参数
        self.previous_positions = deque(maxlen=10)  # 存储最近的位置
        self.previous_3d_positions = deque(maxlen=10)  # 存储最近的3D位置
        self.slow_frame_counter = 0  # 连续慢速帧计数器
        self.last_landing_time = 0  # 上次检测到落地的时间
        self.landing_cooldown = 15.0  # 落地检测冷却时间（秒）

    def detect_landing(self, position, timestamp, position_3d=None):
        """
        检测羽毛球是否落地

        参数:
            position: 当前球的2D位置 (x, y)
            timestamp: 当前时间戳
            position_3d: 当前球的3D位置 (x, y, z)，如果有的话

        返回:
            landing_detected: 是否检测到落地
        """
        # 冷却时间检查（避免短时间内重复触发）
        if timestamp - self.last_landing_time < self.landing_cooldown:
            return False

        # 添加当前位置
        self.previous_positions.append((position, timestamp))
        if position_3d is not None:
            self.previous_3d_positions.append((position_3d, timestamp))

        # 至少需要2个点才能计算速度
        if len(self.previous_positions) < 2:
            return False

        # 计算当前2D速度
        curr_pos, curr_time = self.previous_positions[-1]
        prev_pos, prev_time = self.previous_positions[-2]

        # 避免时间差为0
        time_diff = max(curr_time - prev_time, 0.001)

        # 计算位移和速度
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        displacement = np.sqrt(dx ** 2 + dy ** 2)
        speed = displacement / time_diff  # 像素/秒

        # 判断是否为低速
        is_slow = speed < self.threshold

        # 判断高度是否接近地面
        is_near_ground = False
        if position_3d is not None and len(position_3d) >= 3:
            z_coord = position_3d[2]
            is_near_ground = z_coord <= self.height_threshold
        elif len(self.previous_3d_positions) > 0:
            # 如果当前帧没有3D位置但之前有，使用最近的3D位置
            latest_3d_pos = self.previous_3d_positions[-1][0]
            if len(latest_3d_pos) >= 3:
                z_coord = latest_3d_pos[2]
                is_near_ground = z_coord <= self.height_threshold

        # 同时满足低速和接近地面的条件时，计数器增加
        if is_slow and (is_near_ground or position_3d is None):
            self.slow_frame_counter += 1
        else:
            self.slow_frame_counter = 0

        # 如果连续多帧同时满足条件，判定为落地
        if self.slow_frame_counter >= self.confirmation_frames:
            self.last_landing_time = timestamp
            self.slow_frame_counter = 0
            return True

        return False

    def reset(self):
        """重置检测器状态"""
        self.previous_positions.clear()
        self.previous_3d_positions.clear()
        self.slow_frame_counter = 0


class ShuttlecockDetector:
    """羽毛球检测与跟踪器"""

    def __init__(self, model_path, camera_id=0):
        """
        初始化检测器

        参数:
            model_path: YOLOv8模型路径
            camera_id: 相机编号（0或1）
        """
        # 加载YOLO模型
        self.model = YOLO(model_path)
        self.camera_id = camera_id

        # 轨迹存储
        self.max_trajectory_length = config.trajectory_buffer_size
        self.trajectory = deque(maxlen=self.max_trajectory_length)  # 轨迹点 [(点,时间戳),...]
        self.raw_detections = deque(maxlen=self.max_trajectory_length * 2)  # 原始检测结果 [(点列表,时间戳,帧ID),...]

        # 轨迹颜色渐变 (BGR格式)
        self.start_color = (255, 0, 0)  # 蓝色 (最旧的位置)
        self.end_color = (0, 0, 255)  # 红色 (最新的位置)

        # 落地判断
        self.landing_detector = LandingDetector(
            threshold=config.landing_detection_threshold,
            confirmation_frames=config.landing_confirmation_frames,
            height_threshold=config.landing_height_threshold
        )

        # 球的3D位置历史记录
        self.positions_3d = deque(maxlen=self.max_trajectory_length)
        self.timestamps = deque(maxlen=self.max_trajectory_length)

        # 相机投影参数
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rotation_vector = None
        self.translation_vector = None

        # 移除 court_mask 成员变量

        # 轨迹连续性判断阈值
        self.continuity_threshold = 50  # 相邻帧的最大位移距离
        self.frame_interval_multiplier = 1.5  # 间隔帧位移阈值的乘数
        self.last_frame_id = -1  # 上一帧的ID
        self.last_position = None  # 上一帧的位置

    def load_camera_params(self, params_file):
        """从文件加载相机参数"""
        fs = cv2.FileStorage(params_file, cv2.FILE_STORAGE_READ)
        self.camera_matrix = fs.getNode("camera_matrix").mat()
        self.dist_coeffs = fs.getNode("distortion_coefficients").mat().flatten()
        self.rotation_vector = fs.getNode("rotation_vector").mat()
        self.translation_vector = fs.getNode("translation_vector").mat()
        fs.release()

        print(f"Camera {self.camera_id} parameters loaded")

    def detect(self, frame, timestamp, frame_id=None):
        """
        在帧中检测羽毛球

        参数:
            frame: 当前帧
            timestamp: 时间戳
            frame_id: 帧ID，如果没有提供，则使用内部计数

        返回:
            display_frame: 可视化后的帧
            shuttlecock_pos: 选择的羽毛球位置
            landing_detected: 是否检测到落地
        """
        if frame_id is None:
            frame_id = self.last_frame_id + 1

        self.last_frame_id = frame_id

        # 直接使用原始帧进行检测，不应用掩码
        # 使用YOLO进行预测
        results = self.model(frame, conf=0.3)  # 设置置信度阈值

        # 检测结果可视化框架
        display_frame = frame.copy()

        # 找到所有羽毛球
        all_shuttlecock_pos = []

        for r in results:
            # 检查是否有关键点结果
            if hasattr(r, 'keypoints') and r.keypoints is not None:
                kpts = r.keypoints.xy.cpu().numpy() if hasattr(r.keypoints.xy, "cpu") else r.keypoints.xy
                for kp_list in kpts:
                    for kp in kp_list:
                        if not np.isnan(kp).any():  # 排除无效点
                            all_shuttlecock_pos.append((int(kp[0]), int(kp[1])))

            # 如果没有关键点结果，使用边界框
            if len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else r.boxes.xyxy
                classes = r.boxes.cls.cpu().numpy() if hasattr(r.boxes.cls, "cpu") else r.boxes.cls

                # 假设羽毛球类别ID是0
                shuttlecock_indices = np.where(classes == 0)[0]

                for idx in shuttlecock_indices:
                    box = boxes[idx]
                    x1, y1, x2, y2 = map(int, box)

                    # 计算球头中心点
                    all_shuttlecock_pos.append(((x1 + x2) // 2, (y1 + y2) // 2))

        # 保存所有检测结果
        self.raw_detections.append((all_shuttlecock_pos, timestamp, frame_id))

        # 选择最可能的羽毛球位置（基于轨迹的连续性）
        shuttlecock_pos = self._select_best_shuttlecock(all_shuttlecock_pos)

        # 如果检测到羽毛球，添加到轨迹并检查是否落地
        landing_detected = False
        position_3d = None
        if shuttlecock_pos is not None:
            self.trajectory.append((shuttlecock_pos, timestamp, frame_id))
            self.last_position = shuttlecock_pos

            # 如果已有相机参数，进行3D位置估计
            if hasattr(self, 'camera_matrix') and hasattr(self, 'rotation_vector'):
                position_3d = self.estimate_3d_position(shuttlecock_pos)
                if position_3d is not None:
                    self.positions_3d.append((position_3d, timestamp))

            # 进行落地检测，传入3D位置信息
            landing_detected = self.landing_detector.detect_landing(shuttlecock_pos, timestamp, position_3d)

            # 在当前位置绘制一个圆圈
            cv2.circle(display_frame, shuttlecock_pos, 5, (0, 255, 0), -1)

        # 绘制所有检测到的羽毛球（不同颜色标识）
        for pos in all_shuttlecock_pos:
            if pos == shuttlecock_pos:
                continue  # 跳过已经绘制的主要羽毛球
            cv2.circle(display_frame, pos, 5, (0, 165, 255), 1)  # 浅橙色，非填充

        # 绘制轨迹
        self.draw_trajectory(display_frame)

        return display_frame, shuttlecock_pos, landing_detected
    def _select_best_shuttlecock(self, detected_positions):
        """
        从多个检测结果中选择最佳的羽毛球位置

        参数:
            detected_positions: 当前帧检测到的所有羽毛球位置

        返回:
            best_position: 最佳的羽毛球位置
        """
        if not detected_positions:
            return None

        # 如果只有一个检测结果，直接返回
        if len(detected_positions) == 1:
            return detected_positions[0]

        # 如果没有历史轨迹，返回第一个检测结果
        if not self.trajectory:
            return detected_positions[0]

        # 获取上一个轨迹点
        last_pos = self.last_position

        if last_pos is None:
            return detected_positions[0]

        # 计算每个检测结果与上一个点的距离
        distances = [np.linalg.norm(np.array(pos) - np.array(last_pos)) for pos in detected_positions]

        # 找到距离最近的点
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]

        # 检查最小距离是否小于阈值
        if min_dist <= self.continuity_threshold:
            return detected_positions[min_dist_idx]

        # 如果所有点都超过阈值，返回距离最近的点
        return detected_positions[min_dist_idx]

    def estimate_3d_position(self, point_2d):
        """使用投影矩阵估计3D位置（射线法）"""
        if point_2d is None or not hasattr(self, 'camera_matrix') or not hasattr(self, 'rotation_vector'):
            return None

        # 将点从图像坐标转换为归一化相机坐标
        point_2d = np.array([[point_2d[0], point_2d[1]]], dtype=np.float32)
        point_2d_undistorted = cv2.undistortPoints(point_2d, self.camera_matrix, self.dist_coeffs)

        # 构建射线向量
        ray_dir = np.array([point_2d_undistorted[0][0][0],
                            point_2d_undistorted[0][0][1],
                            1.0], dtype=np.float32)

        # 将射线方向从相机坐标系转换到世界坐标系
        rotation_matrix, _ = cv2.Rodrigues(self.rotation_vector)
        ray_dir_world = np.dot(rotation_matrix.T, ray_dir)
        ray_dir_world = ray_dir_world / np.linalg.norm(ray_dir_world)  # 单位化

        # 相机中心在世界坐标系中的位置
        camera_center_world = -np.dot(rotation_matrix.T, self.translation_vector)

        # 假设z=0平面（场地平面）
        # 计算射线与z=0平面的交点
        if abs(ray_dir_world[2]) > 1e-10:  # 避免除以零
            t = -camera_center_world[2] / ray_dir_world[2]
            if t > 0:  # 确保交点在相机前方
                intersection = camera_center_world + t * ray_dir_world
                return (intersection[0][0], intersection[1][0], 0.0)

        return None

    def draw_trajectory(self, frame):
        """在帧上绘制羽毛球轨迹"""
        if len(self.trajectory) > 1:
            points = [t[0] for t in self.trajectory]  # 提取点坐标

            for i in range(len(points) - 1):
                # 计算颜色插值 (老的点是蓝色，新的点是红色)
                alpha = i / (len(points) - 1)
                b = int(self.start_color[0] * (1 - alpha) + self.end_color[0] * alpha)
                g = int(self.start_color[1] * (1 - alpha) + self.end_color[1] * alpha)
                r = int(self.start_color[2] * (1 - alpha) + self.end_color[2] * alpha)

                # 点的大小也随时间变化 (老的点小，新的点大)
                radius = max(3, int(3 + (i * 7 / len(points))))

                # 绘制点
                point = points[i]
                cv2.circle(frame, point, radius, (b, g, r), -1)

                # 连接相邻点
               # next_point = points[i + 1]
                #cv2.line(frame, point, next_point, (b, g, r), max(1, radius // 3))

    def reset_trajectory(self):
        """重置轨迹数据"""
        self.trajectory.clear()
        self.raw_detections.clear()
        self.positions_3d.clear()
        self.timestamps.clear()
        self.landing_detector.reset()
        self.last_frame_id = -1
        self.last_position = None

    def get_recent_trajectory(self, time_window=0.5):
        """获取指定时间窗口内的轨迹点"""
        if not self.trajectory:
            return [], []

        recent_points = []
        recent_timestamps = []
        current_time = self.trajectory[-1][1] if self.trajectory else 0

        for point, timestamp, _ in self.trajectory:
            if current_time - timestamp <= time_window:
                recent_points.append(point)
                recent_timestamps.append(timestamp)

        return recent_points, recent_timestamps

    def get_recent_3d_positions(self, time_window=0.5):
        """获取指定时间窗口内的3D位置点"""
        if not self.positions_3d:
            return [], []

        recent_points = []
        recent_timestamps = []
        current_time = self.positions_3d[-1][1] if self.positions_3d else 0

        for position, timestamp in self.positions_3d:
            if current_time - timestamp <= time_window:
                recent_points.append(position)
                recent_timestamps.append(timestamp)

        return recent_points, recent_timestamps

    def filter_target_court_trajectory(self, stereo_processor):
        """
        筛选出目标场地上方的轨迹点
     按帧吗,,其实应该时间戳,,,
        参数:
            stereo_processor: 双目处理器实例，用于三维重建

        返回:
            filtered_trajectory: 筛选后的轨迹点列表 [(点,时间戳),...]
        """
        if not self.raw_detections or not hasattr(stereo_processor, 'camera1_params'):
            return []

        # 获取所有历史检测结果
        frame_results = {}  # {帧ID: (点列表, 时间戳), ...}
        for points, timestamp, frame_id in self.raw_detections:
            frame_results[frame_id] = (points, timestamp)

        # 逐帧处理，构建连续轨迹
        continuous_trajectory = []
        current_point = None

        # 按帧ID排序
        sorted_frame_ids = sorted(frame_results.keys())

        # 第一次遍历：筛选出目标场地上方的点
        for frame_id in sorted_frame_ids:
            points, timestamp = frame_results[frame_id]

            # 如果当前帧没有检测点，继续下一帧
            if not points:
                continue

            # 如果是第一帧或者当前没有参考点，先用第一个点
            if current_point is None:
                current_point = points[0]
                continuous_trajectory.append((current_point, timestamp, frame_id))
                continue

            # 计算当前帧中每个点与上一个点的距离
            min_dist = float('inf')
            best_point = None

            for point in points:
                dist = np.linalg.norm(np.array(point) - np.array(current_point))
                if dist < min_dist:
                    min_dist = dist
                    best_point = point

            # 检查最小距离是否在阈值范围内
            frame_gap = frame_id - continuous_trajectory[-1][2]  # 与上一个点的帧间隔
            threshold = self.continuity_threshold * (1 + (frame_gap - 1) * self.frame_interval_multiplier)

            if min_dist <= threshold:
                current_point = best_point
                continuous_trajectory.append((current_point, timestamp, frame_id))

        return continuous_trajectory

    def identify_natural_falling_segment(self, trajectory, timestamps=None, frame_ids=None):
        """
        识别自然下落段轨迹

        参数:
            trajectory: 轨迹点列表 [(x,y), ...] 或 [(x,y,时间戳,帧ID), ...]
            timestamps: 时间戳列表 (如果trajectory中没有包含)
            frame_ids: 帧ID列表 (如果trajectory中没有包含)

        返回:
            natural_segment: 自然下落段轨迹点列表
            natural_timestamps: 对应的时间戳列表
        """
        # 检查输入格式
        if not trajectory:
            return [], []

        # 处理输入格式
        points = []
        ts = []
        fids = []

        if len(trajectory[0]) == 2:  # 只包含坐标
            points = trajectory
            ts = timestamps if timestamps else [0] * len(trajectory)
            fids = frame_ids if frame_ids else list(range(len(trajectory)))
        elif len(trajectory[0]) == 3:  # 包含坐标和时间戳
            points = [t[0] for t in trajectory]
            ts = [t[1] for t in trajectory]
            fids = frame_ids if frame_ids else list(range(len(trajectory)))
        elif len(trajectory[0]) == 4 or len(trajectory[0]) == 3:  # 包含坐标、时间戳和帧ID
            points = [t[0] for t in trajectory]
            ts = [t[1] for t in trajectory]
            fids = [t[2] for t in trajectory]

        if len(points) < 5:
            return trajectory, ts

        # 计算相邻点之间的速度和加速度
        velocities = []
        accelerations = []
        directions = []

        for i in range(1, len(points)):
            # 计算位移
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]

            # 计算时间差
            dt = max(ts[i] - ts[i - 1], 0.001)

            # 计算速度向量和大小
            velocity_vector = np.array([dx / dt, dy / dt])
            velocity_magnitude = np.linalg.norm(velocity_vector)
            velocities.append(velocity_magnitude)

            # 保存归一化的方向向量
            if velocity_magnitude > 1e-6:
                directions.append(velocity_vector / velocity_magnitude)
            else:
                directions.append(np.array([0, 0]))

        # 计算加速度和方向变化
        for i in range(1, len(velocities)):
            # 计算速度变化
            dv = velocities[i] - velocities[i - 1]
            dt = max(ts[i + 1] - ts[i], 0.001)
            acceleration = dv / dt
            accelerations.append(acceleration)

        # 计算方向变化
        direction_changes = []
        for i in range(1, len(directions)):
            # 计算方向变化的角度
            dot_product = np.clip(np.dot(directions[i], directions[i - 1]), -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            angle_deg = np.degrees(angle_rad)
            direction_changes.append(angle_deg)

        # 标记外力影响点
        external_force_points = []
        acceleration_threshold = np.std(accelerations) * 2 if accelerations else 0  # 加速度阈值
        direction_threshold = 50.0  # 方向变化阈值

        for i in range(len(direction_changes)):
            # i+1 是速度的索引，i+2 是点的索引
            if i + 2 >= len(points):
                break

            # 检查是否有突然的加速度和明显的方向变化
            if (abs(accelerations[i]) > acceleration_threshold and
                    direction_changes[i] > direction_threshold):
                external_force_points.append(i + 2)

        # 寻找最长的自然下落段（未受外力影响的段）
        segments = []
        current_segment = []

        for i in range(len(points)):
            if i in external_force_points:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
            current_segment.append(i)

        if current_segment:
            segments.append(current_segment)

        # 找出最长的段
        longest_segment = max(segments, key=len) if segments else list(range(len(points)))

        # 提取自然下落段
        natural_indices = longest_segment
        natural_segment = [points[i] for i in natural_indices]
        natural_timestamps = [ts[i] for i in natural_indices]

        return natural_segment, natural_timestamps


class StereoProcessor:
    """双目视觉处理器"""

    def __init__(self):
        """初始化双目处理器"""
        # 相机参数
        self.camera1_params = None
        self.camera2_params = None

        # 3D点历史
        self.points_3d = []
        self.timestamps = []

    def load_camera_parameters(self, camera1_file, camera2_file):
        """加载两个相机的参数"""
        # 加载相机1参数
        self.camera1_params = self._load_camera_params(camera1_file)

        # 加载相机2参数
        self.camera2_params = self._load_camera_params(camera2_file)

        print("Stereo camera parameters loaded successfully")

    def load_camera_params(self, params_file):
        """从文件加载相机参数"""
        fs = cv2.FileStorage(params_file, cv2.FILE_STORAGE_READ)
        self.camera_matrix = fs.getNode("camera_matrix").mat()
        self.dist_coeffs = fs.getNode("distortion_coefficients").mat().flatten()
        self.rotation_vector = fs.getNode("rotation_vector").mat()
        self.translation_vector = fs.getNode("translation_vector").mat()
        fs.release()

        print(f"Camera {self.camera_id} parameters loaded")

        # 移除可能的掩码加载代码

    def triangulate_point(self, point1, point2):
        """
        通过两个相机视图中的点进行三角测量

        参数:
            point1: 相机1中的点 (x, y)
            point2: 相机2中的点 (x, y)

        返回:
            point_3d: 三维点 (x, y, z)
        """
        if self.camera1_params is None or self.camera2_params is None:
            print("Camera parameters not loaded")
            return None

        # 将像素坐标转换为归一化相机坐标
        point1_normalized = cv2.undistortPoints(
            np.array([point1], dtype=np.float32),
            self.camera1_params['camera_matrix'],
            self.camera1_params['dist_coeffs']
        )

        point2_normalized = cv2.undistortPoints(
            np.array([point2], dtype=np.float32),
            self.camera2_params['camera_matrix'],
            self.camera2_params['dist_coeffs']
        )

        # 从归一化坐标创建射线向量
        ray1_dir = np.array([
            point1_normalized[0][0][0],
            point1_normalized[0][0][1],
            1.0
        ])

        ray2_dir = np.array([
            point2_normalized[0][0][0],
            point2_normalized[0][0][1],
            1.0
        ])

        # 将射线方向转换到世界坐标系
        rotation_matrix1, _ = cv2.Rodrigues(self.camera1_params['rotation_vector'])
        rotation_matrix2, _ = cv2.Rodrigues(self.camera2_params['rotation_vector'])

        ray1_dir_world = np.dot(rotation_matrix1.T, ray1_dir)
        ray2_dir_world = np.dot(rotation_matrix2.T, ray2_dir)

        # 标准化方向向量
        ray1_dir_world = ray1_dir_world / np.linalg.norm(ray1_dir_world)
        ray2_dir_world = ray2_dir_world / np.linalg.norm(ray2_dir_world)

        # 相机在世界坐标系中的位置
        camera1_position = self.camera1_params['camera_position']
        camera2_position = self.camera2_params['camera_position']

        # 找到两条射线之间的最短连线中点
        # 射线之间的法向量
        n = np.cross(ray1_dir_world.flatten(), ray2_dir_world.flatten())

        # 如果射线平行，无法三角化
        if np.linalg.norm(n) < 1e-10:
            print("Warning: Rays are parallel, triangulation may be unstable")
            return None

        # 计算参数方程
        n1 = np.cross(ray1_dir_world.flatten(), n)
        n2 = np.cross(ray2_dir_world.flatten(), n)

        # 计算两条射线上的最近点参数
        c1 = camera1_position.flatten()
        c2 = camera2_position.flatten()

        t1 = np.dot((c2 - c1), n2) / np.dot(ray1_dir_world.flatten(), n2)
        t2 = np.dot((c1 - c2), n1) / np.dot(ray2_dir_world.flatten(), n1)

        # 计算射线上的点
        p1 = c1 + t1 * ray1_dir_world.flatten()
        p2 = c2 + t2 * ray2_dir_world.flatten()

        # 取中点作为三维点的估计
        point_3d = (p1 + p2) / 2

        return point_3d

    def process_stereo_points(self, point1, point2, timestamp):
        """
        处理双目视觉点，计算3D坐标

        参数:
            point1: 相机1中的点 (x, y)
            point2: 相机2中的点 (x, y)
            timestamp: 时间戳

        返回:
            point_3d: 三维点 (x, y, z)
        """
        if point1 is None or point2 is None:
            return None

        # 三角测量
        point_3d = self.triangulate_point(point1, point2)

        if point_3d is not None:
            # 存储3D点历史
            self.points_3d.append(point_3d)
            self.timestamps.append(timestamp)

        return point_3d

    def reconstruct_3d_trajectory(self, trajectory1, timestamps1, trajectory2, timestamps2):
        """
        重建两个相机中的轨迹，返回3D轨迹

        参数:
            trajectory1: 相机1中的轨迹点列表
            timestamps1: 相机1中的时间戳列表
            trajectory2: 相机2中的轨迹点列表
            timestamps2: 相机2中的时间戳列表

        返回:
            trajectory_3d: 3D轨迹点列表
            trajectory_timestamps: 对应的时间戳列表
        """
        # 检查轨迹长度
        if len(trajectory1) < 3 or len(trajectory2) < 3:
            print(f"Trajectories too short for reconstruction: camera1={len(trajectory1)}, camera2={len(trajectory2)}")
            return [], []

        # 找到两个相机中时间戳的交集
        trajectory_3d = []
        trajectory_timestamps = []

        for i, ts1 in enumerate(timestamps1):
            # 查找最近的时间戳
            closest_ts2_idx = None
            min_diff = 0.02  # 20ms阈值

            for j, ts2 in enumerate(timestamps2):
                diff = abs(ts1 - ts2)
                if diff < min_diff:
                    min_diff = diff
                    closest_ts2_idx = j

            if closest_ts2_idx is not None:
                # 确保索引在有效范围内
                if i < len(trajectory1) and closest_ts2_idx < len(trajectory2):
                    # 三角测量
                    point_3d = self.triangulate_point(trajectory1[i], trajectory2[closest_ts2_idx])

                    if point_3d is not None:
                        # 额外的检查，确保点有意义
                        if not np.isnan(point_3d).any() and np.all(np.abs(point_3d) < 1000):  # 排除异常大的值
                            trajectory_3d.append(point_3d)
                            trajectory_timestamps.append(ts1)

        # 按时间戳排序
        if trajectory_3d:
            sorted_pairs = sorted(zip(trajectory_timestamps, trajectory_3d), key=lambda x: x[0])
            trajectory_timestamps, trajectory_3d = zip(*sorted_pairs)
            trajectory_timestamps = list(trajectory_timestamps)
            trajectory_3d = list(trajectory_3d)

        return trajectory_3d, trajectory_timestamps

    def is_point_above_court(self, point_3d, z_threshold=0.0, court_boundary=None):
        """
        判断3D点是否在球场上方

        参数:
            point_3d: 3D点坐标 (x, y, z)
            z_threshold: z坐标阈值，高于此值视为在场地上方
            court_boundary: 场地边界 [x_min, y_min, x_max, y_max]，如果为None则使用默认值

        返回:
            is_above: 是否在场地上方
        """
        if point_3d is None:
            return False

        x, y, z = point_3d

        # 检查z坐标
        if z < z_threshold:
            return False

        # 检查xy坐标是否在场地范围内
        if court_boundary is None:
            # 默认球场边界 (厘米)
            court_boundary = [0, 0, 610, 1340]

        x_min, y_min, x_max, y_max = court_boundary

        return x_min <= x <= x_max and y_min <= y <= y_max

    def get_recent_3d_points(self, time_window=0.5):
        """获取最近时间窗口内的3D点"""
        if not self.timestamps:
            return [], []

        current_time = self.timestamps[-1]
        recent_indices = [i for i, t in enumerate(self.timestamps)
                          if current_time - t <= time_window]

        recent_points = [self.points_3d[i] for i in recent_indices]
        recent_timestamps = [self.timestamps[i] for i in recent_indices]

        return recent_points, recent_timestamps

    def filter_trajectory_by_court(self, trajectory1, timestamps1, trajectory2, timestamps2):
        """
        筛选出在目标场地上方的轨迹点

        参数:
            trajectory1: 相机1中的轨迹点列表
            timestamps1: 相机1中的时间戳列表
            trajectory2: 相机2中的轨迹点列表
            timestamps2: 相机2中的时间戳列表

        返回:
            filtered_traj1: 相机1中筛选后的轨迹点
            filtered_ts1: 相机1中筛选后的时间戳
            filtered_traj2: 相机2中筛选后的轨迹点
            filtered_ts2: 相机2中筛选后的时间戳
        """
        # 检查轨迹长度
        if len(trajectory1) < 3 or len(trajectory2) < 3:
            return trajectory1, timestamps1, trajectory2, timestamps2

        # 为每个点对计算3D坐标
        point_pairs = []  # [(idx1, idx2, point_3d), ...]

        for i, ts1 in enumerate(timestamps1):
            # 查找最近的时间戳
            closest_ts2_idx = None
            min_diff = 0.02  # 20ms阈值

            for j, ts2 in enumerate(timestamps2):
                diff = abs(ts1 - ts2)
                if diff < min_diff:
                    min_diff = diff
                    closest_ts2_idx = j

            if closest_ts2_idx is not None:
                # 确保索引在有效范围内
                if i < len(trajectory1) and closest_ts2_idx < len(trajectory2):
                    # 三角测量
                    point_3d = self.triangulate_point(trajectory1[i], trajectory2[closest_ts2_idx])

                    if point_3d is not None:
                        point_pairs.append((i, closest_ts2_idx, point_3d))

        # 筛选出在场地上方的点对
        valid_pairs = []

        for idx1, idx2, point_3d in point_pairs:
            if self.is_point_above_court(point_3d):
                valid_pairs.append((idx1, idx2))

        # 提取筛选后的轨迹点
        if valid_pairs:
            valid_idx1 = [pair[0] for pair in valid_pairs]
            valid_idx2 = [pair[1] for pair in valid_pairs]

            filtered_traj1 = [trajectory1[i] for i in valid_idx1]
            filtered_ts1 = [timestamps1[i] for i in valid_idx1]
            filtered_traj2 = [trajectory2[i] for i in valid_idx2]
            filtered_ts2 = [timestamps2[i] for i in valid_idx2]

            return filtered_traj1, filtered_ts1, filtered_traj2, filtered_ts2

        # 如果没有有效点对，返回原始轨迹
        return trajectory1, timestamps1, trajectory2, timestamps2

    def reset(self):
        """重置处理器状态"""
        self.points_3d = []
        self.timestamps = []