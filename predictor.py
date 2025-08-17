import numpy as np
import cv2
from filterpy.kalman import ExtendedKalmanFilter
import time
from utils import config


class ExtendedKalmanFilterPredictor:
    """基于空气动力学模型的扩展卡尔曼滤波器实现"""

    def __init__(self):
        """初始化EKF"""
        # 状态维度 [x, y, z, vx, vy, vz]
        self.dim_x = 6

        # 测量维度 [x, y, z]
        self.dim_z = 3

        # 创建EKF
        self.ekf = ExtendedKalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)

        # 空气动力学参数
        self.params = config.get_aero_params()

        # 记录上一次时间
        self.last_time = None
        self.initialized = False

    def initialize(self, position, velocity, timestamp):
        """
        初始化滤波器状态

        参数:
            position: 初始位置 (x, y, z)
            velocity: 初始速度 (vx, vy, vz)
            timestamp: 初始时间戳
        """
        # 设置初始状态
        self.ekf.x = np.array([
            position[0],  # x
            position[1],  # y
            position[2],  # z
            velocity[0],  # vx
            velocity[1],  # vy
            velocity[2]  # vz
        ])

        # 设置状态转移协方差 (过程噪声)
        process_noise = config.ekf_process_noise
        self.ekf.Q = np.eye(6) * process_noise
        self.ekf.Q[0:3, 0:3] *= 0.1  # 位置状态噪声较小
        self.ekf.Q[3:6, 3:6] *= 10.0  # 速度状态噪声较大

        # 设置测量噪声
        measurement_noise = config.ekf_measurement_noise
        self.ekf.R = np.eye(3) * measurement_noise

        # 状态不确定性初始协方差
        self.ekf.P = np.eye(6) * 10.0

        # 记录时间戳
        self.last_time = timestamp
        self.initialized = True

    def state_transition(self, x, dt):
        """
        空气动力学模型状态转移函数

        参数:
            x: 当前状态 [x, y, z, vx, vy, vz]
            dt: 时间步长

        返回:
            新状态
        """
        # 提取位置和速度
        pos = x[0:3]
        vel = x[3:6]

        # 计算速度大小
        v_mag = np.linalg.norm(vel)

        # 空气动力学参数
        mass = self.params['mass']
        g = self.params['gravity']
        aero_length = self.params['aero_length']

        # 加速度计算: a = g - v||v||/L
        if v_mag > 1e-6:  # 避免除以零
            drag_dir = vel / v_mag  # 阻力方向
            drag_acc = v_mag ** 2 / aero_length * drag_dir  # 阻力加速度
        else:
            drag_acc = np.zeros(3)

        # 重力加速度 (只在z方向上)
        gravity_acc = np.array([0, 0, -g])

        # 总加速度
        acc = gravity_acc - drag_acc

        # 更新位置和速度
        new_pos = pos + vel * dt + 0.5 * acc * dt ** 2
        new_vel = vel + acc * dt

        # 组合新状态
        new_state = np.concatenate([new_pos, new_vel])

        return new_state

    def predict_landing(self, dt=0.01, max_steps=1000):
        """
        预测落地点

        参数:
            dt: 时间步长
            max_steps: 最大预测步数

        返回:
            landing_position: 落地点位置
            landing_time: 落地时间
            trajectory: 预测轨迹
        """
        if not self.initialized:
            return None, None, []

        # 获取当前状态
        current_state = self.ekf.x.copy()

        # 预测轨迹
        trajectory = [current_state[0:3].copy()]
        state = current_state.copy()
        t = 0

        for i in range(max_steps):
            # 计算下一个状态
            state = self.state_transition(state, dt)
            trajectory.append(state[0:3].copy())
            t += dt

            # 检查是否落地 (z <= 0)
            if state[2] <= 0:
                # 线性插值找到精确的落地点
                prev_pos = trajectory[-2]
                curr_pos = trajectory[-1]

                if prev_pos[2] > 0:  # 确保前一点在空中
                    ratio = prev_pos[2] / (prev_pos[2] - curr_pos[2])
                    landing_x = prev_pos[0] + ratio * (curr_pos[0] - prev_pos[0])
                    landing_y = prev_pos[1] + ratio * (curr_pos[1] - prev_pos[1])

                    landing_position = np.array([landing_x, landing_y, 0.0])
                    landing_time = self.last_time + t - dt * ratio

                    return landing_position, landing_time, trajectory

        # 未找到落地点
        return None, None, trajectory


class TrajectoryPredictor:
    """羽毛球轨迹预测器"""

    def __init__(self):
        """初始化预测器"""
        # 配置参数
        self.poly_fit_degree = config.poly_fit_degree  # 多项式拟合次数
        self.ekf = ExtendedKalmanFilterPredictor()

    def identify_natural_falling_segment(self, trajectory_3d, timestamps):
        """
        识别自然下落段轨迹

        参数:
            trajectory_3d: 3D轨迹点列表
            timestamps: 时间戳列表

        返回:
            natural_segment: 自然下落段轨迹
            natural_timestamps: 对应的时间戳
        """
        if len(trajectory_3d) < 5:
            return trajectory_3d, timestamps

        # 计算相邻点之间的方向变化
        directions = []
        for i in range(1, len(trajectory_3d)):
            # 计算3D方向向量
            dx = trajectory_3d[i][0] - trajectory_3d[i - 1][0]
            dy = trajectory_3d[i][1] - trajectory_3d[i - 1][1]
            dz = trajectory_3d[i][2] - trajectory_3d[i - 1][2]

            # 避免零向量
            mag = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            if mag > 1e-6:
                directions.append(np.array([dx, dy, dz]) / mag)
            else:
                directions.append(None)

        # 找到方向变化小于40°的连续段
        segment_start = 0
        best_segment_start = 0
        best_segment_length = 0
        current_segment_length = 1

        for i in range(len(directions)):
            if directions[i] is None:
                continue

            if i > 0 and directions[i - 1] is not None:
                # 计算方向变化角度
                dot_product = np.dot(directions[i], directions[i - 1])
                # 确保点积在[-1, 1]范围内
                dot_product = max(-1.0, min(1.0, dot_product))
                # 计算角度（弧度）
                angle_rad = np.arccos(dot_product)
                # 转换为角度
                angle_deg = np.degrees(angle_rad)

                if angle_deg < 40.0:
                    # 方向变化小，属于同一段
                    current_segment_length += 1
                else:
                    # 方向变化大，开始新段
                    if current_segment_length > best_segment_length:
                        best_segment_length = current_segment_length
                        best_segment_start = segment_start

                    segment_start = i
                    current_segment_length = 1

        # 检查最后一段
        if current_segment_length > best_segment_length:
            best_segment_length = current_segment_length
            best_segment_start = segment_start

        # 如果找到的段长度不够，返回原始轨迹
        if best_segment_length < 5:
            return trajectory_3d, timestamps

        # 提取最佳段
        best_indices = range(best_segment_start, best_segment_start + best_segment_length)
        natural_segment = [trajectory_3d[i] for i in best_indices]
        natural_timestamps = [timestamps[i] for i in best_indices]

        return natural_segment, natural_timestamps

    def estimate_velocity(self, positions, timestamps):
        """
        估计速度向量

        参数:
            positions: 位置点列表
            timestamps: 时间戳列表

        返回:
            velocity: 速度向量 [vx, vy, vz]
        """
        if len(positions) < 3 or len(timestamps) < 3:
            return [0, 0, 0]

        # 相对时间
        t = np.array(timestamps) - timestamps[0]

        # 分别对x,y,z坐标进行多项式拟合
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        z_coords = [p[2] for p in positions]

        x_poly = np.polyfit(t, x_coords, 2)
        y_poly = np.polyfit(t, y_coords, 2)
        z_poly = np.polyfit(t, z_coords, 2)

        # 通过多项式的导数计算最后一个点的速度
        t_last = t[-1]
        vx = np.polyval(np.polyder(x_poly), t_last)
        vy = np.polyval(np.polyder(y_poly), t_last)
        vz = np.polyval(np.polyder(z_poly), t_last)

        return [vx, vy, vz]

    def predict_landing_point(self, trajectory_3d, timestamps):
        """
        预测羽毛球落点

        参数:
            trajectory_3d: 3D轨迹点列表
            timestamps: 时间戳列表

        返回:
            landing_position: 落地点坐标
            landing_time: 预计落地时间
            trajectory_points: 预测的轨迹点列表
        """
        # 识别自然下落段
        natural_segment, natural_timestamps = self.identify_natural_falling_segment(trajectory_3d, timestamps)

        if len(natural_segment) < 5:
            return None, None, []

        # 估计速度
        velocity = self.estimate_velocity(natural_segment, natural_timestamps)

        # 初始化EKF
        self.ekf.initialize(natural_segment[-1], velocity, natural_timestamps[-1])

        # 预测落地点
        landing_position, landing_time, predicted_trajectory = self.ekf.predict_landing()

        return landing_position, landing_time, predicted_trajectory


class CourtBoundaryAnalyzer:
    """场地边界分析器"""

    def __init__(self):
        """初始化场地边界分析器"""
        # 场地尺寸 (单位: cm)
        self.court_width = 610  # 总宽度
        self.court_height = 1340  # 总长度
        self.net_line_y = 670  # 网中心线y坐标

        # 单打边线位置
        self.singles_left = 46
        self.singles_right = 564

        # 双打边线位置
        self.doubles_left = 0
        self.doubles_right = 610

        # 前后场服务线
        self.front_service_y = 470  # 前场服务线y坐标
        self.back_service_y = 78  # 后场服务线y坐标

    def is_point_in_court(self, point, game_type='singles'):
        """
        判断点是否在场地内

        参数:
            point: (x, y, z) 坐标
            game_type: 'singles'(单打) 或 'doubles'(双打)

        返回:
            在场内返回True，否则返回False
        """
        x, y = point[0], point[1]

        # 判断长度边界
        if y < 0 or y > self.court_height:
            return False

        # 判断宽度边界
        if game_type == 'singles':
            if x < self.singles_left or x > self.singles_right:
                return False
        else:  # 双打
            if x < self.doubles_left or x > self.doubles_right:
                return False

        return True

    def get_boundary_violation(self, point, game_type='singles'):
        """
        获取边界违规信息

        参数:
            point: (x, y, z) 坐标
            game_type: 'singles'(单打) 或 'doubles'(双打)

        返回:
            violation_type: 边界违规类型
            distance: 到最近边界的距离
        """
        x, y = point[0], point[1]
        violation_type = None
        distance = 0

        # 检查长度方向
        if y < 0:
            violation_type = "Bottom Line Out"
            distance = -y
        elif y > self.court_height:
            violation_type = "Top Line Out"
            distance = y - self.court_height

        # 检查宽度方向
        if game_type == 'singles':
            if x < self.singles_left:
                if violation_type is None or x < 0:  # 考虑角落情况
                    violation_type = "Left Singles Line Out"
                    distance = self.singles_left - x
            elif x > self.singles_right:
                if violation_type is None or x > self.court_width:  # 考虑角落情况
                    violation_type = "Right Singles Line Out"
                    distance = x - self.singles_right
        else:  # 双打
            if x < self.doubles_left:
                violation_type = "Left Doubles Line Out"
                distance = self.doubles_left - x
            elif x > self.doubles_right:
                violation_type = "Right Doubles Line Out"
                distance = x - self.doubles_right

        return violation_type, distance