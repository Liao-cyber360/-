import numpy as np
import cv2
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
import time
from utils import config


class TrajectoryPredictor:
    """羽毛球轨迹预测器 - 兼容性修复版"""

    def __init__(self):
        self.gravity = 9.8 * 100  # cm/s²
        self.air_resistance_coeff = 0.1  # 空气阻力系数
        self.prediction_time_horizon = 3.0  # 预测时间范围(秒)

        # 预测结果
        self.last_prediction = None
        self.last_prediction_time = 0

        print("TrajectoryPredictor initialized with compatibility fixes")

    def predict_landing_point(self, trajectory_points, timestamps):
        """预测羽毛球落地点"""
        if len(trajectory_points) < 5:
            print(f"Insufficient points for prediction: {len(trajectory_points)}")
            return None, None, None

        try:
            # 转换为numpy数组
            points = np.array(trajectory_points)
            times = np.array(timestamps)

            print(f"Original trajectory: {len(points)} points")
            print(f"Time range: {times[0]:.3f} to {times[-1]:.3f}")

            # 修复时间序列问题
            cleaned_points, cleaned_times = self._clean_trajectory_data(points, times)

            if cleaned_points is None or len(cleaned_points) < 3:
                print("Failed to clean trajectory data")
                return None, None, None

            print(f"Cleaned trajectory: {len(cleaned_points)} points")

            # 检查轨迹有效性
            if not self._validate_trajectory(cleaned_points, cleaned_times):
                print("Trajectory validation failed")
                return None, None, None

            # 直接使用物理模型（跳过有问题的EKF）
            print("Using physics model for prediction")
            landing_position, landing_time, predicted_trajectory = self._predict_with_physics_model(cleaned_points,
                                                                                                    cleaned_times)

            if landing_position is not None:
                self.last_prediction = {
                    'position': landing_position,
                    'time': landing_time,
                    'trajectory': predicted_trajectory
                }
                self.last_prediction_time = time.time()

                print(f"Prediction successful: landing at ({landing_position[0]:.1f}, {landing_position[1]:.1f})")
                return landing_position, landing_time, predicted_trajectory
            else:
                print("Physics model prediction failed")
                return None, None, None

        except Exception as e:
            print(f"Error in trajectory prediction: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def _clean_trajectory_data(self, points, times):
        """清理轨迹数据，解决时间序列问题"""
        if len(points) == 0:
            return None, None

        try:
            # 1. 创建数据索引
            data_indices = list(range(len(points)))

            # 2. 按时间戳排序
            sorted_indices = sorted(data_indices, key=lambda i: times[i])
            sorted_points = points[sorted_indices]
            sorted_times = times[sorted_indices]

            # 3. 移除重复时间戳
            unique_indices = []
            last_time = None
            tolerance = 1e-6  # 时间容差

            for i, t in enumerate(sorted_times):
                if last_time is None or abs(t - last_time) > tolerance:
                    unique_indices.append(i)
                    last_time = t
                else:
                    print(f"Removing duplicate timestamp: {t}")

            if len(unique_indices) < 3:
                print(f"Too few unique timestamps: {len(unique_indices)}")
                return None, None

            cleaned_points = sorted_points[unique_indices]
            cleaned_times = sorted_times[unique_indices]

            # 4. 检查时间间隔的合理性
            time_diffs = np.diff(cleaned_times)

            # 移除时间间隔过大的点（可能是断续）
            valid_indices = [0]  # 保留第一个点
            for i in range(1, len(cleaned_times)):
                dt = time_diffs[i - 1]
                if dt <= 0.2:  # 最大允许0.2秒间隔
                    valid_indices.append(i)
                else:
                    print(f"Removing point with large time gap: {dt:.3f}s")

            if len(valid_indices) < 3:
                print(f"Too few points after time gap filtering: {len(valid_indices)}")
                return None, None

            final_points = cleaned_points[valid_indices]
            final_times = cleaned_times[valid_indices]

            # 5. 如果时间间隔仍然不均匀，重新采样
            if len(final_points) >= 5:
                resampled_points, resampled_times = self._resample_trajectory(final_points, final_times)
                if resampled_points is not None:
                    final_points = resampled_points
                    final_times = resampled_times

            print(
                f"Data cleaning complete: {len(final_points)} points, time span: {final_times[-1] - final_times[0]:.3f}s")

            return final_points, final_times

        except Exception as e:
            print(f"Error in data cleaning: {e}")
            return None, None

    def _resample_trajectory(self, points, times):
        """重新采样轨迹到均匀时间间隔"""
        try:
            if len(points) < 3:
                return points, times

            # 计算总时间跨度
            time_span = times[-1] - times[0]
            target_fps = 30.0
            target_dt = 1.0 / target_fps

            # 如果时间跨度太短，保持原有数据
            if time_span < 0.1:
                return points, times

            # 计算目标采样点数
            num_samples = max(5, int(time_span / target_dt))
            target_times = np.linspace(times[0], times[-1], num_samples)

            # 对每个坐标进行插值
            resampled_points = []

            for dim in range(3):  # x, y, z
                coords = points[:, dim]
                # 使用线性插值
                interp_coords = np.interp(target_times, times, coords)
                resampled_points.append(interp_coords)

            resampled_points = np.array(resampled_points).T

            print(f"Resampled trajectory: {len(points)} -> {len(resampled_points)} points")

            return resampled_points, target_times

        except Exception as e:
            print(f"Error in trajectory resampling: {e}")
            return points, times

    def _validate_trajectory(self, points, times):
        """验证轨迹有效性"""
        if len(points) < 3:
            print("Too few points for validation")
            return False

        try:
            # 1. 检查时间序列
            time_diffs = np.diff(times)
            tolerance = 1e-3  # 1毫秒容差
            invalid_times = time_diffs < -tolerance

            if np.any(invalid_times):
                invalid_count = np.sum(invalid_times)
                if invalid_count <= len(time_diffs) // 3:  # 允许1/3的时间点有问题
                    print("Warning: Some time inconsistencies found, but proceeding")
                else:
                    print("Too many time sequence errors, rejecting trajectory")
                    return False

            # 2. 检查下降趋势
            z_coords = points[:, 2]
            n = len(z_coords)
            third = max(1, n // 3)
            first_third_avg = np.mean(z_coords[:third])
            last_third_avg = np.mean(z_coords[-third:])
            height_drop = first_third_avg - last_third_avg

            if height_drop < 5:  # 需要至少5cm的下降
                print(f"Insufficient height drop: {height_drop:.2f}cm")
                return False

            # 3. 检查合理的运动速度
            velocities = []
            for i in range(1, len(points)):
                dt = time_diffs[i - 1]
                if dt > 0:
                    distance = np.linalg.norm(points[i] - points[i - 1])
                    vel = distance / dt
                    velocities.append(vel)

            if velocities:
                max_vel = max(velocities)
                avg_vel = np.mean(velocities)

                if max_vel > 8000:  # 80m/s 绝对上限
                    print(f"Unrealistic maximum velocity: {max_vel:.1f} cm/s")
                    return False

                if avg_vel < 5:  # 平均速度太低
                    print(f"Average velocity too low: {avg_vel:.1f} cm/s")
                    return False

            # 4. 检查空间合理性
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            horizontal_movement = max(x_range, y_range)

            if horizontal_movement < 3:  # 水平移动少于3cm
                print(f"Insufficient horizontal movement: {horizontal_movement:.1f}cm")
                return False

            # 5. 检查时间合理性
            total_time = times[-1] - times[0]
            if total_time < 0.05:  # 少于50毫秒
                print(f"Trajectory time span too short: {total_time:.3f}s")
                return False

            print("✅ Trajectory validation PASSED")
            return True

        except Exception as e:
            print(f"❌ Error in trajectory validation: {e}")
            return False

    def _predict_with_physics_model(self, points, times):
        """使用物理模型的预测方法"""
        try:
            print("Using physics model for prediction")

            # 使用最后几个点进行抛物线拟合
            if len(points) >= 3:
                recent_points = points[-min(8, len(points)):]
                recent_times = times[-len(recent_points):]
            else:
                recent_points = points
                recent_times = times

            print(f"Physics model using {len(recent_points)} points")

            # 分别拟合x, y, z坐标
            time_relative = recent_times - recent_times[0]

            try:
                # Z方向二次拟合（抛物线）
                if len(recent_points) >= 3:
                    z_coeffs = np.polyfit(time_relative, recent_points[:, 2], 2)
                    print(f"Z coefficients: {z_coeffs}")
                else:
                    # 线性拟合
                    z_coeffs = np.polyfit(time_relative, recent_points[:, 2], 1)
                    z_coeffs = np.array([0, z_coeffs[0], z_coeffs[1]])

                # X, Y方向线性拟合
                x_coeffs = np.polyfit(time_relative, recent_points[:, 0], 1)
                y_coeffs = np.polyfit(time_relative, recent_points[:, 1], 1)

                print(f"X coefficients: {x_coeffs}")
                print(f"Y coefficients: {y_coeffs}")

                # 求解Z=0的时间
                a, b, c = z_coeffs
                print(f"Solving Z=0: {a:.3f}*t² + {b:.3f}*t + {c:.3f} = 0")

                landing_time_rel = None

                if abs(a) > 1e-6:  # 二次方程
                    discriminant = b ** 2 - 4 * a * c
                    print(f"Discriminant: {discriminant:.6f}")

                    if discriminant >= 0:
                        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
                        t2 = (-b - np.sqrt(discriminant)) / (2 * a)

                        print(f"Quadratic solutions: t1={t1:.3f}, t2={t2:.3f}")

                        # 选择未来的时间点
                        future_times = [t for t in [t1, t2] if t > time_relative[-1]]

                        if future_times:
                            landing_time_rel = min(future_times)
                            print(f"Selected landing time: {landing_time_rel:.3f}s")

                elif abs(b) > 1e-6:  # 线性方程 (a≈0)
                    landing_time_rel = -c / b
                    print(f"Linear solution: t={landing_time_rel:.3f}s")

                    if landing_time_rel <= time_relative[-1]:
                        landing_time_rel = None

                if landing_time_rel is not None and landing_time_rel > time_relative[-1]:
                    landing_time_abs = recent_times[0] + landing_time_rel

                    # 计算落地位置
                    landing_x = np.polyval(x_coeffs, landing_time_rel)
                    landing_y = np.polyval(y_coeffs, landing_time_rel)

                    landing_position = np.array([landing_x, landing_y])

                    print(f"Predicted landing: ({landing_x:.1f}, {landing_y:.1f}) at t={landing_time_abs:.3f}")

                    # 生成预测轨迹
                    predicted_trajectory = []
                    trajectory_dt = 0.05  # 50ms间隔
                    t_current = time_relative[-1]

                    while t_current <= landing_time_rel:
                        pred_x = np.polyval(x_coeffs, t_current)
                        pred_y = np.polyval(y_coeffs, t_current)
                        pred_z = max(0, np.polyval(z_coeffs, t_current))

                        predicted_trajectory.append({
                            'position': np.array([pred_x, pred_y, pred_z]),
                            'velocity': np.array([0, 0, 0]),  # 简化
                            'time': recent_times[0] + t_current
                        })

                        t_current += trajectory_dt

                    return landing_position, landing_time_abs, predicted_trajectory

            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Polynomial fitting failed: {e}")

            # 简单线性外推作为最后备选
            if len(points) >= 2:
                print("Using simple linear extrapolation")

                time_diff = times[-1] - times[-2]
                if time_diff <= 0:
                    time_diff = 0.033  # 默认30fps

                last_velocity = (points[-1] - points[-2]) / time_diff

                if last_velocity[2] < -10:  # 向下运动至少10cm/s
                    fall_time = -points[-1][2] / last_velocity[2]

                    if fall_time > 0 and fall_time < 5:  # 合理的落地时间
                        landing_position = points[-1][:2] + last_velocity[:2] * fall_time
                        landing_time = times[-1] + fall_time

                        print(
                            f"Linear extrapolation: landing at ({landing_position[0]:.1f}, {landing_position[1]:.1f}) in {fall_time:.2f}s")

                        return landing_position, landing_time, []

            print("All physics model methods failed")
            return None, None, None

        except Exception as e:
            print(f"Physics model prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None


class CourtBoundaryAnalyzer:
    """羽毛球场地边界分析器"""

    def __init__(self):
        # 标准羽毛球场地尺寸 (cm)
        self.court_boundaries = {
            'singles': {
                'length': 1340,  # 13.4m
                'width': 518,  # 5.18m
                'service_length': 390,  # 3.9m (from net to service line)
                'net_height': 152.4,  # 1.524m at center
            },
            'doubles': {
                'length': 1340,  # 13.4m
                'width': 610,  # 6.1m
                'service_length': 390,  # 3.9m
                'net_height': 152.4,  # 1.524m at center
            }
        }

        # 场地中心作为原点 (0, 0)
        self.court_center = np.array([0, 0])

    def is_point_in_court(self, point, game_type='singles'):
        """判断点是否在场地范围内"""
        if len(point) < 2:
            return False

        x, y = point[0], point[1]

        boundaries = self.court_boundaries[game_type]
        half_length = boundaries['length'] / 2-100
        half_width = boundaries['width'] / 2-100

        # 检查是否在场地边界内
        in_bounds = (abs(x) <= half_length and abs(y) <= half_width)

        return in_bounds

    def get_landing_zone_info(self, point, game_type='singles'):
        """获取落地点的区域信息"""
        if not self.is_point_in_court(point, game_type):
            return "OUT_OF_BOUNDS"

        x, y = point[0], point[1]
        boundaries = self.court_boundaries[game_type]

        # 判断前场后场
        if abs(x) <= boundaries['service_length']:
            zone = "FRONT_COURT"
        else:
            zone = "BACK_COURT"

        # 判断左右
        if y > 0:
            side = "RIGHT"
        elif y < 0:
            side = "LEFT"
        else:
            side = "CENTER"

        return f"{zone}_{side}"

    def calculate_distance_to_boundary(self, point, game_type='singles'):
        """计算点到场地边界的最短距离"""
        x, y = point[0], point[1]

        boundaries = self.court_boundaries[game_type]
        half_length = boundaries['length'] / 2
        half_width = boundaries['width'] / 2

        # 计算到各边界的距离
        dist_to_length_boundary = half_length - abs(x)
        dist_to_width_boundary = half_width - abs(y)

        # 返回最短距离
        return min(dist_to_length_boundary, dist_to_width_boundary)