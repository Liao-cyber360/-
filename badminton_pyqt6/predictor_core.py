"""
预测算法核心模块
从原有predictor.py迁移算法逻辑，去除UI依赖
"""
import numpy as np
import cv2
from filterpy.kalman import ExtendedKalmanFilter
import time
from scipy.optimize import minimize
try:
    from .config import config
except ImportError:
    from config import config


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
        
        # 记录状态
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
            position[0], position[1], position[2],
            velocity[0], velocity[1], velocity[2]
        ])
        
        # 设置状态转移协方差 (过程噪声)
        process_noise = config.ekf_process_noise
        self.ekf.Q = np.eye(6) * process_noise
        
        # 设置观测协方差 (测量噪声)
        measurement_noise = config.ekf_measurement_noise
        self.ekf.R = np.eye(3) * measurement_noise
        
        # 设置初始协方差
        self.ekf.P = np.eye(6) * 1.0
        
        self.last_time = timestamp
        self.initialized = True
    
    def state_transition_function(self, x, dt):
        """
        状态转移函数（包含空气动力学模型）
        
        参数:
            x: 当前状态 [x, y, z, vx, vy, vz]
            dt: 时间间隔
        
        返回:
            next_state: 下一状态
        """
        pos = x[:3]
        vel = x[3:]
        
        # 速度大小
        speed = np.linalg.norm(vel)
        
        if speed < 1e-6:
            # 速度太小，简单线性运动
            new_pos = pos + vel * dt
            new_vel = vel - np.array([0, 0, self.params['gravity']]) * dt
        else:
            # 空气阻力
            drag_acc = -(self.params['drag_coefficient'] * self.params['air_density'] * 
                        self.params['cross_section'] * speed * vel) / (2 * self.params['mass'])
            
            # 重力
            gravity_acc = np.array([0, 0, -self.params['gravity']])
            
            # 总加速度
            total_acc = drag_acc + gravity_acc
            
            # 更新位置和速度
            new_pos = pos + vel * dt + 0.5 * total_acc * dt**2
            new_vel = vel + total_acc * dt
        
        return np.concatenate([new_pos, new_vel])
    
    def state_transition_jacobian(self, x, dt):
        """计算状态转移函数的雅可比矩阵"""
        jacobian = np.eye(6)
        
        # 位置对位置的偏导数
        jacobian[:3, :3] = np.eye(3)
        
        # 位置对速度的偏导数
        jacobian[:3, 3:] = np.eye(3) * dt
        
        # 速度对速度的偏导数（包含阻力项）
        vel = x[3:]
        speed = np.linalg.norm(vel)
        
        if speed > 1e-6:
            # 阻力项的雅可比
            drag_factor = (self.params['drag_coefficient'] * self.params['air_density'] * 
                          self.params['cross_section']) / (2 * self.params['mass'])
            
            for i in range(3):
                for j in range(3):
                    if i == j:
                        jacobian[3+i, 3+j] = 1 - drag_factor * dt * (2 * speed + vel[i]**2 / speed)
                    else:
                        jacobian[3+i, 3+j] = -drag_factor * dt * (vel[i] * vel[j] / speed)
        else:
            jacobian[3:, 3:] = np.eye(3)
        
        return jacobian
    
    def observation_function(self, x):
        """观测函数（直接观测位置）"""
        return x[:3]
    
    def observation_jacobian(self, x):
        """观测函数的雅可比矩阵"""
        jacobian = np.zeros((3, 6))
        jacobian[:3, :3] = np.eye(3)
        return jacobian
    
    def predict(self, dt):
        """预测步骤"""
        if not self.initialized:
            return None
        
        # 预测函数
        def f(x):
            return self.state_transition_function(x, dt)
        
        # 雅可比函数
        def F(x):
            return self.state_transition_jacobian(x, dt)
        
        # 执行预测
        self.ekf.predict(f, F)
        
        return self.ekf.x.copy()
    
    def update(self, measurement):
        """更新步骤"""
        if not self.initialized:
            return None
        
        # 观测函数
        def h(x):
            return self.observation_function(x)
        
        # 观测雅可比
        def H(x):
            return self.observation_jacobian(x)
        
        # 执行更新
        self.ekf.update(measurement, h, H)
        
        return self.ekf.x.copy()
    
    def process_measurement(self, position, timestamp):
        """
        处理新的测量数据
        
        参数:
            position: 观测位置 (x, y, z)
            timestamp: 时间戳
        
        返回:
            filtered_state: 滤波后的状态
        """
        if not self.initialized:
            return None
        
        dt = timestamp - self.last_time
        if dt <= 0:
            return self.ekf.x.copy()
        
        # 预测
        self.predict(dt)
        
        # 更新
        self.update(np.array(position))
        
        self.last_time = timestamp
        
        return self.ekf.x.copy()


class TrajectoryPredictor:
    """轨迹预测器核心"""
    
    def __init__(self):
        """初始化预测器"""
        self.ekf_predictor = ExtendedKalmanFilterPredictor()
        self.polynomial_predictor = PolynomialPredictor()
        self.physics_predictor = PhysicsModelPredictor()
        
        # 预测结果缓存
        self.last_prediction = None
        self.prediction_confidence = 0.0
    
    def predict_landing_point(self, trajectory_3d, timestamps, method='ensemble'):
        """
        预测落点
        
        参数:
            trajectory_3d: 3D轨迹点列表
            timestamps: 时间戳列表
            method: 预测方法 ('ekf', 'polynomial', 'physics', 'ensemble')
        
        返回:
            prediction_result: 预测结果字典
        """
        if len(trajectory_3d) < 5:
            return None
        
        predictions = {}
        
        try:
            # EKF预测
            if method in ['ekf', 'ensemble']:
                ekf_result = self._predict_with_ekf(trajectory_3d, timestamps)
                if ekf_result:
                    predictions['ekf'] = ekf_result
            
            # 多项式预测
            if method in ['polynomial', 'ensemble']:
                poly_result = self._predict_with_polynomial(trajectory_3d, timestamps)
                if poly_result:
                    predictions['polynomial'] = poly_result
            
            # 物理模型预测
            if method in ['physics', 'ensemble']:
                physics_result = self._predict_with_physics(trajectory_3d, timestamps)
                if physics_result:
                    predictions['physics'] = physics_result
            
            # 集成预测
            if method == 'ensemble' and len(predictions) > 1:
                final_result = self._ensemble_prediction(predictions)
            elif predictions:
                # 使用单一方法的结果
                final_result = list(predictions.values())[0]
            else:
                return None
            
            self.last_prediction = final_result
            return final_result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def _predict_with_ekf(self, trajectory_3d, timestamps):
        """使用EKF进行预测"""
        if len(trajectory_3d) < 3:
            return None
        
        # 初始化EKF
        initial_pos = np.array(trajectory_3d[0])
        
        # 估计初始速度
        if len(trajectory_3d) >= 2:
            dt = timestamps[1] - timestamps[0]
            if dt > 0:
                initial_vel = (np.array(trajectory_3d[1]) - np.array(trajectory_3d[0])) / dt
            else:
                initial_vel = np.zeros(3)
        else:
            initial_vel = np.zeros(3)
        
        self.ekf_predictor.initialize(initial_pos, initial_vel, timestamps[0])
        
        # 处理轨迹数据
        filtered_states = []
        for i, (pos, ts) in enumerate(zip(trajectory_3d, timestamps)):
            if i > 0:
                state = self.ekf_predictor.process_measurement(pos, ts)
                if state is not None:
                    filtered_states.append(state)
        
        if not filtered_states:
            return None
        
        # 预测落点
        last_state = filtered_states[-1]
        last_time = timestamps[-1]
        
        # 模拟轨迹直到落地
        current_state = last_state.copy()
        dt = 0.01  # 10ms步长
        
        trajectory_points = []
        max_time = 5.0  # 最大预测时间5秒
        
        for step in range(int(max_time / dt)):
            current_state = self.ekf_predictor.state_transition_function(current_state, dt)
            trajectory_points.append(current_state[:3].copy())
            
            # 检查是否落地
            if current_state[2] <= 0:  # Z坐标小于等于0
                landing_point = current_state[:3].copy()
                landing_point[2] = 0  # 确保Z=0
                
                return {
                    'landing_point': landing_point,
                    'trajectory': trajectory_points,
                    'method': 'ekf',
                    'confidence': 0.8
                }
        
        return None
    
    def _predict_with_polynomial(self, trajectory_3d, timestamps):
        """使用多项式拟合进行预测"""
        return self.polynomial_predictor.predict(trajectory_3d, timestamps)
    
    def _predict_with_physics(self, trajectory_3d, timestamps):
        """使用物理模型进行预测"""
        return self.physics_predictor.predict(trajectory_3d, timestamps)
    
    def _ensemble_prediction(self, predictions):
        """集成多种预测方法的结果"""
        if not predictions:
            return None
        
        # 权重设置
        weights = {
            'ekf': 0.4,
            'polynomial': 0.3,
            'physics': 0.3
        }
        
        # 加权平均落点
        total_weight = 0
        weighted_landing = np.zeros(3)
        
        for method, result in predictions.items():
            if method in weights and result['landing_point'] is not None:
                weight = weights[method] * result['confidence']
                weighted_landing += weight * np.array(result['landing_point'])
                total_weight += weight
        
        if total_weight > 0:
            final_landing = weighted_landing / total_weight
            
            # 计算集成置信度
            confidence = min(1.0, total_weight)
            
            return {
                'landing_point': final_landing,
                'trajectory': predictions['ekf']['trajectory'] if 'ekf' in predictions else [],
                'method': 'ensemble',
                'confidence': confidence,
                'sub_predictions': predictions
            }
        
        return None


class PolynomialPredictor:
    """多项式拟合预测器"""
    
    def predict(self, trajectory_3d, timestamps):
        """使用多项式拟合预测落点"""
        if len(trajectory_3d) < 5:
            return None
        
        try:
            # 准备数据
            times = np.array(timestamps) - timestamps[0]  # 相对时间
            positions = np.array(trajectory_3d)
            
            # 分别拟合X, Y, Z轴
            degree = min(config.poly_fit_degree, len(trajectory_3d) - 1)
            
            poly_x = np.polyfit(times, positions[:, 0], degree)
            poly_y = np.polyfit(times, positions[:, 1], degree)
            poly_z = np.polyfit(times, positions[:, 2], degree)
            
            # 预测落地时间（Z=0）
            poly_z_func = np.poly1d(poly_z)
            
            # 寻找Z=0的时间
            roots = np.roots(poly_z)
            valid_roots = [r for r in roots if np.isreal(r) and r.real > times[-1]]
            
            if not valid_roots:
                return None
            
            landing_time = min(valid_roots).real
            
            # 计算落点
            landing_x = np.polyval(poly_x, landing_time)
            landing_y = np.polyval(poly_y, landing_time)
            
            # 生成预测轨迹
            future_times = np.linspace(times[-1], landing_time, 100)
            trajectory_points = []
            
            for t in future_times:
                x = np.polyval(poly_x, t)
                y = np.polyval(poly_y, t)
                z = max(0, np.polyval(poly_z, t))
                trajectory_points.append([x, y, z])
            
            return {
                'landing_point': [landing_x, landing_y, 0],
                'trajectory': trajectory_points,
                'method': 'polynomial',
                'confidence': 0.6
            }
            
        except Exception as e:
            print(f"Polynomial prediction error: {e}")
            return None


class PhysicsModelPredictor:
    """基于物理模型的预测器"""
    
    def __init__(self):
        self.params = config.get_aero_params()
    
    def predict(self, trajectory_3d, timestamps):
        """使用物理模型预测落点"""
        if len(trajectory_3d) < 3:
            return None
        
        try:
            # 估计初始状态
            positions = np.array(trajectory_3d)
            times = np.array(timestamps)
            
            # 使用最后几个点估计初始速度
            if len(positions) >= 2:
                dt = times[-1] - times[-2]
                if dt > 0:
                    velocity = (positions[-1] - positions[-2]) / dt
                else:
                    return None
            else:
                return None
            
            initial_state = np.concatenate([positions[-1], velocity])
            
            # 参数优化（拟合物理模型到观测数据）
            optimized_params = self._optimize_parameters(positions, times)
            
            # 使用优化后的参数预测轨迹
            trajectory_points = self._simulate_trajectory(
                initial_state, optimized_params
            )
            
            if not trajectory_points:
                return None
            
            # 找到落地点
            landing_point = None
            for point in trajectory_points:
                if point[2] <= 0:
                    landing_point = [point[0], point[1], 0]
                    break
            
            if landing_point is None:
                return None
            
            return {
                'landing_point': landing_point,
                'trajectory': trajectory_points,
                'method': 'physics',
                'confidence': 0.7
            }
            
        except Exception as e:
            print(f"Physics prediction error: {e}")
            return None
    
    def _optimize_parameters(self, positions, times):
        """优化物理参数以拟合观测数据"""
        # 简化：直接使用配置参数
        return self.params
    
    def _simulate_trajectory(self, initial_state, params):
        """模拟物理轨迹"""
        dt = 0.01  # 10ms步长
        max_time = 5.0  # 最大模拟时间
        
        trajectory_points = []
        current_state = initial_state.copy()
        
        for step in range(int(max_time / dt)):
            pos = current_state[:3]
            vel = current_state[3:]
            
            # 检查落地
            if pos[2] <= 0:
                break
            
            trajectory_points.append(pos.copy())
            
            # 物理更新
            speed = np.linalg.norm(vel)
            
            if speed > 1e-6:
                # 阻力加速度
                drag_acc = -(params['drag_coefficient'] * params['air_density'] * 
                           params['cross_section'] * speed * vel) / (2 * params['mass'])
            else:
                drag_acc = np.zeros(3)
            
            # 重力加速度
            gravity_acc = np.array([0, 0, -params['gravity']])
            
            # 总加速度
            total_acc = drag_acc + gravity_acc
            
            # 更新状态
            new_pos = pos + vel * dt + 0.5 * total_acc * dt**2
            new_vel = vel + total_acc * dt
            
            current_state = np.concatenate([new_pos, new_vel])
        
        return trajectory_points


class CourtBoundaryAnalyzer:
    """场地边界分析器"""
    
    def __init__(self):
        """初始化边界分析器"""
        # 场地尺寸 (单位: cm)
        self.court_width = 610  # 场地宽度
        self.court_length = 670  # 半场长度
        
        # 边界定义
        self.boundaries = {
            'singles': {
                'x_min': 4, 'x_max': 606,
                'y_min': 4, 'y_max': 666
            },
            'doubles': {
                'x_min': 0, 'x_max': 610,
                'y_min': 0, 'y_max': 670
            }
        }
    
    def analyze_landing(self, landing_point, court_type='singles'):
        """
        分析落点是否在场地内
        
        参数:
            landing_point: 落点坐标 [x, y, z]
            court_type: 场地类型 ('singles' 或 'doubles')
        
        返回:
            analysis_result: 分析结果字典
        """
        if landing_point is None:
            return None
        
        x, y = landing_point[:2]
        boundary = self.boundaries.get(court_type, self.boundaries['singles'])
        
        # 边界检查
        in_bounds_x = boundary['x_min'] <= x <= boundary['x_max']
        in_bounds_y = boundary['y_min'] <= y <= boundary['y_max']
        in_bounds = in_bounds_x and in_bounds_y
        
        # 计算到边界的距离
        distances = {
            'left': x - boundary['x_min'],
            'right': boundary['x_max'] - x,
            'bottom': y - boundary['y_min'],
            'top': boundary['y_max'] - y
        }
        
        # 找到最近的边界
        closest_boundary = min(distances.items(), key=lambda x: abs(x[1]))
        
        return {
            'in_bounds': in_bounds,
            'court_type': court_type,
            'landing_point': landing_point,
            'distances': distances,
            'closest_boundary': closest_boundary[0],
            'closest_distance': closest_boundary[1]
        }
    
    def get_court_regions(self):
        """获取场地区域定义"""
        return {
            'service_areas': {
                'left_service': {
                    'x_min': 76, 'x_max': 305,
                    'y_min': 198, 'y_max': 472
                },
                'right_service': {
                    'x_min': 305, 'x_max': 534,
                    'y_min': 198, 'y_max': 472
                }
            },
            'back_court': {
                'x_min': 4, 'x_max': 606,
                'y_min': 4, 'y_max': 198
            },
            'front_court': {
                'x_min': 4, 'x_max': 606,
                'y_min': 472, 'y_max': 666
            }
        }