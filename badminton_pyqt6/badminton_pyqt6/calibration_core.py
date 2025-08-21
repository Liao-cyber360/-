"""
标定算法核心模块
从原有calibration.py迁移算法逻辑，去除UI依赖
"""
import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
try:
    from config import config
except ImportError:
    from config import config


class CalibrationCore:
    """相机标定核心算法"""
    
    def __init__(self, camera_params_file, yolo_model_path, device='cpu'):
        """
        初始化标定器
        
        参数:
            camera_params_file: 内参标定文件路径
            yolo_model_path: YOLO检测模型路径
            device: 运行设备 ('cpu' 或 'cuda')
        """
        self.camera_params_file = camera_params_file
        self.yolo_model_path = yolo_model_path
        self.device = device
        
        # 标定状态
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_width = 1280
        self.image_height = 720
        
        # 场地点数据
        self.court_3d_points = None
        self.court_point_labels = None
        self.merged_3d_points = None
        self.merged_point_labels = None
        
        # 检测结果
        self.matched_corners = {}
        self.manual_corners = []
        self.homography_matrix = None
        
        # 初始化
        self._load_camera_params()
        self._setup_court_points()
        self._load_yolo_model()
    
    def _load_camera_params(self):
        """从文件加载相机内参"""
        if not os.path.exists(self.camera_params_file):
            raise FileNotFoundError(f"Camera params file not found: {self.camera_params_file}")
        
        fs = cv2.FileStorage(self.camera_params_file, cv2.FILE_STORAGE_READ)
        self.camera_matrix = fs.getNode("camera_matrix").mat()
        self.dist_coeffs = fs.getNode("distortion_coefficients").mat().flatten()
        self.image_width = int(fs.getNode("image_width").real() or 1280)
        self.image_height = int(fs.getNode("image_height").real() or 720)
        fs.release()
    
    def _setup_court_points(self):
        """设置场地3D点坐标"""
        # 仅使用近半场点 (Y=0 到 670)
        pts = np.array([
            # 底线区域 (Y = 0 to 4)
            [0, 0, 0], [610, 0, 0],  # 外角点
            [4, 4, 0], [606, 4, 0],  # 内角点
            
            # 发球线区域 (Y = 198 to 202)
            [4, 198, 0], [606, 198, 0],  # 发球线外角点
            [76, 198, 0], [534, 198, 0],  # 发球线内角点
            [4, 202, 0], [606, 202, 0],  # 发球线外角点
            [76, 202, 0], [534, 202, 0],  # 发球线内角点
            
            # 中线区域 (Y = 333 to 337)
            [301, 333, 0], [309, 333, 0],  # 中线点
            [301, 337, 0], [309, 337, 0],  # 中线点
            
            # 网前区域 (Y = 472 to 476)
            [4, 472, 0], [606, 472, 0],  # 网前线外角点
            [76, 472, 0], [534, 472, 0],  # 网前线内角点
            [4, 476, 0], [606, 476, 0],  # 网前线外角点
            [76, 476, 0], [534, 476, 0],  # 网前线内角点
            
            # 网线区域 (Y = 666 to 670)
            [0, 666, 0], [610, 666, 0],  # 网线外角点
            [4, 666, 0], [606, 666, 0],  # 网线内角点
            [0, 670, 0], [610, 670, 0],  # 网线外角点
            [4, 670, 0], [606, 670, 0],  # 网线内角点
        ])
        
        # 对应的标签
        labels = [
            "底线左外角", "底线右外角", "底线左内角", "底线右内角",
            "发球线左外角1", "发球线右外角1", "发球线左内角1", "发球线右内角1",
            "发球线左外角2", "发球线右外角2", "发球线左内角2", "发球线右内角2",
            "中线左点1", "中线右点1", "中线左点2", "中线右点2",
            "网前线左外角1", "网前线右外角1", "网前线左内角1", "网前线右内角1",
            "网前线左外角2", "网前线右外角2", "网前线左内角2", "网前线右内角2",
            "网线左外角1", "网线右外角1", "网线左内角1", "网线右内角1",
            "网线左外角2", "网线右外角2", "网线左内角2", "网线右内角2",
        ]
        
        self.court_3d_points = pts.astype(np.float32)
        self.court_point_labels = labels
        self.merged_3d_points = pts.astype(np.float32)
        self.merged_point_labels = labels
    
    def _load_yolo_model(self):
        """加载YOLO模型"""
        if not os.path.exists(self.yolo_model_path):
            raise FileNotFoundError(f"YOLO model not found: {self.yolo_model_path}")
        
        self.yolo_model = YOLO(self.yolo_model_path)
    
    def detect_court_corners_yolov8(self, image):
        """使用YOLOv8检测场地角点"""
        results = self.yolo_model.predict(
            source=image,
            conf=0.3,
            iou=0.5,
            device=self.device,
            verbose=False
        )
        
        corners = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    corners.append((center_x, center_y))
        
        return corners
    
    def consolidate_corner_points(self, corners, threshold=30):
        """合并相近的角点"""
        if not corners:
            return []
        
        consolidated = []
        corners = np.array(corners)
        
        for corner in corners:
            if not consolidated:
                consolidated.append(corner)
            else:
                distances = [np.linalg.norm(corner - cons) for cons in consolidated]
                if min(distances) > threshold:
                    consolidated.append(corner)
        
        return consolidated
    
    def match_corners_to_3d_points(self, corners, boundary_points):
        """使用初始单应性矩阵辅助匹配检测到的角点与3D坐标"""
        if len(boundary_points) < 4:
            return {}
        
        # 创建初始单应性矩阵
        boundary_2d = np.array(boundary_points, dtype=np.float32)
        boundary_3d_2d = np.array([
            [0, 0], [610, 0], [610, 670], [0, 670]
        ], dtype=np.float32)
        
        self.homography_matrix = cv2.getPerspectiveTransform(boundary_2d, boundary_3d_2d)
        
        # 将检测到的角点投影到场地坐标系
        projected_corners = []
        for corner in corners:
            corner_array = np.array([[corner]], dtype=np.float32)
            projected = cv2.perspectiveTransform(corner_array, self.homography_matrix)[0][0]
            projected_corners.append(projected)
        
        # 匹配投影点与3D点
        matched = {}
        threshold = 50.0
        
        for i, proj_corner in enumerate(projected_corners):
            best_match_idx = None
            best_distance = float('inf')
            
            for j, point_3d in enumerate(self.merged_3d_points):
                distance = np.linalg.norm(proj_corner - point_3d[:2])
                if distance < threshold and distance < best_distance:
                    best_distance = distance
                    best_match_idx = j
            
            if best_match_idx is not None:
                matched[best_match_idx] = (
                    corners[i],
                    self.merged_3d_points[best_match_idx],
                    self.merged_point_labels[best_match_idx]
                )
        
        # 添加手动选择的角点
        manual_point_indices = [0, 1, 30, 28]  # 四个主要角点
        for i, corner in enumerate(self.manual_corners):
            if i < len(manual_point_indices):
                idx = manual_point_indices[i]
                point_3d = self.merged_3d_points[idx]
                label = self.merged_point_labels[idx]
                matched[idx] = (corner, point_3d, label)
        
        return matched
    
    def calibrate_extrinsic_parameters(self, matched_corners):
        """使用匹配的角点计算外参矩阵"""
        if len(matched_corners) < 4:
            raise ValueError("Need at least 4 matched corners for calibration")
        
        # 提取2D和3D点
        image_points = []
        object_points = []
        
        for corner_2d, point_3d, label in matched_corners.values():
            image_points.append(corner_2d)
            object_points.append(point_3d)
        
        image_points = np.array(image_points, dtype=np.float32)
        object_points = np.array(object_points, dtype=np.float32)
        
        # 使用solvePnP计算外参
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        if not success:
            raise RuntimeError("Failed to solve PnP")
        
        # 转换为旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        return rotation_matrix, tvec, rvec
    
    def save_calibration_results(self, output_dir, rotation_matrix, tvec, rvec):
        """保存标定结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存外参矩阵
        result_file = os.path.join(output_dir, "extrinsic_parameters.yaml")
        fs = cv2.FileStorage(result_file, cv2.FILE_STORAGE_WRITE)
        
        fs.write("rotation_matrix", rotation_matrix)
        fs.write("translation_vector", tvec)
        fs.write("rotation_vector", rvec)
        fs.write("camera_matrix", self.camera_matrix)
        fs.write("distortion_coefficients", self.dist_coeffs)
        fs.write("image_width", self.image_width)
        fs.write("image_height", self.image_height)
        
        fs.release()
        
        return result_file
    
    def calibrate_from_manual_points(self, manual_points):
        """从手动选择的点进行标定"""
        self.manual_corners = manual_points
        
        # 使用手动点匹配
        matched_corners = self.match_corners_to_3d_points([], manual_points)
        
        # 计算外参
        rotation_matrix, tvec, rvec = self.calibrate_extrinsic_parameters(matched_corners)
        
        return rotation_matrix, tvec, rvec, matched_corners
    
    def draw_court_lines(self, image, rotation_matrix=None, tvec=None):
        """在图像上绘制场地线（4cm宽）"""
        if rotation_matrix is None or tvec is None:
            return image
        
        result = image.copy()
        
        # 定义场地线的3D点
        court_lines = [
            # 外边界
            [(0, 0, 0), (610, 0, 0)],  # 底线
            [(0, 670, 0), (610, 670, 0)],  # 网线
            [(0, 0, 0), (0, 670, 0)],  # 左边线
            [(610, 0, 0), (610, 670, 0)],  # 右边线
            
            # 内边界
            [(4, 4, 0), (606, 4, 0)],  # 内底线
            [(4, 666, 0), (606, 666, 0)],  # 内网线
            [(4, 4, 0), (4, 666, 0)],  # 左内线
            [(606, 4, 0), (606, 666, 0)],  # 右内线
            
            # 发球线
            [(76, 198, 0), (534, 198, 0)],
            [(76, 472, 0), (534, 472, 0)],
            
            # 中线
            [(305, 198, 0), (305, 472, 0)],
        ]
        
        rvec, _ = cv2.Rodrigues(rotation_matrix)
        
        for line in court_lines:
            points_3d = np.array(line, dtype=np.float32)
            points_2d, _ = cv2.projectPoints(
                points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs
            )
            
            if len(points_2d) >= 2:
                pt1 = tuple(map(int, points_2d[0][0]))
                pt2 = tuple(map(int, points_2d[1][0]))
                cv2.line(result, pt1, pt2, (0, 255, 0), 2)
        
        return result