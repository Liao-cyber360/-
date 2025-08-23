import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from utils import config


class BadmintonCalibrator:
    """羽毛球相机标定类"""

    def __init__(self, camera_params_file, yolo_model_path, device='cpu'):
        """
        初始化标定器

        参数:
            camera_params_file: 内参标定文件路径
            yolo_model_path: YOLO检测模型路径
            device: 运行设备 ('cpu' 或 'cuda')
        """
        self.load_camera_params(camera_params_file)
        self.court_3d_points, self.court_point_labels, self.merged_3d_points, self.merged_point_labels = self.setup_court_points()
        self.current_image = None
        self.matched_corners = {}
        self.yolo_model = YOLO(yolo_model_path)
        self.device = device
        self.homography_matrix = None
        self.manual_corners = []  # 存储手动选择的四个点
        self.camera_params_file = camera_params_file

    def load_camera_params(self, params_file):
        """从文件加载相机内参"""
        fs = cv2.FileStorage(params_file, cv2.FILE_STORAGE_READ)
        self.camera_matrix = fs.getNode("camera_matrix").mat()
        self.dist_coeffs = fs.getNode("distortion_coefficients").mat().flatten()
        self.image_width = int(fs.getNode("image_width").real() or 1280)
        self.image_height = int(fs.getNode("image_height").real() or 720)
        fs.release()

        print(f"Camera parameters loaded from: {params_file}")
        print(f"Camera matrix: \n{self.camera_matrix}")
        print(f"Distortion coefficients: {self.dist_coeffs}")

    def setup_court_points(self):
        """设置场地3D点坐标"""
        # 仅使用近半场点 (Y=0 到 670)
        pts = np.array([
            # 底线区域 (Y = 0 to 4)
            [0, 0, 0], [610, 0, 0],  # 外角点
            [4, 4, 0], [606, 4, 0],  # 内角点
            [46, 4, 0], [50, 4, 0],  # 左单打边线交点
            [303, 4, 0], [307, 4, 0],  # 中线交点
            [560, 4, 0], [564, 4, 0],  # 右单打边线交点

            # 双打后发球线区域 (Y = 76 to 80)
            [4, 76, 0], [4, 80, 0],  # 左双打边线交点
            [46, 76, 0], [50, 76, 0], [46, 80, 0], [50, 80, 0],  # 左单打十字交点
            [303, 76, 0], [307, 76, 0], [303, 80, 0], [307, 80, 0],  # 中线十字交点
            [560, 76, 0], [564, 76, 0], [560, 80, 0], [564, 80, 0],  # 右单打十字交点
            [606, 76, 0], [606, 80, 0],  # 右双打边线交点

            # 前发球线区域 (Y = 468 to 472)
            [4, 468, 0], [4, 472, 0],  # 左双打边线交点
            [46, 468, 0], [50, 468, 0], [46, 472, 0], [50, 472, 0],  # 左单打十字交点
            [303, 468, 0], [307, 468, 0], [303, 472, 0], [307, 472, 0],  # 中线十字交点
            [560, 468, 0], [564, 468, 0], [560, 472, 0], [564, 472, 0],  # 右单打十字交点
            [606, 468, 0], [606, 472, 0],  # 右双打边线交点
        ], dtype=np.float32)

        # 为每个点添加标签，便于匹配和验证
        labels = [
            "Bottom Left Outer", "Bottom Right Outer",
            "Bottom Left Inner", "Bottom Right Inner",
            "Left Singles Bottom Left", "Left Singles Bottom Right",
            "Center Line Bottom Left", "Center Line Bottom Right",
            "Right Singles Bottom Left", "Right Singles Bottom Right",

            "Left Doubles Back Service Bottom", "Left Doubles Back Service Top",
            "Left Singles Back Service Bottom Left", "Left Singles Back Service Bottom Right",
            "Left Singles Back Service Top Left", "Left Singles Back Service Top Right",
            "Center Line Back Service Bottom Left", "Center Line Back Service Bottom Right",
            "Center Line Back Service Top Left", "Center Line Back Service Top Right",
            "Right Singles Back Service Bottom Left", "Right Singles Back Service Bottom Right",
            "Right Singles Back Service Top Left", "Right Singles Back Service Top Right",
            "Right Doubles Back Service Bottom", "Right Doubles Back Service Top",

            "Left Doubles Front Service Bottom", "Left Doubles Front Service Top",
            "Left Singles Front Service Bottom Left", "Left Singles Front Service Bottom Right",
            "Left Singles Front Service Top Left", "Left Singles Front Service Top Right",
            "Center Line Front Service Bottom Left", "Center Line Front Service Bottom Right",
            "Center Line Front Service Top Left", "Center Line Front Service Top Right",
            "Right Singles Front Service Bottom Left", "Right Singles Front Service Bottom Right",
            "Right Singles Front Service Top Left", "Right Singles Front Service Top Right",
            "Right Doubles Front Service Bottom", "Right Doubles Front Service Top"
        ]

        # 创建合并后的点集（将相近的十字角点合并为一个点）
        merged_pts = []
        merged_labels = []

        # 添加四个外角点 - 这些会被手动选择
        merged_pts.append([0, 0, 0])
        merged_labels.append("(0, 0, 0)")
        merged_pts.append([610, 0, 0])
        merged_labels.append("(610, 0, 0)")
        merged_pts.append([606, 472, 0])
        merged_labels.append("(606, 472, 0)")
        merged_pts.append([4, 472, 0])
        merged_labels.append("(4, 472, 0)")

        # 添加额外的两个点
        merged_pts.append([610, 472, 0])
        merged_labels.append("(610, 472, 0)")
        merged_pts.append([0, 472, 0])
        merged_labels.append("(0, 472, 0)")

        # 添加左单打边线底点（合并）
        merged_pts.append([48, 4, 0])
        merged_labels.append("(48, 4, 0)")

        # 添加中线底点（合并）
        merged_pts.append([305, 4, 0])
        merged_labels.append("(305, 4, 0)")

        # 添加右单打边线底点（合并）
        merged_pts.append([562, 4, 0])
        merged_labels.append("(562, 4, 0)")

        # 添加左双打后发球线交点（合并）- 从 (4, 76), (4, 80) 到 (4, 78)
        merged_pts.append([4, 78, 0])
        merged_labels.append("(4, 78, 0)")

        # 添加左单打后发球线交点（合并）
        merged_pts.append([48, 78, 0])
        merged_labels.append("(48, 78, 0)")

        # 添加中线后发球线交点（合并）
        merged_pts.append([305, 78, 0])
        merged_labels.append("(305, 78, 0)")

        # 添加右单打后发球线交点（合并）
        merged_pts.append([562, 78, 0])
        merged_labels.append("(562, 78, 0)")

        # 添加右双打后发球线交点（合并）- 从 (606, 76), (606, 80) 到 (606, 78)
        merged_pts.append([606, 78, 0])
        merged_labels.append("(606, 78, 0)")

        # 添加左双打前发球线交点（合并）- 从 (4, 468), (4, 472) 到 (4, 470)
        # 注意：我们不将 (4, 472) 合并进来，因为它是手动选择的点之一
        merged_pts.append([4, 470, 0])
        merged_labels.append("(4, 470, 0)")

        # 添加左单打前发球线交点（合并）
        merged_pts.append([48, 470, 0])
        merged_labels.append("(48, 470, 0)")

        # 添加中线前发球线交点（合并）
        merged_pts.append([305, 470, 0])
        merged_labels.append("(305, 470, 0)")

        # 添加右单打前发球线交点（合并）
        merged_pts.append([562, 470, 0])
        merged_labels.append("(562, 470, 0)")

        # 添加右双打前发球线交点（合并）
        # 注意：我们不将 (606, 472) 合并进来，因为它是手动选择的点之一
        merged_pts.append([606, 470, 0])
        merged_labels.append("(606, 470, 0)")

        return pts, labels, np.array(merged_pts, dtype=np.float32), merged_labels

    def zoom_point_selection(self, image, roi_x, roi_y, zoom_radius, point_name):
        """放大区域进行精确点选择，使用左键在放大区域中选择点"""
        # 提取放大区域
        h, w = image.shape[:2]
        x1 = max(0, roi_x - zoom_radius)
        y1 = max(0, roi_y - zoom_radius)
        x2 = min(w, roi_x + zoom_radius)
        y2 = min(h, roi_y + zoom_radius)

        zoom_img = image[y1:y2, x1:x2].copy()

        # 创建放大窗口
        window = f"Zoom Selection - {point_name}"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 800, 800)

        selected_point = [None]

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_point[0] = (x, y)

        cv2.setMouseCallback(window, mouse_cb)

        # 显示指导信息
        disp = zoom_img.copy()
        text = f"Click exactly on the {point_name}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(disp, (10, 30), (10 + text_size[0] + 10, 70), (0, 0, 0), -1)
        cv2.putText(disp, text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 添加一个十字线或圆形标记在图像中心，帮助定位
        center_x, center_y = zoom_img.shape[1] // 2, zoom_img.shape[0] // 2
        cv2.line(disp, (center_x - 10, center_y), (center_x + 10, center_y), (0, 255, 255), 1)
        cv2.line(disp, (center_x, center_y - 10), (center_x, center_y + 10), (0, 255, 255), 1)

        while True:
            # 如果有选择的点，画出来
            if selected_point[0] is not None:
                disp = zoom_img.copy()
                cv2.putText(disp, text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.circle(disp, selected_point[0], 5, (0, 255, 255), -1)
                cv2.putText(disp, "Press SPACE to confirm or ESC to re-select",
                            (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow(window, disp)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                selected_point[0] = None  # 重置
            elif key == ord(' ') and selected_point[0] is not None:  # SPACE
                break

        cv2.destroyWindow(window)

        # 如果用户选择了点，转换回原图坐标
        if selected_point[0] is not None:
            global_x = x1 + selected_point[0][0]
            global_y = y1 + selected_point[0][1]
            return (global_x, global_y)

        return None

    def select_initial_court_boundary(self, image):
        """让操作员依次精确点击四个场地角点，使用右键直接显示放大区域"""
        points = []
        window = "Select Court Boundary"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        # 设置提示文本
        point_names = [
            "Bottom Left Outer (0,0)",
            "Bottom Right Outer (610,0)",
            "Top Right Inner (606,472)",
            "Top Left Inner (4,472)"
        ]

        current_point_idx = 0
        zoom_radius = int(min(image.shape[0], image.shape[1]) * 0.1)

        # 使用闭包使self在回调中可用
        calibrator = self

        def mouse_cb(event, x, y, flags, param):
            nonlocal current_point_idx, calibrator

            # 右键点击 - 直接激活放大区域并选择点
            if event == cv2.EVENT_RBUTTONDOWN and current_point_idx < 4:
                # 直接显示放大视图并进行精确选择
                selected_point = calibrator.zoom_point_selection(
                    image,
                    x,
                    y,
                    zoom_radius,
                    point_names[current_point_idx]
                )

                if selected_point is not None:
                    points.append(selected_point)
                    current_point_idx += 1

        cv2.setMouseCallback(window, mouse_cb)

        while current_point_idx < 4:
            disp = image.copy()

            # 显示当前需要点击的位置提示
            text = f"Right-click near {point_names[current_point_idx]} to zoom and select"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(disp, (45, 35), (45 + text_size[0] + 10, 55 + text_size[1]), (0, 0, 0), -1)
            cv2.putText(disp, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 显示已选择的点
            for i, pt in enumerate(points):
                cv2.circle(disp, pt, 8, (0, 255, 255), -1)
                cv2.putText(disp, f"{i + 1}", (pt[0] + 10, pt[1] + 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)

            cv2.imshow(window, disp)
            key = cv2.waitKey(1) & 0xFF

            # ESC键清除所有点重新开始
            if key == 27:
                points = []
                current_point_idx = 0

        cv2.destroyWindow(window)

        # 保存手动选择的四个点供后续标定使用
        self.manual_corners = points.copy()

        # 创建掩码图像
        mask = np.zeros(image.shape[:2], np.uint8)
        if len(points) == 4:
            cv2.fillPoly(mask, [np.array(points)], 255)

        # 略微向外扩展多边形（扩展5%）
        expanded_points = []
        if len(points) == 4:
            # 计算中心点
            center_x = sum(p[0] for p in points) / 4
            center_y = sum(p[1] for p in points) / 4

            # 向外扩展
            for p in points:
                vector_x = p[0] - center_x
                vector_y = p[1] - center_y
                expanded_points.append((
                    int(p[0] + vector_x * 0.05),
                    int(p[1] + vector_y * 0.15)
                ))

            # 更新掩码
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.fillPoly(mask, [np.array(expanded_points)], 255)

        # 创建带遮罩的图像
        masked_img = image.copy()
        # 将遮罩外区域进行马赛克处理
        outside_mask = cv2.bitwise_not(mask)
        # 创建马赛克效果
        mosaic_img = image.copy()
        mosaic_block_size = 20
        h, w = image.shape[:2]

        for i in range(0, h, mosaic_block_size):
            for j in range(0, w, mosaic_block_size):
                if i + mosaic_block_size <= h and j + mosaic_block_size <= w:
                    roi = image[i:i + mosaic_block_size, j:j + mosaic_block_size]
                    color = roi.mean(axis=(0, 1))
                    mosaic_img[i:i + mosaic_block_size, j:j + mosaic_block_size] = color

        # 将马赛克应用到遮罩外区域
        masked_img = cv2.bitwise_and(masked_img, masked_img, mask=mask)
        mosaic_outside = cv2.bitwise_and(mosaic_img, mosaic_img, mask=outside_mask)
        final_img = cv2.add(masked_img, mosaic_outside)

        # 计算初始单应性矩阵用于辅助匹配
        court_corners_2d = np.array([
            [0, 0],
            [610, 0],
            [606, 472],
            [4, 472]
        ], dtype=np.float32)

        self.initial_homography = cv2.findHomography(np.array(points), court_corners_2d)[0]

        return final_img, mask, points

    def capture_and_process_frames(self, video_path, num_frames=30):
        """从视频中提取连续多帧并处理"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return None, None, None

        # 获取视频基本信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print("Error: Could not determine total frames in video")
            cap.release()
            return None, None, None

        # 获取第一帧用于初始边界选择
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            cap.release()
            return None, None, None

        # 让用户选择初始场地边界
        masked_first_frame, mask, boundary_points = self.select_initial_court_boundary(first_frame)

        # 选择起始帧位置（用户可以调整）
        starting_frame = 0
        frame_selection_window = "Select Starting Frame"
        cv2.namedWindow(frame_selection_window, cv2.WINDOW_NORMAL)

        def on_trackbar_change(val):
            cap.set(cv2.CAP_PROP_POS_FRAMES, val)
            ret, frame = cap.read()
            if ret:
                # 应用相同的掩码处理
                masked_frame = frame.copy()
                masked_frame = cv2.bitwise_and(masked_frame, masked_frame, mask=mask)

                # 创建马赛克效果
                mosaic_frame = frame.copy()
                mosaic_block_size = 20
                h, w = frame.shape[:2]

                for y in range(0, h, mosaic_block_size):
                    for x in range(0, w, mosaic_block_size):
                        if y + mosaic_block_size <= h and x + mosaic_block_size <= w:
                            roi = frame[y:y + mosaic_block_size, x:x + mosaic_block_size]
                            color = roi.mean(axis=(0, 1))
                            mosaic_frame[y:y + mosaic_block_size, x:x + mosaic_block_size] = color

                outside_mask = cv2.bitwise_not(mask)
                mosaic_outside = cv2.bitwise_and(mosaic_frame, mosaic_frame, mask=outside_mask)
                final_frame = cv2.add(masked_frame, mosaic_outside)

                # 显示框架编号
                cv2.putText(final_frame, f"Frame: {val}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 2)
                cv2.imshow(frame_selection_window, final_frame)

        # 创建轨迹条
        cv2.createTrackbar("Frame", frame_selection_window, 0, max(0, total_frames - num_frames - 1),
                           on_trackbar_change)

        # 初始显示
        on_trackbar_change(0)

        # 等待用户选择并确认
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # 空格确认
                starting_frame = cv2.getTrackbarPos("Frame", frame_selection_window)
                break
            elif key == 27:  # ESC取消
                starting_frame = 0
                break

        cv2.destroyWindow(frame_selection_window)

        # 将视频定位到选择的起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)

        processed_frames = []
        all_detected_corners = []

        # 展示处理进度的窗口
        progress_window = "Processing Frames"
        cv2.namedWindow(progress_window, cv2.WINDOW_NORMAL)

        for i in range(num_frames):
            # 读取连续帧
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could only read {i} frames of {num_frames} requested")
                break

            # 应用相同的掩码和马赛克处理
            masked_frame = frame.copy()
            masked_frame = cv2.bitwise_and(masked_frame, masked_frame, mask=mask)

            # 创建马赛克效果用于遮罩外区域
            mosaic_frame = frame.copy()
            mosaic_block_size = 20
            h, w = frame.shape[:2]

            for y in range(0, h, mosaic_block_size):
                for x in range(0, w, mosaic_block_size):
                    if y + mosaic_block_size <= h and x + mosaic_block_size <= w:
                        roi = frame[y:y + mosaic_block_size, x:x + mosaic_block_size]
                        color = roi.mean(axis=(0, 1))
                        mosaic_frame[y:y + mosaic_block_size, x:x + mosaic_block_size] = color

            outside_mask = cv2.bitwise_not(mask)
            mosaic_outside = cv2.bitwise_and(mosaic_frame, mosaic_frame, mask=outside_mask)
            final_frame = cv2.add(masked_frame, mosaic_outside)

            # 使用YOLOv8检测角点
            corners = self.detect_court_corners_yolov8(final_frame)
            all_detected_corners.extend(corners)

            # 保存处理后的帧
            processed_frames.append(final_frame)

            # 显示处理进度并改进文本可见性
            progress_img = final_frame.copy()
            # 在图像上显示检测到的角点
            for pt in corners:
                cv2.circle(progress_img, pt, 4, (0, 0, 255), -1)  # 将点大小改回4

            # 在文本后添加背景以提高可见性
            progress_text = f"Processing frame {i + 1}/{num_frames}"
            text_size, _ = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(progress_img, (45, 35), (45 + text_size[0] + 10, 55 + text_size[1]), (0, 0, 0), -1)
            cv2.putText(progress_img, progress_text,
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow(progress_window, progress_img)
            cv2.waitKey(10)  # 短暂显示

        cv2.destroyWindow(progress_window)
        cap.release()

        # 使用较大的聚类阈值合并角点
        consolidated_corners = self.consolidate_corner_points(all_detected_corners, threshold=30)  # 聚类阈值改回30

        return processed_frames, consolidated_corners, boundary_points

    def detect_court_corners_yolov8(self, image):
        """使用YOLOv8检测场地角点"""
        results = self.yolo_model(image)
        corners = []

        for r in results:
            # 检查是否有关键点结果
            if hasattr(r, 'keypoints') and r.keypoints is not None:
                kpts = r.keypoints.xy.cpu().numpy() if hasattr(r.keypoints.xy, "cpu") else r.keypoints.xy
                for kp_list in kpts:
                    for kp in kp_list:
                        if not np.isnan(kp).any():  # 排除无效点
                            corners.append((int(kp[0]), int(kp[1])))

        return corners

    def consolidate_corner_points(self, corners, threshold=30):
        """合并相近的角点"""
        if not corners:
            return []

        clusters = []
        for point in corners:
            # 查找点所属的聚类
            found_cluster = False
            for cluster in clusters:
                for cluster_point in cluster:
                    # 如果点与聚类中的任一点距离小于阈值，将其加入该聚类
                    if np.linalg.norm(np.array(point) - np.array(cluster_point)) < threshold:
                        cluster.append(point)
                        found_cluster = True
                        break
                if found_cluster:
                    break

            # 如果未找到所属聚类，创建新聚类
            if not found_cluster:
                clusters.append([point])

        # 对每个聚类计算中心点
        consolidated_points = []
        for cluster in clusters:
            if len(cluster) >= 1:
                x_mean = int(np.mean([p[0] for p in cluster]))
                y_mean = int(np.mean([p[1] for p in cluster]))
                consolidated_points.append((x_mean, y_mean))

        # 将手动选择的四个角点添加到合并后的点集
        for corner in self.manual_corners:
            if corner not in consolidated_points:
                # 检查是否已经有非常接近的点
                is_close = False
                for pt in consolidated_points:
                    if np.linalg.norm(np.array(corner) - np.array(pt)) < threshold:
                        is_close = True
                        break
                if not is_close:
                    consolidated_points.append(corner)

        return consolidated_points

    def match_corners_to_3d_points(self, corners, boundary_points):
        """使用初始单应性矩阵辅助匹配检测到的角点与3D坐标"""
        if not corners or self.initial_homography is None:
            return {}

        matched = {}

        # 使用单应性矩阵将3D点投影到图像平面
        projected_3d_points = {}
        for i, (point_3d, label) in enumerate(zip(self.merged_3d_points, self.merged_point_labels)):
            # 将3D点投影到2D平面 (z=0，所以只需考虑x,y)
            pt_2d = np.array([point_3d[0], point_3d[1]], dtype=np.float32).reshape(1, 1, 2)
            projected_pt = cv2.perspectiveTransform(pt_2d, np.linalg.inv(self.initial_homography))
            projected_pt = (int(projected_pt[0][0][0]), int(projected_pt[0][0][1]))
            projected_3d_points[i] = (projected_pt, point_3d, label)

        # 为每个检测到的角点找最近的3D点
        for corner in corners:
            min_dist = float('inf')
            best_match = None

            for idx, (projected_pt, point_3d, label) in projected_3d_points.items():
                dist = np.linalg.norm(np.array(corner) - np.array(projected_pt))
                if dist < min_dist and dist < 50:  # 设置最大匹配距离
                    min_dist = dist
                    best_match = (idx, point_3d, label)

            if best_match:
                matched[best_match[0]] = (corner, best_match[1], best_match[2])

        # 确保手动点击的四个点被匹配
        # 找到对应的3D点索引 (0, 0), (610, 0), (606, 472), (4, 472)
        manual_point_indices = [0, 1, 2, 3]  # 对应 merged_3d_points 中的索引

        for i, corner in enumerate(self.manual_corners):
            if i < len(manual_point_indices):
                idx = manual_point_indices[i]
                point_3d = self.merged_3d_points[idx]
                label = self.merged_point_labels[idx]
                matched[idx] = (corner, point_3d, label)

        return matched

    def calibrate_extrinsic_parameters(self, matched_corners):
        """使用匹配的角点计算外参矩阵"""
        if not matched_corners:
            return False

        # 提取匹配的2D和3D点
        points_2d = []
        points_3d = []

        for idx, (corner, point_3d, _) in matched_corners.items():
            points_2d.append(corner)
            points_3d.append(point_3d)

        points_2d = np.array(points_2d, dtype=np.float32)
        points_3d = np.array(points_3d, dtype=np.float32)

        # 至少需要6个点
        if len(points_2d) < 6:
            print("Error: Need at least 6 matched points for reliable calibration")
            return False

        # 计算PnP解
        success, rotation_vector, translation_vector = cv2.solvePnP(
            points_3d, points_2d, self.camera_matrix, self.dist_coeffs)

        if not success:
            print("Error: Failed to solve PnP")
            return False

        # 保存外参结果
        self.rotation_vector = rotation_vector
        self.translation_vector = translation_vector

        # 创建投影矩阵
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        projection_matrix = np.hstack((rotation_matrix, translation_vector))
        self.projection_matrix = np.dot(self.camera_matrix, projection_matrix)

        return True

    def draw_court_lines(self, image):
        """在图像上绘制场地线（4cm宽）"""
        if not hasattr(self, 'rotation_vector') or not hasattr(self, 'translation_vector'):
            return image

        result = image.copy()

        # 定义场地线段对（每对线段定义一条4cm宽的线）
        court_line_pairs = [
            # 底线对
            (([0, 0, 0], [610, 0, 0]), ([4, 4, 0], [606, 4, 0])),

            # 左双打边线
            (([0, 0, 0], [0, 472, 0]), ([4, 4, 0], [4, 472, 0])),

            # 右双打边线
            (([610, 0, 0], [610, 472, 0]), ([606, 4, 0], [606, 472, 0])),

            # 左单打边线
            (([46, 4, 0], [46, 472, 0]), ([50, 4, 0], [50, 472, 0])),

            # 右单打边线
            (([560, 4, 0], [560, 472, 0]), ([564, 4, 0], [564, 472, 0])),

            # 中线
            (([303, 4, 0], [303, 472, 0]), ([307, 4, 0], [307, 472, 0])),

            # 后发球线
            (([4, 76, 0], [606, 76, 0]), ([4, 80, 0], [606, 80, 0])),

            # 前发球线
            (([4, 468, 0], [606, 468, 0]), ([4, 472, 0], [606, 472, 0])),

            # 顶线对
            (([0, 472, 0], [610, 472, 0]), ([4, 472, 0], [606, 472, 0])),
        ]

        # 绘制每条线
        for (start1_3d, end1_3d), (start2_3d, end2_3d) in court_line_pairs:
            # 投影第一条线的端点
            start1_3d = np.array([start1_3d], dtype=np.float32).reshape(1, 3)
            end1_3d = np.array([end1_3d], dtype=np.float32).reshape(1, 3)

            start1_2d, _ = cv2.projectPoints(start1_3d, self.rotation_vector,
                                             self.translation_vector, self.camera_matrix, self.dist_coeffs)
            end1_2d, _ = cv2.projectPoints(end1_3d, self.rotation_vector,
                                           self.translation_vector, self.camera_matrix, self.dist_coeffs)

            # 投影第二条线的端点
            start2_3d = np.array([start2_3d], dtype=np.float32).reshape(1, 3)
            end2_3d = np.array([end2_3d], dtype=np.float32).reshape(1, 3)

            start2_2d, _ = cv2.projectPoints(start2_3d, self.rotation_vector,
                                             self.translation_vector, self.camera_matrix, self.dist_coeffs)
            end2_2d, _ = cv2.projectPoints(end2_3d, self.rotation_vector,
                                           self.translation_vector, self.camera_matrix, self.dist_coeffs)

            # 转换为整数坐标
            start1 = (int(start1_2d[0][0][0]), int(start1_2d[0][0][1]))
            end1 = (int(end1_2d[0][0][0]), int(end1_2d[0][0][1]))
            start2 = (int(start2_2d[0][0][0]), int(start2_2d[0][0][1]))
            end2 = (int(end2_2d[0][0][0]), int(end2_2d[0][0][1]))

            # 创建填充的多边形来表示宽线
            pts = np.array([start1, end1, end2, start2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(result, [pts], (0, 255, 0))

            # 同时在边缘画线以提高清晰度
            cv2.line(result, start1, end1, (255, 255, 255), 1)
            cv2.line(result, start2, end2, (255, 255, 255), 1)
            cv2.line(result, start1, start2, (255, 255, 255), 1)
            cv2.line(result, end1, end2, (255, 255, 255), 1)

        return result

    def save_calibration_results(self, output_dir, processed_frames, final_frame):
        """保存标定结果"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存最终结果帧
        cv2.imwrite(os.path.join(output_dir, "calibration_result.jpg"), final_frame)

        # 保存标定参数
        if hasattr(self, 'rotation_vector') and hasattr(self, 'translation_vector'):
            fs = cv2.FileStorage(os.path.join(output_dir, "extrinsic_parameters.yaml"), cv2.FILE_STORAGE_WRITE)
            fs.write("camera_matrix", self.camera_matrix)
            fs.write("distortion_coefficients", self.dist_coeffs)
            fs.write("rotation_vector", self.rotation_vector)
            fs.write("translation_vector", self.translation_vector)
            fs.write("projection_matrix", self.projection_matrix)

            # 保存时间戳
            fs.write("calibration_date", time.strftime("%Y-%m-%d_%H-%M-%S"))
            fs.write("image_width", self.image_width)
            fs.write("image_height", self.image_height)

            fs.release()

            print(f"Calibration results saved to {output_dir}")
            return os.path.join(output_dir, "extrinsic_parameters.yaml")

        return None

    def calibrate_from_video(self, video_path, output_dir="./calibration_results"):
        """从视频进行外参标定的主函数"""
        # 显示初始指导
        cv2.namedWindow("Calibration Guide", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration Guide", 800, 600)

        guide_img = np.zeros((600, 800, 3), dtype=np.uint8)
        guide_texts = [
            "Badminton Court Calibration",
            "",
            "1. First frame will be shown for selecting court corners",
            "2. Right-click near each corner to select precise position",
            "3. Follow order: Bottom Left, Bottom Right, Top Right, Top Left",
            "4. After corner selection, automatic court point detection will run",
            "",
            "Press SPACE to continue..."
        ]

        for i, text in enumerate(guide_texts):
            cv2.putText(guide_img, text, (50, 100 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Calibration Guide", guide_img)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
        cv2.destroyWindow("Calibration Guide")

        # 直接调用capture_and_process_frames获取处理帧和检测到的角点
        processed_frames, detected_corners, boundary_points = self.capture_and_process_frames(video_path, num_frames=20)

        if not processed_frames or not detected_corners:
            print("Error: Failed to process frames or detect corners")
            return False

        # 匹配角点与3D坐标
        matched_corners = self.match_corners_to_3d_points(detected_corners, boundary_points)

        # 直接使用匹配的角点计算外参，跳过交互式验证
        if not self.calibrate_extrinsic_parameters(matched_corners):
            print("Error: Failed to calculate extrinsic parameters")
            return False

        # 绘制场地线
        result_frame = self.draw_court_lines(processed_frames[0])

        # 保存结果
        output_file = self.save_calibration_results(output_dir, processed_frames, result_frame)

        # 显示最终结果
        cv2.namedWindow("Calibration Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Calibration Result", result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return output_file


def calibrate_cameras(video_path1, video_path2, output_dir):
    """对两个摄像机进行标定"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 相机1标定
    print("Calibrating Camera 1...")
    calibrator1 = BadmintonCalibrator(config.camera_params_file_1, config.yolo_court_model)
    extrinsic_file1 = calibrator1.calibrate_from_video(video_path1, os.path.join(output_dir, "camera1"))

    # 相机2标定
    print("Calibrating Camera 2...")
    calibrator2 = BadmintonCalibrator(config.camera_params_file_2, config.yolo_court_model)
    extrinsic_file2 = calibrator2.calibrate_from_video(video_path2, os.path.join(output_dir, "camera2"))

    # 返回标定文件路径
    return extrinsic_file1, extrinsic_file2