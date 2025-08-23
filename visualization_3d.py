import threading
import time
import copy
import numpy as np
import open3d as o3d


class Interactive3DVisualizer:
    """Enhanced Interactive 3D visualization with fixed landing point colors and camera view"""

    def __init__(self, width=610, height=1340):
        self.width = width + 200
        self.height = height + 200
        self.net_height = 155
        self.court_width = width
        self.court_height = height

        # Window management - Enhanced state tracking
        self.vis = None
        self.view_control = None
        self.window_created = False
        self.window_visible = False
        self.window_should_close = False
        self.creation_in_progress = False

        # Fixed initialization
        self.view_initialized = False
        self.initialization_attempts = 0
        self.max_init_attempts = 3

        # Geometry objects storage - Centralized management
        self.geometries = {}
        self.geometry_names = [
            'court_lines', 'net', 'floor', 'all_valid_points',
            'prediction_trajectory_points', 'rejected_points',
            'low_quality_points', 'triangulation_failed_points',
            'predicted_trajectory_line', 'landing_point'
        ]

        # Data management with thread safety
        self.data_lock = threading.Lock()
        self.reset_data()

        # Visualization controls - Extended options
        self.visibility_flags = {
            'all_valid_points': True,
            'prediction_points': True,
            'rejected_points': True,
            'low_quality_points': True,
            'triangulation_failed': True,
            'predicted_trajectory': True
        }

        # Update control - Optimized timing
        self.last_update_time = 0
        self.update_interval = 1.0 / 20.0
        self.needs_geometry_update = False
        self.geometry_update_lock = threading.Lock()

        print(f"Enhanced 3D Visualizer initialized at {time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"User: {self._get_current_user()}")

    def _get_current_user(self):
        """Get current user login"""
        return "Liao-cyber360"

    def reset_data(self):
        """Reset all visualization data to clean state"""
        self.all_valid_points_data = []
        self.prediction_points_data = []
        self.rejected_points_data = []
        self.low_quality_points_data = []
        self.triangulation_failed_data = []
        self.predicted_trajectory_data = []
        self.landing_position = None
        self.in_bounds = None
        self.last_landing_position = np.array([0, 0, 0])

    def _create_window_safely(self):
        """Safely create and initialize the 3D window with comprehensive error handling"""
        if self.creation_in_progress:
            print("⚠️ Window creation already in progress")
            return False

        try:
            self.creation_in_progress = True

            # Clean up any existing visualizer
            if self.vis is not None:
                self._cleanup_visualizer()

            # Create fresh visualizer instance
            self.vis = o3d.visualization.Visualizer()

            # Create window with error handling
            success = self.vis.create_window(
                window_name="Badminton 3D Debug Visualization (Press Q in main window to close)",
                width=1200,
                height=800,
                left=100,
                top=100
            )

            if not success:
                print("❌ Failed to create Open3D window")
                self._cleanup_visualizer()
                return False

            # Configure rendering options
            render_option = self.vis.get_render_option()
            render_option.background_color = np.array([0.1, 0.1, 0.1])
            render_option.point_size = 8.0
            render_option.line_width = 3.0
            render_option.show_coordinate_frame = True

            # Initialize all geometries
            self._initialize_all_geometries()

            # Ensure window is stable before setting view
            for _ in range(5):
                self.vis.poll_events()
                self.vis.update_renderer()
                time.sleep(0.03)

            # Get view control after window is fully ready
            self.view_control = self.vis.get_view_control()

            # Set initial view with retry mechanism
            view_set = self._set_initial_view_with_retry()

            # Update window state
            self.window_created = True
            self.window_visible = True
            self.window_should_close = False
            self.view_initialized = view_set

            print("✅ 3D visualization window created successfully")
            print(f"   View initialized: {self.view_initialized}")
            return True

        except Exception as e:
            print(f"❌ Error creating 3D window: {e}")
            import traceback
            traceback.print_exc()
            self._cleanup_visualizer()
            return False
        finally:
            self.creation_in_progress = False

    def _set_initial_view_with_retry(self):
        """Set initial view with proper retry mechanism and FIXED camera positioning"""
        if not self.view_control:
            print("⚠️ View control not available")
            return False

        for attempt in range(3):
            try:
                # Give the window time to be ready
                time.sleep(0.1)

                # 🔧 FIXED: 更合适的相机视角设置
                # 设置相机位置到场地斜上方，距离适中
                court_center_x = self.width / 2  # 约 405
                court_center_y = self.height / 2  # 约 770
                court_center_z = 100  # 1米高度

                # 1. 设置观察目标点（场地中心）
                self.view_control.set_lookat([court_center_x, court_center_y, court_center_z])
                time.sleep(0.05)

                # 2. 设置相机前方向（从斜上方观看）
                # 这个向量决定了相机朝向，调整为更合适的角度
                self.view_control.set_front([0.2, -0.5, 0.8])  # 从斜前上方观看
                time.sleep(0.05)

                # 3. 设置上方向
                self.view_control.set_up([0, 0, 1])  # Z轴向上
                time.sleep(0.05)

                # 4. 设置合适的缩放级别
                self.view_control.set_zoom(0.3)  # 增加缩放，拉近距离

                # 测试视角设置是否成功
                try:
                    current_params = self.view_control.convert_to_pinhole_camera_parameters()
                    if current_params is not None:
                        print(f"✅ Enhanced view set successfully on attempt {attempt + 1}")
                        print(f"   Camera target: ({court_center_x:.0f}, {court_center_y:.0f}, {court_center_z:.0f})")
                        print(f"   Zoom level: 0.8 (closer view)")
                        return True
                except:
                    pass

            except Exception as e:
                if attempt < 2:
                    print(f"⚠️ View setup attempt {attempt + 1} failed: {e}")
                    time.sleep(0.1)
                else:
                    print(f"❌ Failed to set initial view after {attempt + 1} attempts: {e}")

        # Even if view setting failed, the window might still be usable
        print("⚠️ View setting failed, but window should still be interactive")
        return False

    def _initialize_all_geometries(self):
        """Initialize all geometry objects and add to visualizer"""
        try:
            # Create court and environment
            self.geometries['court_lines'] = o3d.geometry.LineSet()
            self.geometries['net'] = o3d.geometry.LineSet()

            # Create floor
            self.geometries['floor'] = o3d.geometry.TriangleMesh.create_box(
               width=self.width, height=0.1, depth=self.height
            )
            self.geometries['floor'].translate([0, -0.05, 0])
            self.geometries['floor'].paint_uniform_color([0.2, 0.2, 0.2])

            # Create point clouds for different data types
            point_cloud_names = [
                'all_valid_points', 'prediction_trajectory_points', 'rejected_points',
                'low_quality_points', 'triangulation_failed_points'
            ]
            for name in point_cloud_names:
                self.geometries[name] = o3d.geometry.PointCloud()

            # Create trajectory line and landing point
            self.geometries['predicted_trajectory_line'] = o3d.geometry.LineSet()
            self.geometries['landing_point'] = o3d.geometry.TriangleMesh.create_sphere(radius=15)  # 稍微大一点便于观察

            # Move landing point out of view initially
            self.geometries['landing_point'].translate([10000, 10000, 10000])

            # Create court and net geometry
            self._create_court()
            self._create_net()

            # Add all geometries to visualizer
            for name, geometry in self.geometries.items():
                self.vis.add_geometry(geometry)

            print("✅ All geometries initialized and added to visualizer")

        except Exception as e:
            print(f"❌ Error initializing geometries: {e}")
            raise

    def _create_court(self):
        """Create badminton court lines with proper coordinates and enhanced visibility"""
        offset_x = 100
        offset_y = 100
        court_width = self.court_width
        court_height = self.court_height

        # Court points with enhanced layout
        points = [
            # Extended area boundary (lighter gray)
            [0, 0, 0], [self.width, 0, 0], [self.width, self.height, 0], [0, self.height, 0],

            # Actual court boundary (doubles) - WHITE for visibility
            [offset_x, offset_y, 0], [offset_x + court_width, offset_y, 0],
            [offset_x + court_width, offset_y + court_height, 0], [offset_x, offset_y + court_height, 0],

            # Singles court boundary - BRIGHT BLUE
            [offset_x + 46, offset_y, 0], [offset_x + 564, offset_y, 0],
            [offset_x + 564, offset_y + court_height, 0], [offset_x + 46, offset_y + court_height, 0],

            # Service lines - YELLOW for better visibility
            [offset_x, offset_y + 76, 0], [offset_x + court_width, offset_y + 76, 0],  # Back service line
            [offset_x, offset_y + 468, 0], [offset_x + court_width, offset_y + 468, 0],  # Front service line

            # Center line - BRIGHT GREEN
            [offset_x + court_width / 2, offset_y, 0], [offset_x + court_width / 2, offset_y + court_height, 0],
        ]

        # Court lines connections
        lines = [
            # Extended area (4 lines)
            [0, 1], [1, 2], [2, 3], [3, 0],
            # Doubles court (4 lines)
            [4, 5], [5, 6], [6, 7], [7, 4],
            # Singles court (4 lines)
            [8, 9], [9, 10], [10, 11], [11, 8],
            # Service lines (2 lines)
            [12, 13], [14, 15],
            # Center line (1 line)
            [16, 17]
        ]

        # Enhanced line colors for better visibility
        colors = []
        colors.extend([[0.5, 0.5, 0.5]] * 4)  # Extended area - dark gray
        colors.extend([[1, 1, 1]] * 4)  # Doubles court - bright white
        colors.extend([[0.3, 0.7, 1]] * 4)  # Singles court - bright blue
        colors.extend([[1, 1, 0]] * 2)  # Service lines - yellow
        colors.extend([[0, 1, 0]] * 1)  # Center line - bright green

        self.geometries['court_lines'].points = o3d.utility.Vector3dVector(points)
        self.geometries['court_lines'].lines = o3d.utility.Vector2iVector(lines)
        self.geometries['court_lines'].colors = o3d.utility.Vector3dVector(colors)

    def _create_net(self):
        """Create badminton net with enhanced visibility"""
        offset_x = 100
        offset_y = 100
        court_width = self.court_width
        court_height = self.court_height
        net_height = self.net_height

        # Net points
        points = [
            [offset_x, offset_y + court_height / 2, 0],  # Left bottom
            [offset_x + court_width, offset_y + court_height / 2, 0],  # Right bottom
            [offset_x + court_width, offset_y + court_height / 2, net_height],  # Right top
            [offset_x, offset_y + court_height / 2, net_height],  # Left top
        ]

        # Net lines
        lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
        colors = [[1, 1, 1]] * 4  # Bright white for better visibility

        self.geometries['net'].points = o3d.utility.Vector3dVector(points)
        self.geometries['net'].lines = o3d.utility.Vector2iVector(lines)
        self.geometries['net'].colors = o3d.utility.Vector3dVector(colors)

    def update_landing_point(self, position, in_bounds):
        """🔧 FIXED: Update landing point position and boundary status with correct color logic"""
        with self.data_lock:
            if position is not None:
                if len(position) == 2:
                    self.landing_position = [position[0], position[1], 0]
                else:
                    self.landing_position = copy.deepcopy(position)

                # 🔧 FIXED: 确保 in_bounds 状态正确传递和存储
                self.in_bounds = bool(in_bounds)  # 确保是布尔值

                status = 'IN BOUNDS' if self.in_bounds else 'OUT OF BOUNDS'
                print(f"🎯 Landing point updated: ({position[0]:.1f}, {position[1]:.1f}) - {status}")
                print(f"   Color will be: {'GREEN' if self.in_bounds else 'RED'}")
            else:
                self.landing_position = None
                self.in_bounds = None
                print("🎯 Landing point cleared")
            self.needs_geometry_update = True

    def _update_landing_point(self):
        """🔧 FIXED: Update landing point sphere geometry with CORRECT color logic"""
        try:
            if self.landing_position is not None and 'landing_point' in self.geometries:
                new_position = np.array([
                    self.landing_position[0] + 100,  # 转换到可视化坐标系
                    self.landing_position[1] + 100,
                    self.landing_position[2] if len(self.landing_position) > 2 else 5  # 稍微离地面高一点
                ])

                # Calculate displacement from last position
                displacement = new_position - self.last_landing_position
                self.geometries['landing_point'].translate(displacement)

                # 🔧 FIXED: 正确的颜色设置逻辑
                if self.in_bounds is not None:  # 确保有边界判定结果
                    if self.in_bounds:
                        # 界内 - 绿色
                        self.geometries['landing_point'].paint_uniform_color([0, 1, 0])
                        color_name = "GREEN (IN BOUNDS)"
                    else:
                        # 界外 - 红色
                        self.geometries['landing_point'].paint_uniform_color([1, 0, 0])
                        color_name = "RED (OUT OF BOUNDS)"

                    print(f"🎨 Landing point color set to: {color_name}")
                else:
                    # 没有边界判定信息 - 黄色表示未知
                    self.geometries['landing_point'].paint_uniform_color([1, 1, 0])
                    print("🎨 Landing point color set to: YELLOW (UNKNOWN)")

                self.last_landing_position = new_position
            else:
                # Move landing point out of view when no valid position
                displacement = np.array([10000, 10000, 10000]) - self.last_landing_position
                self.geometries['landing_point'].translate(displacement)
                self.last_landing_position = np.array([10000, 10000, 10000])

            # 更新几何体
            if self.vis and self.window_created:
                self.vis.update_geometry(self.geometries['landing_point'])

        except Exception as e:
            print(f"❌ Error updating landing point: {e}")
            import traceback
            traceback.print_exc()

    # ... 保持其他方法不变，只修改上述关键方法 ...

    def toggle_window(self):
        """Improved window toggle with comprehensive state management"""
        if not self.window_visible:
            # Create and show window
            if self._create_window_safely():
                print(f"✅ 3D visualization opened at {time.strftime('%H:%M:%S')} UTC")
                self._print_usage_info()

                # Force immediate update to show existing data
                if any([self.all_valid_points_data, self.prediction_points_data,
                        self.predicted_trajectory_data]):
                    with self.geometry_update_lock:
                        self.needs_geometry_update = True
                        self._update_all_geometries()
            else:
                print("❌ Failed to create 3D visualization window")
        else:
            # Close window
            self.close_window()

    def close_window(self):
        """Properly close the window and reset all states"""
        try:
            print("🔄 Closing 3D visualization window...")
            self.window_should_close = True
            self.window_visible = False

            if self.vis and self.window_created:
                # Gracefully destroy window
                self.vis.destroy_window()
                time.sleep(0.1)  # Allow cleanup time

            self._cleanup_visualizer()
            print("✅ 3D visualization window closed successfully")

        except Exception as e:
            print(f"⚠️ Error during window closure (non-critical): {e}")
            self._cleanup_visualizer()

    def _cleanup_visualizer(self):
        """Clean up all visualizer resources and reset state"""
        try:
            # Reset all state variables
            self.vis = None
            self.view_control = None
            self.window_created = False
            self.window_visible = False
            self.view_initialized = False
            self.initialization_attempts = 0
            self.window_should_close = False
            self.creation_in_progress = False

            # Clear geometry references
            self.geometries.clear()

            # Reset update flags
            with self.geometry_update_lock:
                self.needs_geometry_update = False

        except Exception as e:
            print(f"Warning during cleanup: {e}")

    @property
    def window_visible(self):
        """Property to check if window is visible"""
        return self._window_visible

    @window_visible.setter
    def window_visible(self, value):
        """Setter for window visibility state"""
        self._window_visible = value

    def update_debug_data(self, debug_data):
        """Update debug data with thread safety and enhanced logging"""
        with self.data_lock:
            # All valid points (150 frames)
            if 'all_valid_points' in debug_data:
                self.all_valid_points_data = copy.deepcopy(debug_data['all_valid_points'])

            # Prediction trajectory points (8-15 selected points)
            if 'prediction_points' in debug_data:
                self.prediction_points_data = copy.deepcopy(debug_data['prediction_points'])

            # Rejected points (out of bounds)
            if 'rejected_points' in debug_data:
                self.rejected_points_data = copy.deepcopy(debug_data['rejected_points'])

            # Low quality points
            if 'low_quality_points' in debug_data:
                self.low_quality_points_data = copy.deepcopy(debug_data['low_quality_points'])

            # Triangulation failed points
            if 'triangulation_failed_points' in debug_data:
                self.triangulation_failed_data = copy.deepcopy(debug_data['triangulation_failed_points'])

            self.needs_geometry_update = True

            print(f"📊 Debug data updated at {time.strftime('%H:%M:%S')} UTC:")
            print(f"   ✅ All valid points: {len(self.all_valid_points_data)}")
            print(f"   🎯 Prediction points: {len(self.prediction_points_data)}")
            print(f"   ❌ Rejected points: {len(self.rejected_points_data)}")
            print(f"   ⚠️  Low quality points: {len(self.low_quality_points_data)}")
            print(f"   🔺 Triangulation failed: {len(self.triangulation_failed_data)}")

    def update_predicted_trajectory(self, trajectory_data):
        """Update predicted trajectory data with proper error handling"""
        with self.data_lock:
            if trajectory_data:
                self.predicted_trajectory_data = copy.deepcopy(trajectory_data)
                print(f"🎯 Predicted trajectory updated: {len(self.predicted_trajectory_data)} points")
            else:
                self.predicted_trajectory_data = []
                print("🎯 Predicted trajectory cleared")
            self.needs_geometry_update = True

    def update_if_visible(self):
        """Non-blocking update with comprehensive error handling"""
        if not self.window_visible or not self.window_created or self.window_should_close:
            return True

        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return True

        try:
            # Verify window is still valid
            if not self.vis:
                self.window_visible = False
                return True

            # Update geometries if needed
            with self.geometry_update_lock:
                if self.needs_geometry_update:
                    self._update_all_geometries()
                    self.needs_geometry_update = False

            # Poll events with error handling
            events_ok = self.vis.poll_events()
            if not events_ok:
                print("🔄 3D window closed by user")
                self.close_window()
                return True

            # Update renderer
            self.vis.update_renderer()
            self.last_update_time = current_time
            return True

        except Exception as e:
            error_msg = str(e).lower()
            if any(word in error_msg for word in ['destroyed', 'invalid', 'window', 'setviewpoint']):
                print(f"🔄 3D window error, closing: {e}")
                self.close_window()
            else:
                print(f"⚠️ Minor 3D visualization error: {e}")
            return True

    def _update_all_geometries(self):
        """Update all geometry objects with current data"""
        if not self.window_created or not self.vis or self.window_should_close:
            return

        try:
            with self.data_lock:
                # Update point clouds
                self._update_point_cloud('all_valid_points', self.all_valid_points_data,
                                         [0.5, 1, 0.5], self.visibility_flags['all_valid_points'])

                self._update_point_cloud('prediction_trajectory_points', self.prediction_points_data,
                                         [0, 0, 1], self.visibility_flags['prediction_points'])

                # Update special point types
                self._update_rejected_points()
                self._update_low_quality_points()
                self._update_triangulation_failed_points()

                # Update trajectory and landing point
                self._update_predicted_trajectory()
                self._update_landing_point()  # 使用修复后的方法

        except Exception as e:
            print(f"Error updating geometries: {e}")

    def _update_point_cloud(self, geometry_name, data, color, visible):
        """Generic method to update point cloud geometry"""
        if geometry_name not in self.geometries:
            return

        try:
            if data and visible:
                # Transform points to visualization coordinates
                if geometry_name in ['all_valid_points', 'prediction_trajectory_points']:
                    points_array = np.array([(p[0] + 100, p[1] + 100, p[2]) for p in data])
                else:
                    points_array = np.array(data)

                self.geometries[geometry_name].points = o3d.utility.Vector3dVector(points_array)
                colors = [color for _ in range(len(data))]
                self.geometries[geometry_name].colors = o3d.utility.Vector3dVector(colors)
            else:
                # Clear geometry when not visible or no data
                self.geometries[geometry_name].points = o3d.utility.Vector3dVector([])
                self.geometries[geometry_name].colors = o3d.utility.Vector3dVector([])

            self.vis.update_geometry(self.geometries[geometry_name])

        except Exception as e:
            print(f"Error updating {geometry_name}: {e}")

    def _update_rejected_points(self):
        """Update rejected points geometry (red points)"""
        try:
            if (self.rejected_points_data and
                    self.visibility_flags['rejected_points'] and
                    'rejected_points' in self.geometries):

                points_array = np.array([
                    (p['point_3d'][0] + 100, p['point_3d'][1] + 100, p['point_3d'][2])
                    for p in self.rejected_points_data
                ])

                self.geometries['rejected_points'].points = o3d.utility.Vector3dVector(points_array)
                colors = [[1, 0, 0] for _ in range(len(self.rejected_points_data))]  # Red
                self.geometries['rejected_points'].colors = o3d.utility.Vector3dVector(colors)
            else:
                self.geometries['rejected_points'].points = o3d.utility.Vector3dVector([])
                self.geometries['rejected_points'].colors = o3d.utility.Vector3dVector([])

            self.vis.update_geometry(self.geometries['rejected_points'])

        except Exception as e:
            print(f"Error updating rejected points: {e}")

    def _update_low_quality_points(self):
        """Update low quality points geometry (orange points)"""
        try:
            if (self.low_quality_points_data and
                    self.visibility_flags['low_quality_points'] and
                    'low_quality_points' in self.geometries):

                points_array = np.array([
                    (p['point_3d'][0] + 100, p['point_3d'][1] + 100, p['point_3d'][2])
                    for p in self.low_quality_points_data
                ])

                self.geometries['low_quality_points'].points = o3d.utility.Vector3dVector(points_array)
                colors = [[1, 0.5, 0] for _ in range(len(self.low_quality_points_data))]  # Orange
                self.geometries['low_quality_points'].colors = o3d.utility.Vector3dVector(colors)
            else:
                self.geometries['low_quality_points'].points = o3d.utility.Vector3dVector([])
                self.geometries['low_quality_points'].colors = o3d.utility.Vector3dVector([])

            self.vis.update_geometry(self.geometries['low_quality_points'])

        except Exception as e:
            print(f"Error updating low quality points: {e}")

    def _update_triangulation_failed_points(self):
        """Update triangulation failed points geometry (gray points)"""
        try:
            if (self.triangulation_failed_data and
                    self.visibility_flags['triangulation_failed'] and
                    'triangulation_failed_points' in self.geometries):

                points_array = []
                for p in self.triangulation_failed_data:
                    left_point = p['left_point']
                    # Simple 2D to 3D mapping for failed triangulation points
                    x = (left_point[0] - 640) * 0.5
                    y = (left_point[1] - 360) * 0.5
                    points_array.append([x + 100, y + 100, 10])

                if points_array:
                    self.geometries['triangulation_failed_points'].points = o3d.utility.Vector3dVector(
                        np.array(points_array)
                    )
                    colors = [[0.5, 0.5, 0.5] for _ in range(len(points_array))]  # Gray
                    self.geometries['triangulation_failed_points'].colors = o3d.utility.Vector3dVector(colors)
                else:
                    self.geometries['triangulation_failed_points'].points = o3d.utility.Vector3dVector([])
                    self.geometries['triangulation_failed_points'].colors = o3d.utility.Vector3dVector([])
            else:
                self.geometries['triangulation_failed_points'].points = o3d.utility.Vector3dVector([])
                self.geometries['triangulation_failed_points'].colors = o3d.utility.Vector3dVector([])

            self.vis.update_geometry(self.geometries['triangulation_failed_points'])

        except Exception as e:
            print(f"Error updating triangulation failed points: {e}")

    def _update_predicted_trajectory(self):
        """Update predicted trajectory line geometry (cyan to blue gradient)"""
        try:
            if (self.predicted_trajectory_data and
                    len(self.predicted_trajectory_data) > 1 and
                    self.visibility_flags['predicted_trajectory'] and
                    'predicted_trajectory_line' in self.geometries):

                trajectory_points = []
                for p in self.predicted_trajectory_data:
                    if isinstance(p, dict) and 'position' in p:
                        pos = p['position']
                    else:
                        pos = p

                    if len(pos) == 2:
                        trajectory_points.append([pos[0] + 100, pos[1] + 100, 0])
                    else:
                        trajectory_points.append([pos[0] + 100, pos[1] + 100, pos[2]])

                if len(trajectory_points) > 1:
                    points_array = np.array(trajectory_points)
                    lines = [[i, i + 1] for i in range(len(points_array) - 1)]

                    self.geometries['predicted_trajectory_line'].points = o3d.utility.Vector3dVector(points_array)
                    self.geometries['predicted_trajectory_line'].lines = o3d.utility.Vector2iVector(lines)

                    # Cyan to blue gradient
                    n_lines = len(lines)
                    colors = []
                    for i in range(n_lines):
                        ratio = i / max(1, n_lines - 1)
                        colors.append([0, 0.5 + ratio * 0.5, 1])  # Cyan to blue

                    self.geometries['predicted_trajectory_line'].colors = o3d.utility.Vector3dVector(colors)
                else:
                    self.geometries['predicted_trajectory_line'].points = o3d.utility.Vector3dVector([])
                    self.geometries['predicted_trajectory_line'].lines = o3d.utility.Vector2iVector([])
            else:
                self.geometries['predicted_trajectory_line'].points = o3d.utility.Vector3dVector([])
                self.geometries['predicted_trajectory_line'].lines = o3d.utility.Vector2iVector([])

            self.vis.update_geometry(self.geometries['predicted_trajectory_line'])

        except Exception as e:
            print(f"Error updating predicted trajectory: {e}")

    def toggle_visualization_elements(self, element_type):
        """Toggle display of specific visualization elements"""
        toggle_map = {
            'all_valid': 'all_valid_points',
            'prediction': 'prediction_points',
            'rejected': 'rejected_points',
            'low_quality': 'low_quality_points',
            'triangulation_failed': 'triangulation_failed',
            'predicted_trajectory': 'predicted_trajectory'
        }

        if element_type in toggle_map:
            flag_name = toggle_map[element_type]
            self.visibility_flags[flag_name] = not self.visibility_flags[flag_name]
            status = 'ON' if self.visibility_flags[flag_name] else 'OFF'

            display_names = {
                'all_valid': 'All valid points',
                'prediction': 'Prediction points',
                'rejected': 'Rejected points',
                'low_quality': 'Low quality points',
                'triangulation_failed': 'Triangulation failed points',
                'predicted_trajectory': 'Predicted trajectory'
            }

            print(f"📊 {display_names[element_type]}: {status}")
            self.needs_geometry_update = True
        else:
            print(f"⚠️ Unknown element type: {element_type}")

    def _print_usage_info(self):
        """Print comprehensive usage information with updated camera info"""
        print("   🎯 Enhanced 3D Debug Visualization Features:")
        print("   - Light green points: All valid points (150 frames)")
        print("   - Dark blue points: Prediction trajectory points (selected for prediction)")
        print("   - Cyan gradient line: Predicted trajectory path")
        print("   - Red points: Boundary-filtered rejected points")
        print("   - Orange points: Low quality assessment points")
        print("   - Gray points: Triangulation failed points")
        print("   - Landing point sphere: GREEN=In bounds, RED=Out of bounds, YELLOW=Unknown")
        print("   📋 Enhanced Interactive Controls:")
        print("   - Mouse left-click + drag: Rotate view around center")
        print("   - Mouse right-click + drag: Pan view")
        print("   - Mouse scroll wheel: Zoom in/out")
        print("   - Keys 1-6: Toggle different point types on/off")
        print("   - Press Q in main window to close this visualization")
        print("   🎥 Camera View: Optimized angle and distance for court visualization")

    def print_debug_statistics(self):
        """Print comprehensive debug statistics"""
        with self.data_lock:
            print("\n" + "=" * 80)
            print("🔍 3D VISUALIZATION DEBUG STATISTICS")
            print(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"User: {self._get_current_user()}")
            print("=" * 80)

            # Data counts
            print(f"✅ All valid points (150 frames): {len(self.all_valid_points_data)}")
            print(f"🎯 Prediction trajectory points: {len(self.prediction_points_data)}")
            print(f"🎯 Predicted trajectory line points: {len(self.predicted_trajectory_data)}")
            print(f"❌ Rejected (out-of-bounds) points: {len(self.rejected_points_data)}")
            print(f"⚠️  Low quality points: {len(self.low_quality_points_data)}")
            print(f"🔺 Triangulation failed points: {len(self.triangulation_failed_data)}")

            # Landing point status
            if self.landing_position is not None:
                status = "IN BOUNDS (GREEN)" if self.in_bounds else "OUT OF BOUNDS (RED)" if self.in_bounds is not None else "UNKNOWN (YELLOW)"
                print(f"🎯 Landing point: ({self.landing_position[0]:.1f}, {self.landing_position[1]:.1f}) - {status}")

            # Rejection analysis
            if self.rejected_points_data:
                print("\n📋 Rejection reasons breakdown:")
                reasons = {}
                for point in self.rejected_points_data:
                    reason = point.get('reason', 'unknown')
                    reasons[reason] = reasons.get(reason, 0) + 1
                for reason, count in reasons.items():
                    print(f"   {reason}: {count} points")

            # Quality analysis
            if self.low_quality_points_data:
                print("\n📋 Low quality reasons breakdown:")
                reasons = {}
                for point in self.low_quality_points_data:
                    reason = point.get('reason', 'unknown')
                    reasons[reason] = reasons.get(reason, 0) + 1
                for reason, count in reasons.items():
                    print(f"   {reason}: {count} points")

            # Trajectory analysis
            if self.prediction_points_data:
                points = np.array(self.prediction_points_data)
                print(f"\n📊 Prediction trajectory analysis:")
                print(f"   Points count: {len(points)}")
                if len(points) > 1:
                    distances = [np.linalg.norm(points[i] - points[i - 1]) for i in range(1, len(points))]
                    print(f"   Average inter-point distance: {np.mean(distances):.1f} cm")
                    print(f"   Height range: {np.max(points[:, 2]) - np.min(points[:, 2]):.1f} cm")
                    print(f"   Z coordinates range: {np.min(points[:, 2]):.1f} to {np.max(points[:, 2]):.1f} cm")

            # Visualization status
            print("\n📊 Visualization element status:")
            status_map = {
                'all_valid_points': '1 - All valid points',
                'prediction_points': '2 - Prediction points',
                'rejected_points': '3 - Rejected points',
                'low_quality_points': '4 - Low quality points',
                'triangulation_failed': '5 - Triangulation failed',
                'predicted_trajectory': '6 - Predicted trajectory'
            }

            for flag_name, description in status_map.items():
                status = 'ON' if self.visibility_flags.get(flag_name, True) else 'OFF'
                print(f"   {description}: {status}")

            # System status
            print(f"\n🔧 System status:")
            print(f"   Window created: {self.window_created}")
            print(f"   Window visible: {self.window_visible}")
            print(f"   View initialized: {self.view_initialized}")
            print(f"   Creation in progress: {self.creation_in_progress}")
            print("=" * 80)

    def reset(self):
        """Comprehensive reset of all visualization data and state"""
        with self.data_lock:
            self.reset_data()

            with self.geometry_update_lock:
                self.needs_geometry_update = True

            print(f"🔄 3D visualization data reset at {time.strftime('%H:%M:%S')} UTC")

            # If window is visible, update geometries to clear display
            if self.window_visible and self.window_created:
                self._update_all_geometries()

    def start(self):
        """Initialize the visualizer system"""
        print(f"🚀 Enhanced 3D Debug Visualizer ready at {time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("   Call toggle_window() to show 3D visualization")
        print("   Features: Fixed camera view, Correct landing point colors")
        print("   System ready for debug data visualization")

    def stop(self):
        """Stop visualizer and clean up all resources"""
        try:
            print(f"🛑 Stopping 3D visualizer at {time.strftime('%H:%M:%S')} UTC...")

            if self.vis and self.window_created:
                self.vis.destroy_window()
                time.sleep(0.2)  # Allow complete cleanup

        except Exception as e:
            print(f"⚠️ Error during visualizer shutdown: {e}")
        finally:
            self._cleanup_visualizer()
            print("✅ Enhanced 3D Debug Visualizer stopped and all resources cleaned up")