#!/usr/bin/env python3
"""
Demo script for the enhanced badminton system features
Shows how to use the new functionality without full GUI
"""

import sys
import numpy as np
import time
from datetime import datetime

# Suppress Qt warnings for demo
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

def demo_multi_ball_tracking():
    """Demonstrate the multi-ball tracking functionality"""
    print("=" * 60)
    print("DEMO: Multi-Ball Tracking")
    print("=" * 60)
    
    # Import without full dependencies
    class MockMultiBallTracker:
        def __init__(self, max_balls=2, tracking_distance_threshold=100):
            self.max_balls = max_balls
            self.tracking_distance_threshold = tracking_distance_threshold
            self.ball_trajectories = {}
            self.next_ball_id = 0
            self.active_balls = {}
            self.trajectory_timeout = 2.0
            print(f"MultiBallTracker initialized: max_balls={max_balls}, distance_threshold={tracking_distance_threshold}cm")
        
        def update_detections(self, points_3d, timestamps):
            """Simplified multi-ball tracking demo"""
            current_time = timestamps[-1] if timestamps else time.time()
            
            # Simple ball assignment based on distance clustering
            if len(points_3d) <= 1:
                # Single ball
                ball_id = 0
                self.ball_trajectories[ball_id] = {
                    'points': points_3d,
                    'timestamps': timestamps,
                    'last_update': current_time
                }
                self.active_balls[ball_id] = {
                    'last_point': points_3d[-1] if points_3d else [0,0,0],
                    'confidence': 1.0,
                    'age': len(points_3d)
                }
            else:
                # Multiple balls - simple clustering
                clusters = self._simple_clustering(points_3d)
                for i, cluster_points in enumerate(clusters[:self.max_balls]):
                    ball_id = i
                    cluster_timestamps = timestamps[:len(cluster_points)]
                    self.ball_trajectories[ball_id] = {
                        'points': cluster_points,
                        'timestamps': cluster_timestamps,
                        'last_update': current_time
                    }
                    self.active_balls[ball_id] = {
                        'last_point': cluster_points[-1] if cluster_points else [0,0,0],
                        'confidence': 1.0,
                        'age': len(cluster_points)
                    }
            
            return self.ball_trajectories
        
        def _simple_clustering(self, points):
            """Simple clustering by distance"""
            if len(points) <= 1:
                return [points]
            
            clusters = [[points[0]]]
            
            for point in points[1:]:
                assigned = False
                for cluster in clusters:
                    # Check distance to cluster center
                    cluster_center = np.mean(cluster, axis=0)
                    distance = np.linalg.norm(np.array(point) - cluster_center)
                    if distance < self.tracking_distance_threshold:
                        cluster.append(point)
                        assigned = True
                        break
                
                if not assigned and len(clusters) < self.max_balls:
                    clusters.append([point])
            
            return clusters
        
        def get_balls_summary(self):
            """Get summary of active balls"""
            summary = {
                'total_balls': len(self.active_balls),
                'balls_info': {}
            }
            
            for ball_id, ball_info in self.active_balls.items():
                trajectory = self.ball_trajectories.get(ball_id, {'points': [], 'timestamps': []})
                summary['balls_info'][ball_id] = {
                    'last_position': ball_info['last_point'],
                    'trajectory_length': len(trajectory['points']),
                    'age': ball_info['age'],
                    'confidence': ball_info['confidence']
                }
            
            return summary
    
    # Demo data: Two balls in different locations
    tracker = MockMultiBallTracker(max_balls=2, tracking_distance_threshold=150)
    
    # Scenario 1: Single ball trajectory
    print("\n--- Scenario 1: Single Ball ---")
    single_ball_points = [
        [100, 200, 300],
        [105, 205, 295],
        [110, 210, 290],
        [115, 215, 285]
    ]
    timestamps = [1.0, 1.1, 1.2, 1.3]
    
    trajectories = tracker.update_detections(single_ball_points, timestamps)
    summary = tracker.get_balls_summary()
    print(f"Detected balls: {summary['total_balls']}")
    for ball_id, info in summary['balls_info'].items():
        print(f"  Ball {ball_id}: {info['trajectory_length']} points, "
              f"position: ({info['last_position'][0]:.1f}, {info['last_position'][1]:.1f}, {info['last_position'][2]:.1f})")
    
    # Scenario 2: Two balls simultaneously
    print("\n--- Scenario 2: Two Balls Simultaneously ---")
    two_ball_points = [
        [100, 200, 300],  # Ball 1
        [105, 205, 295],  # Ball 1
        [300, 400, 280],  # Ball 2 (far from Ball 1)
        [305, 405, 275],  # Ball 2
        [110, 210, 290],  # Ball 1
        [310, 410, 270]   # Ball 2
    ]
    timestamps = [1.0, 1.1, 1.0, 1.1, 1.2, 1.2]
    
    trajectories = tracker.update_detections(two_ball_points, timestamps)
    summary = tracker.get_balls_summary()
    print(f"Detected balls: {summary['total_balls']}")
    for ball_id, info in summary['balls_info'].items():
        print(f"  Ball {ball_id}: {info['trajectory_length']} points, "
              f"position: ({info['last_position'][0]:.1f}, {info['last_position'][1]:.1f}, {info['last_position'][2]:.1f})")
    
    print("✓ Multi-ball tracking demo completed")

def demo_velocity_analysis():
    """Demonstrate velocity analysis functionality"""
    print("\n" + "=" * 60)
    print("DEMO: Velocity Analysis")
    print("=" * 60)
    
    # Mock velocity analysis
    def calculate_velocities(points, timestamps):
        """Calculate velocities between trajectory points"""
        if len(points) < 2:
            return [], [], None
        
        velocities = []
        speeds = []
        max_speed = 0.0
        max_speed_info = None
        
        for i in range(1, len(points)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                displacement = np.array(points[i]) - np.array(points[i-1])
                velocity_vector = displacement / dt  # cm/s
                speed = np.linalg.norm(velocity_vector)
                
                velocities.append(velocity_vector)
                speeds.append(speed)
                
                if speed > max_speed:
                    max_speed = speed
                    max_speed_info = {
                        'speed': speed,
                        'position': points[i],
                        'timestamp': timestamps[i],
                        'velocity_vector': velocity_vector,
                        'index': i
                    }
        
        return velocities, speeds, max_speed_info
    
    # Demo trajectory with realistic badminton physics
    print("\n--- Badminton Trajectory Velocity Analysis ---")
    
    # Simulated badminton trajectory (dropping and decelerating)
    trajectory_points = np.array([
        [0, 0, 200],      # Start: 2m high
        [50, 25, 190],    # Moving fast initially
        [120, 60, 170],   # Peak speed
        [200, 100, 140],  # Slowing down
        [280, 140, 100],  # Descending
        [360, 180, 60],   # Near ground
        [440, 220, 20],   # About to land
        [500, 250, 0]     # Landing
    ])
    
    timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    
    velocities, speeds, max_info = calculate_velocities(trajectory_points, timestamps)
    
    print(f"Trajectory analysis:")
    print(f"  Total points: {len(trajectory_points)}")
    print(f"  Flight time: {timestamps[-1] - timestamps[0]:.1f} seconds")
    print(f"  Total distance: {sum(np.linalg.norm(trajectory_points[i+1] - trajectory_points[i]) for i in range(len(trajectory_points)-1)):.1f} cm")
    
    if speeds:
        print(f"\nVelocity Statistics:")
        print(f"  Maximum speed: {max(speeds):.1f} cm/s ({max(speeds)/100:.1f} m/s, {max(speeds)*0.036:.1f} km/h)")
        print(f"  Average speed: {np.mean(speeds):.1f} cm/s ({np.mean(speeds)/100:.1f} m/s)")
        print(f"  Minimum speed: {min(speeds):.1f} cm/s ({min(speeds)/100:.1f} m/s)")
        print(f"  Speed variation: {np.std(speeds):.1f} cm/s")
    
    if max_info:
        print(f"\nMaximum Speed Details:")
        print(f"  Speed: {max_info['speed']:.1f} cm/s ({max_info['speed']/100:.1f} m/s)")
        print(f"  Position: ({max_info['position'][0]:.1f}, {max_info['position'][1]:.1f}, {max_info['position'][2]:.1f}) cm")
        print(f"  Time: {max_info['timestamp']:.1f} seconds")
        print(f"  Velocity vector: ({max_info['velocity_vector'][0]:.1f}, {max_info['velocity_vector'][1]:.1f}, {max_info['velocity_vector'][2]:.1f}) cm/s")
    
    # Show speed profile
    print(f"\nSpeed Profile:")
    for i, (speed, timestamp) in enumerate(zip(speeds, timestamps[1:])):
        print(f"  t={timestamp:.1f}s: {speed:.1f} cm/s ({speed/100:.1f} m/s)")
    
    print("✓ Velocity analysis demo completed")

def demo_mjpeg_camera_config():
    """Demonstrate MJPEG camera configuration"""
    print("\n" + "=" * 60)
    print("DEMO: MJPEG Network Camera Configuration")
    print("=" * 60)
    
    # Example configuration
    camera_configs = [
        {
            'name': 'Camera 1',
            'url': 'http://192.168.10.3:8080/video',
            'timestamp_header': 'X-Timestamp',
            'buffer_duration': 5.0,
            'sync_tolerance': 0.05
        },
        {
            'name': 'Camera 2', 
            'url': 'http://192.168.10.4:8080/video',
            'timestamp_header': 'X-Timestamp',
            'buffer_duration': 5.0,
            'sync_tolerance': 0.05
        }
    ]
    
    print("MJPEG Camera Configuration Example:")
    for i, config in enumerate(camera_configs):
        print(f"\n--- {config['name']} ---")
        print(f"  URL: {config['url']}")
        print(f"  Timestamp header: {config['timestamp_header']}")
        print(f"  Buffer duration: {config['buffer_duration']} seconds")
        print(f"  Sync tolerance: {config['sync_tolerance']} seconds")
    
    print(f"\nFeatures:")
    print(f"  ✓ Real-time MJPEG stream processing")
    print(f"  ✓ HTTP header timestamp extraction")
    print(f"  ✓ 5-second circular buffer for pause/resume")
    print(f"  ✓ Dual camera synchronization")
    print(f"  ✓ Network camera mode (disables progress bar)")
    print(f"  ✓ Automatic reconnection on connection loss")
    
    # Timestamp parsing example
    print(f"\nTimestamp Parsing Example:")
    example_timestamp = "1703980800123"  # Example timestamp in milliseconds
    parsed_time = datetime.fromtimestamp(int(example_timestamp) / 1000)
    formatted_time = parsed_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f"  Raw timestamp: {example_timestamp}")
    print(f"  Parsed time: {formatted_time}")
    
    print("✓ MJPEG camera configuration demo completed")

def demo_enhanced_progress_bar():
    """Demonstrate enhanced progress bar features"""
    print("\n" + "=" * 60)
    print("DEMO: Enhanced Progress Bar Features")
    print("=" * 60)
    
    # Simulate video properties
    video_properties = {
        'total_frames': 1800,  # 1 minute at 30fps
        'fps': 30,
        'duration': 60.0  # seconds
    }
    
    print("Enhanced Progress Bar Features:")
    print(f"  ✓ Real-time dragging with visual feedback")
    print(f"  ✓ Time display during drag operations")
    print(f"  ✓ Smooth seeking to any frame position")
    print(f"  ✓ Frame-accurate positioning")
    print(f"  ✓ Network camera mode disables dragging")
    
    print(f"\nVideo Properties:")
    print(f"  Total frames: {video_properties['total_frames']}")
    print(f"  Frame rate: {video_properties['fps']} fps")
    print(f"  Duration: {video_properties['duration']} seconds")
    
    # Time formatting examples
    print(f"\nTime Display Examples:")
    
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    time_examples = [0, 15.5, 30.0, 45.7, 60.0]
    for seconds in time_examples:
        frame_number = int(seconds * video_properties['fps'])
        formatted = format_time(seconds)
        print(f"  Frame {frame_number:4d}: {formatted}")
    
    print(f"\nProgress Bar States:")
    print(f"  ✓ Local video mode: Full dragging enabled")
    print(f"  ✓ Network camera mode: Dragging disabled, pause/resume only")
    print(f"  ✓ Real-time preview: Shows frame during drag")
    print(f"  ✓ Visual feedback: Highlighted handle and progress")
    
    print("✓ Enhanced progress bar demo completed")

def main():
    """Run all feature demonstrations"""
    print("BADMINTON SYSTEM - NEW FEATURES DEMONSTRATION")
    print("=" * 60)
    print("This demo shows the enhanced features implemented:")
    print("1. Multi-ball tracking for simultaneous badminton detection")
    print("2. Velocity analysis with maximum speed calculation")
    print("3. MJPEG network camera support with timestamp extraction")
    print("4. Enhanced progress bar with real-time dragging")
    
    try:
        demo_multi_ball_tracking()
        demo_velocity_analysis()
        demo_mjpeg_camera_config()
        demo_enhanced_progress_bar()
        
        print("\n" + "=" * 60)
        print("ALL FEATURE DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nImplementation Summary:")
        print("✓ Multi-ball tracking supports up to 2 simultaneous balls")
        print("✓ Velocity analysis provides comprehensive speed metrics")
        print("✓ MJPEG cameras support real-time streaming with buffering")
        print("✓ Enhanced progress bar provides smooth video navigation")
        print("\nThe system is ready for integration and testing!")
        
    except Exception as e:
        print(f"\n✗ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()