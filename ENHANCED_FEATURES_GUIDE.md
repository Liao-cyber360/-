# Enhanced Badminton System Features - Implementation Guide

This document describes the four major features that have been implemented to enhance the badminton landing prediction system.

## 1. Enhanced Draggable Progress Bar for Video Playback

### Features Implemented
- **Real-time dragging**: Smooth progress bar that responds to user dragging
- **Visual feedback**: Time display label appears during dragging
- **Frame-accurate seeking**: Precise navigation to any frame
- **Network camera mode**: Automatically disables dragging for live streams
- **Enhanced styling**: Improved visual appearance with hover effects

### Key Components
- `DualVideoWidget` class in `badminton_pyqt6/video_widget.py`
- Added `time_display_label` for showing time during drag
- Enhanced signal system with `seek_preview` for real-time feedback
- Network camera mode detection and UI adaptation

### Usage Example
```python
from badminton_pyqt6.video_widget import DualVideoWidget

# Create enhanced video widget
video_widget = DualVideoWidget()

# Set video properties
video_widget.set_video_info(total_frames=1800, fps=30.0)

# Enable/disable network camera mode
video_widget.set_network_camera_mode(is_network=True)  # Disables dragging
video_widget.set_network_camera_mode(is_network=False) # Enables dragging

# Connect to seek signals
video_widget.seek_requested.connect(handle_seek)      # Final seek position
video_widget.seek_preview.connect(handle_preview)     # Real-time preview
```

### Configuration
- **Local Video Mode**: Full dragging enabled with preview
- **Network Camera Mode**: Dragging disabled, only pause/resume available
- **Time Display**: Shows current/total time in MM:SS or HH:MM:SS format

## 2. Multiple Simultaneous Badminton Ball Handling

### Features Implemented
- **Multi-ball tracking**: Supports up to 2 simultaneous badminton balls
- **Distance-based clustering**: Automatically assigns detections to different balls
- **Trajectory management**: Maintains separate trajectory for each ball
- **Ball selection**: Ability to predict landing for specific ball or auto-select best
- **Status reporting**: Real-time information about active balls

### Key Components
- `MultiBallTracker` class in `detector.py`
- Enhanced `StereoProcessor` with multi-ball support
- Distance-based ball assignment algorithm
- Trajectory timeout and cleanup system

### Usage Example
```python
from detector import MultiBallTracker, StereoProcessor

# Create multi-ball tracker
tracker = MultiBallTracker(max_balls=2, tracking_distance_threshold=100)

# Update with new detections
trajectories = tracker.update_detections(points_3d, timestamps)

# Get active ball status
summary = tracker.get_balls_summary()
print(f"Active balls: {summary['total_balls']}")

# Enhanced stereo processor with multi-ball support
stereo_processor = StereoProcessor()
best_points, best_timestamps, confidence, ball_id = stereo_processor.find_best_trajectory_for_prediction(
    current_time, ball_id=None  # Auto-select or specify ball_id
)
```

### Configuration
- **Max Balls**: Default 2, configurable
- **Distance Threshold**: 100cm default for ball separation
- **Trajectory Timeout**: 2 seconds without updates removes ball
- **Ball Selection**: Automatic (most active) or manual by ball_id

## 3. Maximum Ball Speed Calculation Before Landing

### Features Implemented
- **Velocity calculation**: Comprehensive speed analysis throughout trajectory
- **Maximum velocity tracking**: Records peak speed and position
- **Speed profile**: Complete velocity history with timestamps
- **Unit conversions**: Results in cm/s, m/s, and km/h
- **Statistical analysis**: Max, min, average, and standard deviation

### Key Components
- Enhanced `TrajectoryPredictor` class in `predictor.py`
- `calculate_velocities()` method for detailed analysis
- `get_velocity_analysis()` for comprehensive statistics
- `get_max_velocity_before_landing()` for peak speed info

### Usage Example
```python
from predictor import TrajectoryPredictor

predictor = TrajectoryPredictor()

# Predict with velocity analysis
landing_pos, landing_time, trajectory = predictor.predict_landing_point(points, timestamps)

# Get detailed velocity analysis
analysis = predictor.get_velocity_analysis(points, timestamps)
print(f"Max speed: {analysis['max_speed_ms']:.1f} m/s")
print(f"Average speed: {analysis['avg_speed_ms']:.1f} m/s")

# Get maximum velocity info
max_vel_info = predictor.get_max_velocity_before_landing()
print(f"Peak velocity: {max_vel_info['max_velocity_kmh']:.1f} km/h")
print(f"At position: {max_vel_info['max_velocity_position']}")
```

### Output Information
- **Speed Statistics**: Max, min, average, standard deviation
- **Velocity Vectors**: 3D velocity components at each point
- **Position Tracking**: Where maximum speed occurred
- **Time Information**: When maximum speed was reached
- **Unit Conversions**: Automatic conversion to m/s and km/h

## 4. MJPEG Network Camera Support with Timestamp Extraction

### Features Implemented
- **MJPEG stream processing**: Real-time network camera support
- **HTTP header timestamps**: Configurable timestamp extraction
- **5-second buffering**: Circular buffer for pause/resume functionality
- **Dual camera sync**: Synchronized capture from two network cameras
- **Frame synchronization**: Timestamp-based frame matching
- **Automatic reconnection**: Handles network disconnections

### Key Components
- `MJPEGCameraWorker` class in `badminton_pyqt6/video_worker.py`
- `DualMJPEGWorker` for synchronized dual cameras
- HTTP header timestamp parsing
- Frame buffering and synchronization system

### Usage Example
```python
from badminton_pyqt6.video_worker import MJPEGCameraWorker, DualMJPEGWorker

# Single MJPEG camera
camera_worker = MJPEGCameraWorker(
    camera_id=0,
    camera_url="http://192.168.10.3:8080/video",
    timestamp_header="X-Timestamp"
)

# Connect signals
camera_worker.frame_ready.connect(handle_frame)
camera_worker.error_occurred.connect(handle_error)

# Start streaming
camera_worker.start()

# Pause for buffered processing
camera_worker.pause()
buffered_frames = camera_worker.get_buffered_frames()

# Dual camera setup
dual_worker = DualMJPEGWorker(
    camera1_url="http://192.168.10.3:8080/video",
    camera2_url="http://192.168.10.4:8080/video",
    timestamp_header="X-Timestamp"
)

dual_worker.dual_frame_ready.connect(handle_dual_frames)
dual_worker.start_processing()
```

### Configuration
- **Camera URLs**: Standard MJPEG stream URLs
- **Timestamp Header**: Configurable HTTP header field (default: "X-Timestamp")
- **Buffer Duration**: 5 seconds (150 frames at 30fps)
- **Sync Tolerance**: 50ms default for frame synchronization
- **Reconnection**: Automatic retry on connection loss

### Example Configuration
```python
CAMERA_URL = "http://192.168.10.3:8080/video"
TIMESTAMP_HEADER = "X-Timestamp"  # Confirmed timestamp field

# Timestamp parsing example
def update(self, headers):
    """Update frame statistics"""
    self.frame_count += 1
    
    # Calculate real-time FPS
    current_time = datetime.now()
    if (current_time - self.last_time).total_seconds() >= 1:
        self.fps = self.frame_count
        self.frame_count = 0
        self.last_time = current_time
    
    # Parse timestamp
    ts_str = headers.get(TIMESTAMP_HEADER, "")
    if ts_str and ts_str.isdigit():
        ts_ms = int(ts_str)
        self.last_timestamp = datetime.fromtimestamp(ts_ms / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
```

## Integration Notes

### Network Camera Mode Behavior
When network cameras are detected:
1. Progress bar dragging is automatically disabled
2. Only pause/resume controls remain active
3. 5-second buffer allows for trajectory analysis after pausing
4. System switches to real-time processing mode

### Multi-Ball Prediction Priority
1. If `ball_id` is specified, predict for that specific ball
2. If no `ball_id`, automatically select the most active trajectory
3. Fall back to combined trajectory if multi-ball tracking fails
4. Report which ball the prediction refers to

### Performance Considerations
- **Velocity calculations**: O(n) complexity, minimal overhead
- **Multi-ball tracking**: Distance calculations scale with number of detections
- **MJPEG buffering**: Memory usage limited by buffer size (150 frames)
- **Frame synchronization**: Timestamp comparison with configurable tolerance

### Error Handling
- **Network disconnections**: Automatic reconnection attempts
- **Invalid timestamps**: Fallback to system time
- **Ball tracking timeout**: Automatic cleanup of stale trajectories
- **Buffer overflow**: Circular buffer prevents memory issues

## Testing

Run the demonstration scripts to verify functionality:

```bash
# Test core features without GUI
python demo_new_features.py

# Test PyQt6 components (if display available)
python test_qt_features.py
```

Both scripts provide comprehensive testing of all implemented features and serve as usage examples.

## Dependencies

### Required Packages
- `opencv-python>=4.8.0`: Video processing and MJPEG decoding
- `numpy>=1.24.0`: Numerical computations for velocity analysis
- `PyQt6>=6.6.0`: Enhanced video widget functionality
- `requests>=2.31.0`: MJPEG stream HTTP processing

### Optional Features
- Display environment required for full PyQt6 GUI testing
- Network cameras with MJPEG support for streaming features
- Calibrated stereo camera setup for multi-ball tracking

The implementations are complete and production-ready for integration into the existing badminton prediction system.