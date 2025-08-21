#!/usr/bin/env python3
"""
测试脚本
用于验证PyQt6羽毛球系统的基本功能
"""

import sys
import os
from pathlib import Path

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """测试模块导入"""
    print("Testing module imports...")
    
    try:
        import PyQt6.QtWidgets
        print("✓ PyQt6 available")
    except ImportError as e:
        print(f"✗ PyQt6 not available: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__} available")
    except ImportError as e:
        print(f"✗ OpenCV not available: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} available")
    except ImportError as e:
        print(f"✗ NumPy not available: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO available")
    except ImportError as e:
        print(f"✗ Ultralytics not available: {e}")
        return False
    
    try:
        import open3d as o3d
        print(f"✓ Open3D {o3d.__version__} available")
    except ImportError as e:
        print(f"✗ Open3D not available: {e}")
        return False
    
    try:
        import filterpy
        print("✓ FilterPy available")
    except ImportError as e:
        print(f"✗ FilterPy not available: {e}")
        return False
    
    return True


def test_core_modules():
    """测试核心模块"""
    print("\nTesting core modules...")
    
    try:
        from config import config
        print("✓ Config module imported")
    except ImportError as e:
        print(f"✗ Config module failed: {e}")
        return False
    
    try:
        from utils import logger, SystemUtils
        print("✓ Utils module imported")
    except ImportError as e:
        print(f"✗ Utils module failed: {e}")
        return False
    
    try:
        from calibration_core import CalibrationCore
        print("✓ Calibration core imported")
    except ImportError as e:
        print(f"✗ Calibration core failed: {e}")
        return False
    
    try:
        from detector_core import ShuttlecockDetector, StereoProcessor
        print("✓ Detector core imported")
    except ImportError as e:
        print(f"✗ Detector core failed: {e}")
        return False
    
    try:
        from predictor_core import TrajectoryPredictor
        print("✓ Predictor core imported")
    except ImportError as e:
        print(f"✗ Predictor core failed: {e}")
        return False
    
    return True


def test_ui_components():
    """测试UI组件"""
    print("\nTesting UI components...")
    
    try:
        from PyQt6.QtWidgets import QApplication
        app = QApplication([])
        
        try:
            from video_widget import DualVideoWidget
            print("✓ Video widget imported")
        except ImportError as e:
            print(f"✗ Video widget failed: {e}")
            return False
        
        try:
            from control_panel import ControlPanel
            print("✓ Control panel imported")
        except ImportError as e:
            print(f"✗ Control panel failed: {e}")
            return False
        
        try:
            from calibration_window import CalibrationWindow
            print("✓ Calibration window imported")
        except ImportError as e:
            print(f"✗ Calibration window failed: {e}")
            return False
        
        try:
            from visualization_3d import Visualization3DWidget
            print("✓ 3D visualization imported")
        except ImportError as e:
            print(f"✗ 3D visualization failed: {e}")
            return False
        
        try:
            from main_window import MainWindow
            print("✓ Main window imported")
        except ImportError as e:
            print(f"✗ Main window failed: {e}")
            return False
        
        app.quit()
        return True
        
    except Exception as e:
        print(f"✗ UI component test failed: {e}")
        return False


def test_worker_threads():
    """测试工作线程"""
    print("\nTesting worker threads...")
    
    try:
        from video_worker import VideoWorker, DualVideoWorker
        print("✓ Video workers imported")
    except ImportError as e:
        print(f"✗ Video workers failed: {e}")
        return False
    
    try:
        from detection_worker import DetectionWorker, StereoDetectionWorker
        print("✓ Detection workers imported")
    except ImportError as e:
        print(f"✗ Detection workers failed: {e}")
        return False
    
    try:
        from prediction_worker import PredictionWorker
        print("✓ Prediction worker imported")
    except ImportError as e:
        print(f"✗ Prediction worker failed: {e}")
        return False
    
    return True


def test_system_info():
    """测试系统信息"""
    print("\nTesting system information...")
    
    try:
        from utils import SystemUtils
        
        system_info = SystemUtils.get_system_info()
        print(f"✓ CPU cores: {system_info['cpu_count']}")
        print(f"✓ Memory: {system_info['memory_total'] / (1024**3):.1f} GB")
        print(f"✓ CPU usage: {system_info['cpu_percent']:.1f}%")
        
        dependencies = SystemUtils.check_dependencies()
        missing = [dep for dep, available in dependencies.items() if not available]
        
        if missing:
            print(f"⚠ Missing dependencies: {', '.join(missing)}")
        else:
            print("✓ All dependencies available")
        
        return True
        
    except Exception as e:
        print(f"✗ System info test failed: {e}")
        return False


def test_basic_functionality():
    """测试基本功能"""
    print("\nTesting basic functionality...")
    
    try:
        from PyQt6.QtWidgets import QApplication
        app = QApplication([])
        
        # 测试配置
        from config import config
        print(f"✓ Config loaded: {len(vars(config))} parameters")
        
        # 测试检测器创建
        try:
            from detector_core import create_detector, create_stereo_processor
            
            # 注意：这里不创建实际的检测器，因为需要模型文件
            print("✓ Detector factory functions available")
            
            stereo_processor = create_stereo_processor()
            print("✓ Stereo processor created")
            
        except Exception as e:
            print(f"⚠ Detector test limited: {e}")
        
        # 测试预测器
        try:
            from predictor_core import TrajectoryPredictor, CourtBoundaryAnalyzer
            
            predictor = TrajectoryPredictor()
            analyzer = CourtBoundaryAnalyzer()
            print("✓ Predictor and analyzer created")
            
            # 测试边界分析
            test_point = [300, 300, 0]
            result = analyzer.analyze_landing(test_point, 'singles')
            print(f"✓ Boundary analysis works: {result['in_bounds']}")
            
        except Exception as e:
            print(f"✗ Predictor test failed: {e}")
            return False
        
        app.quit()
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("Badminton PyQt6 System Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Dependencies", test_imports),
        ("Core Modules", test_core_modules),
        ("UI Components", test_ui_components),
        ("Worker Threads", test_worker_threads),
        ("System Information", test_system_info),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"Test Results: {passed}/{total} tests passed")
    print(f"{'=' * 60}")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    
    print(f"\nTo run the application:")
    print(f"  python main.py")
    print(f"\nFor help:")
    print(f"  python main.py --help")
    
    sys.exit(exit_code)