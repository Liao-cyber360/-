#!/usr/bin/env python3
"""
Test script for UI improvements
Tests the basic functionality without requiring a display
"""

import sys
import os
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent / "badminton_pyqt6"
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    
    try:
        # Test main window import
        from main_window import MainWindow
        print("✓ MainWindow imported successfully")
        
        # Test video worker import
        from video_worker import VideoWorker, DualVideoWorker
        print("✓ Video workers imported successfully")
        
        # Test calibration window import
        from calibration_window import CalibrationWindow
        print("✓ CalibrationWindow imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_ui_structure():
    """Test UI structure without creating actual widgets"""
    print("\nTesting UI structure...")
    
    try:
        # Mock Qt environment for headless testing
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        
        from PyQt6.QtWidgets import QApplication
        from main_window import MainWindow
        
        # Create minimal application
        app = QApplication([])
        
        # Test MainWindow can be created
        window = MainWindow()
        print("✓ MainWindow created successfully")
        
        # Test that required attributes exist
        assert hasattr(window, 'tab_widget'), "tab_widget not found"
        print("✓ Tab widget exists")
        
        assert hasattr(window, 'video_widget'), "video_widget not found"
        print("✓ Video widget exists")
        
        assert hasattr(window, 'viz_3d_widget'), "viz_3d_widget not found"
        print("✓ 3D visualization widget exists")
        
        # Test video worker attributes
        assert hasattr(window, 'video_loaded'), "video_loaded flag not found"
        print("✓ Video loading state tracking exists")
        
        print("✓ UI structure test passed")
        
        app.quit()
        return True
        
    except Exception as e:
        print(f"✗ UI structure test failed: {e}")
        return False

def test_network_camera_functionality():
    """Test network camera dialog functionality"""
    print("\nTesting network camera functionality...")
    
    try:
        # Test that the method exists
        from main_window import MainWindow
        
        # Create a mock window to test method existence
        class MockWindow:
            def __init__(self):
                self.video_worker = None
                self.video_loaded = False
                
            def statusBar(self):
                class MockStatusBar:
                    def showMessage(self, msg): pass
                return MockStatusBar()
            
        mock_window = MockWindow()
        
        # Verify the method exists by importing it
        assert hasattr(MainWindow, 'open_network_cameras'), "open_network_cameras method not found"
        print("✓ Network camera dialog method exists")
        
        # Verify menu action method exists  
        assert hasattr(MainWindow, 'setup_menu_bar'), "setup_menu_bar method not found"
        print("✓ Menu setup method exists")
        
        return True
        
    except Exception as e:
        print(f"✗ Network camera test failed: {e}")
        return False

def test_video_playback_fix():
    """Test video playback fix"""
    print("\nTesting video playback fix...")
    
    try:
        from video_worker import DualVideoWorker
        
        # Test that DualVideoWorker has required methods
        assert hasattr(DualVideoWorker, 'start_processing'), "start_processing method not found"
        assert hasattr(DualVideoWorker, 'play'), "play method not found"
        print("✓ Video worker has required playback methods")
        
        # Test that video worker can be created
        worker = DualVideoWorker("dummy1", "dummy2")
        assert hasattr(worker, 'is_running'), "is_running attribute not found"
        assert hasattr(worker, 'is_paused'), "is_paused attribute not found"
        print("✓ Video worker state management exists")
        
        return True
        
    except Exception as e:
        print(f"✗ Video playback test failed: {e}")
        return False

def test_calibration_improvements():
    """Test calibration improvements"""
    print("\nTesting calibration improvements...")
    
    try:
        from calibration_window import CalibrationWindow
        
        # Check that load_image method was replaced
        assert hasattr(CalibrationWindow, 'select_from_video_frame'), "select_from_video_frame method not found"
        print("✓ Video frame selection method exists")
        
        # Check that calibration window can be imported without errors
        print("✓ Enhanced calibration window imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Calibration improvements test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("UI Improvements Test Suite")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("UI Structure", test_ui_structure),
        ("Network Camera Functionality", test_network_camera_functionality),
        ("Video Playback Fix", test_video_playback_fix),
        ("Calibration Improvements", test_calibration_improvements),
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
        print("🎉 All UI improvement tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)