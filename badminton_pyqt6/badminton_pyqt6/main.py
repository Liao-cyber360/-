#!/usr/bin/env python3
"""
Badminton Landing Prediction System - PyQt6 Version
主程序入口点

使用方法:
    python main.py [options]

选项:
    --debug         启用调试模式
    --theme THEME   设置主题 (light/dark)
    --config FILE   指定配置文件
    --help          显示帮助信息
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 设置环境变量（必须在导入其他库之前）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 防止OpenMP冲突

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox, QSplashScreen
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont
except ImportError as e:
    print(f"Error: Failed to import PyQt6: {e}")
    print("Please install PyQt6: pip install PyQt6")
    sys.exit(1)

# 导入应用组件
try:
    from main_window import MainWindow
    from config import config
    from utils import logger, SystemUtils, DialogUtils
    from styles import MAIN_STYLE
except ImportError as e:
    print(f"Error: Failed to import application components: {e}")
    print("Please ensure all required files are present in the application directory")
    sys.exit(1)


class SplashScreen(QSplashScreen):
    """启动画面"""
    
    def __init__(self):
        # 创建启动画面图像
        pixmap = QPixmap(800, 400)
        pixmap.fill(QColor(33, 150, 243))  # Material Blue
        
        # 绘制内容
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 标题
        painter.setPen(QColor(255, 255, 255))
        title_font = QFont("Arial", 28, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.drawText(50, 100, "Badminton Landing Prediction System")
        
        # 副标题
        subtitle_font = QFont("Arial", 16)
        painter.setFont(subtitle_font)
        painter.drawText(50, 140, "PyQt6 Version - Advanced Trajectory Analysis")
        
        # 版本信息
        version_font = QFont("Arial", 12)
        painter.setFont(version_font)
        painter.drawText(50, 180, "Version 1.0.0 | Built with PyQt6, OpenCV, YOLO, Open3D")
        
        # 功能列表
        features = [
            "• Real-time shuttlecock detection and tracking",
            "• Stereo vision 3D trajectory reconstruction",
            "• Advanced aerodynamic trajectory prediction",
            "• Accurate landing point estimation",
            "• Automated in/out boundary judgment"
        ]
        
        feature_font = QFont("Arial", 11)
        painter.setFont(feature_font)
        
        y_pos = 230
        for feature in features:
            painter.drawText(70, y_pos, feature)
            y_pos += 25
        
        # 状态文本区域
        painter.drawText(50, 370, "Initializing...")
        
        painter.end()
        
        super().__init__(pixmap)
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.WindowStaysOnTopHint)
        
        # 状态消息
        self.status_message = "Initializing..."
    
    def show_message(self, message):
        """显示状态消息"""
        self.status_message = message
        self.showMessage(
            message,
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
            QColor(255, 255, 255)
        )
        QApplication.processEvents()


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Badminton Landing Prediction System - PyQt6 Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # 正常启动
  python main.py --debug            # 调试模式启动
  python main.py --theme dark       # 使用深色主题启动
  python main.py --config my.yaml   # 使用指定配置文件启动
        """
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式（显示详细日志）"
    )
    
    parser.add_argument(
        "--theme",
        choices=["light", "dark"],
        default="light",
        help="设置界面主题 (默认: light)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="指定配置文件路径"
    )
    
    parser.add_argument(
        "--no-splash",
        action="store_true",
        help="跳过启动画面"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="检查依赖项并退出"
    )
    
    return parser.parse_args()


def setup_logging(debug_mode=False):
    """设置日志记录"""
    level = logging.DEBUG if debug_mode else logging.INFO
    
    # 配置根日志记录器
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('badminton_system.log', encoding='utf-8')
        ]
    )
    
    # 设置第三方库日志级别
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def check_dependencies():
    """检查系统依赖项"""
    print("Checking system dependencies...")
    
    dependencies = SystemUtils.check_dependencies()
    
    missing_deps = []
    for dep, available in dependencies.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")
        
        if not available:
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Please install missing dependencies using:")
        print(f"  pip install {' '.join(missing_deps)}")
        return False
    
    print("\nAll dependencies are available.")
    return True


def check_system_requirements():
    """检查系统要求"""
    try:
        system_info = SystemUtils.get_system_info()
        
        # 检查内存
        memory_gb = system_info['memory_total'] / (1024**3)
        if memory_gb < 4:
            logger.warning(f"Low memory: {memory_gb:.1f} GB (recommended: 8+ GB)")
        
        # 检查CPU
        if system_info['cpu_count'] < 4:
            logger.warning(f"Low CPU cores: {system_info['cpu_count']} (recommended: 4+ cores)")
        
        # 检查磁盘空间
        if system_info['disk_usage'] > 90:
            logger.warning(f"Low disk space: {system_info['disk_usage']:.1f}% used")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to check system requirements: {e}")
        return False


def initialize_application(splash=None):
    """初始化应用程序"""
    try:
        if splash:
            splash.show_message("Checking system requirements...")
        
        # 检查系统要求
        if not check_system_requirements():
            logger.warning("System requirements check failed, but continuing...")
        
        if splash:
            splash.show_message("Loading configuration...")
        
        # 加载配置
        config_loaded = True
        try:
            # 配置已在导入时加载
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            config_loaded = False
        
        if splash:
            splash.show_message("Creating main window...")
        
        # 创建主窗口
        main_window = MainWindow()
        
        if splash:
            splash.show_message("Initializing components...")
        
        # 应用主题
        if hasattr(config, 'theme'):
            main_window.apply_theme(config.theme)
        
        logger.info("Application initialized successfully")
        
        return main_window, config_loaded
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        return None, False


def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 设置日志
        setup_logging(args.debug)
        
        logger.info("Starting Badminton Landing Prediction System")
        logger.info(f"Arguments: {vars(args)}")
        
        # 检查依赖项
        if args.check_deps:
            success = check_dependencies()
            sys.exit(0 if success else 1)
        
        if not check_dependencies():
            print("Cannot start application due to missing dependencies.")
            sys.exit(1)
        
        # 创建QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("Badminton Landing Prediction System")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("Badminton Analysis Lab")
        
        # 设置应用图标（如果有的话）
        # app.setWindowIcon(QIcon("icon.png"))
        
        # 显示启动画面
        splash = None
        if not args.no_splash:
            splash = SplashScreen()
            splash.show()
            app.processEvents()
        
        # 初始化应用程序
        main_window, config_loaded = initialize_application(splash)
        
        if main_window is None:
            if splash:
                splash.close()
            
            QMessageBox.critical(
                None, "Initialization Error",
                "Failed to initialize the application.\n"
                "Please check the log file for details."
            )
            sys.exit(1)
        
        # 应用命令行参数
        if args.theme:
            main_window.apply_theme(args.theme)
        
        if args.config:
            # 加载指定的配置文件
            logger.info(f"Loading configuration from: {args.config}")
            # 这里可以实现配置文件加载逻辑
        
        # 延迟显示主窗口
        def show_main_window():
            if splash:
                splash.finish(main_window)
            
            main_window.show()
            
            if not config_loaded:
                QMessageBox.warning(
                    main_window, "Configuration Warning",
                    "Some configuration settings could not be loaded.\n"
                    "The application will use default settings."
                )
            
            logger.info("Main window displayed")
        
        # 设置定时器延迟显示（让启动画面显示一段时间）
        if splash:
            QTimer.singleShot(2000, show_main_window)
        else:
            show_main_window()
        
        # 处理未捕获的异常
        def handle_exception(exc_type, exc_value, exc_traceback):
            """处理未捕获的异常"""
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            logger.error(
                "Uncaught exception",
                exc_info=(exc_type, exc_value, exc_traceback)
            )
            
            error_msg = f"An unexpected error occurred:\n{exc_type.__name__}: {exc_value}"
            
            try:
                QMessageBox.critical(None, "Unexpected Error", error_msg)
            except:
                print(f"Critical error: {error_msg}")
        
        sys.excepthook = handle_exception
        
        # 运行应用程序
        logger.info("Starting application event loop")
        exit_code = app.exec()
        
        logger.info(f"Application exiting with code: {exit_code}")
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        try:
            QMessageBox.critical(
                None, "Fatal Error",
                f"A fatal error occurred during startup:\n{e}\n\n"
                "Please check the log file for details."
            )
        except:
            print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    # 确保当前目录在Python路径中
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # 运行主程序
    exit_code = main()
    sys.exit(exit_code)