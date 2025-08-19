# Badminton Landing Prediction System - PyQt6 Version

一个基于PyQt6的现代化羽毛球落点预测系统，具有双目视觉、3D轨迹重建和物理预测功能。

## 特性

- **实时羽毛球检测与跟踪** - 使用YOLO深度学习模型
- **双目视觉3D轨迹重建** - 立体视觉处理和三角测量
- **高级空气动力学轨迹预测** - 基于物理模型的EKF预测
- **精确落点估计** - 多种预测方法集成
- **自动界内/界外判定** - 场地边界分析

## 技术栈

- **界面框架**: PyQt6 (Material Design风格)
- **计算机视觉**: OpenCV
- **深度学习**: Ultralytics YOLO
- **数值计算**: NumPy, SciPy
- **滤波算法**: FilterPy (扩展卡尔曼滤波)
- **3D可视化**: Open3D
- **多线程**: PyQt6 QThread

## 系统要求

### 硬件要求
- CPU: 4核心以上推荐
- 内存: 8GB以上推荐
- 显卡: 支持OpenGL的独立显卡（用于3D可视化）

### 软件要求
- Python 3.8+
- 操作系统: Windows 10+, macOS 10.14+, Ubuntu 18.04+

## 安装

1. 克隆仓库:
```bash
git clone <repository-url>
cd badminton_pyqt6
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 检查依赖项:
```bash
python main.py --check-deps
```

## 使用方法

### 基本启动
```bash
python main.py
```

### 命令行选项
```bash
# 调试模式
python main.py --debug

# 深色主题
python main.py --theme dark

# 跳过启动画面
python main.py --no-splash

# 使用指定配置文件
python main.py --config my_config.yaml
```

## 主要功能

### 1. 相机标定
- 打开"Calibration" -> "Camera Calibration..."
- 加载标定图像
- 手动选择场地角点
- 自动计算相机外参

### 2. 视频分析
- 打开"File" -> "Open Video Files..."
- 选择双路视频文件
- 点击"开始处理"进行实时分析

### 3. 轨迹预测
- 确保有足够的轨迹数据
- 点击"预测"按钮或按空格键
- 查看3D可视化和场地俯视图结果

### 4. 参数调整
- 使用右侧控制面板调整检测和预测参数
- 实时更新处理结果

## 项目结构

```
badminton_pyqt6/
├── main.py                    # 程序入口
├── main_window.py             # 主窗口界面
├── calibration_window.py      # 标定窗口
├── video_widget.py            # 视频显示组件
├── control_panel.py           # 控制面板
├── visualization_3d.py        # 3D可视化
├── calibration_core.py        # 标定算法核心
├── detector_core.py           # 检测算法核心
├── predictor_core.py          # 预测算法核心
├── video_worker.py            # 视频处理线程
├── detection_worker.py        # 检测处理线程
├── prediction_worker.py       # 预测处理线程
├── config.py                  # 配置管理
├── utils.py                   # 工具函数
├── styles.qss                 # 界面样式
└── requirements.txt           # 依赖包
```

## 配置

系统配置通过以下方式管理:
- `config.py`: 默认配置参数
- `config.yaml`: 用户自定义配置文件
- PyQt6 QSettings: 界面设置持久化

### 主要配置项

```yaml
camera:
  params_file_1: "path/to/camera1_params.yaml"
  params_file_2: "path/to/camera2_params.yaml"

models:
  yolo_ball: "path/to/ball_detection_model.pt"
  yolo_court: "path/to/court_detection_model.pt"

physics:
  mass: 0.005          # 羽毛球质量 (kg)
  drag_coefficient: 0.6 # 阻力系数
  air_density: 1.225    # 空气密度 (kg/m³)

ui:
  theme: "light"        # 界面主题
```

## 开发指南

### 添加新功能
1. 在相应的核心模块中实现算法
2. 创建对应的工作线程
3. 在主窗口中集成界面
4. 更新配置和文档

### 调试技巧
- 使用 `--debug` 参数启用详细日志
- 查看 `badminton_system.log` 文件
- 使用控制面板的调试信息标签页

## 故障排除

### 常见问题

**1. 依赖项缺失**
```bash
pip install <missing-package>
```

**2. YOLO模型加载失败**
- 检查模型文件路径
- 确认模型文件完整性
- 使用CPU模式测试

**3. 3D可视化问题**
- 更新显卡驱动
- 检查OpenGL支持
- 尝试软件渲染模式

**4. 相机标定失败**
- 确保标定图像清晰
- 检查场地角点选择精度
- 验证相机内参文件

### 性能优化

- 使用GPU加速YOLO推理
- 调整视频分辨率和帧率
- 优化轨迹缓冲区大小
- 启用多线程处理

## 贡献

欢迎提交问题报告和功能请求。开发贡献请遵循以下流程:

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 许可证

此项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

- 项目维护者: [Your Name]
- 邮箱: [your.email@example.com]
- 项目主页: [Project URL]

## 更新日志

### v1.0.0 (2024-XX-XX)
- 初始PyQt6版本发布
- 完整的GUI界面实现
- 多线程架构
- 3D可视化集成
- Material Design样式