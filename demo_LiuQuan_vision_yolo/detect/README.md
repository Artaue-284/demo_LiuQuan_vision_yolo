# RoboMaster 装甲板检测 & 跟踪 & 位姿估计 (C++ 实现)

RoboMaster 装甲板实时检测、多目标跟踪与 6-DOF 位姿估计系统。使用 YOLO 深度学习模型进行装甲板检测，扩展卡尔曼滤波器 (EKF) 进行多目标跟踪，PnP 算法进行三维位姿估计。

## 功能

| 模块 | 说明 |
|------|------|
| **YOLO ONNX 检测** | 使用 OpenCV DNN 加载 ONNX 模型，支持 6 分类装甲板检测 |
| **EKF 多目标跟踪** | 8 维状态向量 (位置+尺寸+速度)，恒定速度模型，贪心 IoU 数据关联 |
| **PnP 位姿估计** | `solvePnP(IPPE_SQUARE)` 估计 6-DOF 位姿和距离 |
| **FPS 测量** | 实时帧率显示和平均 FPS 统计 |
| **结果保存** | 检测可视化图像/视频自动保存到 `results/` 目录 |

### 装甲板分类 (6 类)

| ID | 名称 | 颜色 | 机器人类型 | 装甲板尺寸 |
|----|------|------|-----------|-----------|
| 0 | red3 | 红 | 红方步兵 (3号) | 230×125mm |
| 1 | red1 | 红 | 红方英雄 (1号) | 340×125mm |
| 2 | redsb | 红 | 红方哨兵 | 200×100mm |
| 3 | blue3 | 蓝 | 蓝方步兵 (3号) | 230×125mm |
| 4 | blue1 | 蓝 | 蓝方英雄 (1号) | 340×125mm |
| 5 | bluesb | 蓝 | 蓝方哨兵 | 200×100mm |

## 项目结构

```
demo_LiuQuan_vision_yolo/
├── yolo_train.py              # YOLO 训练脚本
├── yolo_convert.py            # ONNX 导出脚本
├── models/
│   └── best.onnx              # 训练好的 YOLO ONNX 模型
├── assets/                    # 测试图像/视频
├── results/                   # 输出结果
└── detect/                    # C++ 推理工程
    ├── CMakeLists.txt
    ├── README.md
    ├── include/
    │   ├── Detector.h         # YOLO 检测器接口
    │   ├── KalmanFilter.h     # EKF 单目标跟踪器接口
    │   ├── Tracker.h          # 多目标跟踪管理接口
    │   └── PnPSolver.h        # PnP 位姿估计接口
    └── src/
        ├── main.cpp           # 主程序入口
        ├── Detector.cpp       # 检测器实现
        ├── KalmanFilter.cpp   # EKF 实现
        ├── Tracker.cpp        # 多目标跟踪实现
        └── PnPSolver.cpp      # PnP 实现
```

## 依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| CMake | ≥ 3.16 | 构建系统 |
| C++ 编译器 | C++17 | MSVC 2019+ / GCC 9+ / Clang 9+ |
| OpenCV | ≥ 4.5 | 图像处理、DNN 推理、PnP、GUI |
| ONNX Runtime | (可选) | 替代 OpenCV DNN 进行 ONNX 推理 |

## 构建

### Windows (Visual Studio)

```powershell
# 1) 安装 OpenCV (推荐 vcpkg 或官方预编译包)
# vcpkg install opencv[core,dnn,imgproc,highgui,calib3d, video, videoio]

# 2) 配置 CMake
cd detect
mkdir build && cd build
cmake .. -DOpenCV_DIR="D:/opencv/build" -G "Visual Studio 17 2022"

# 3) 编译
cmake --build . --config Release
```

### Linux / macOS

```bash
# 1) 安装 OpenCV
# Ubuntu:  sudo apt install libopencv-dev
# macOS:   brew install opencv

# 2) 配置 & 编译
cd detect
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 启用 ONNX Runtime (可选)

```bash
cmake .. -DUSE_ONNX_RUNTIME=ON -DONNX_RUNTIME_DIR=/path/to/onnxruntime
```

## 使用

```bash
# 单张图像
./rm_armor_detect ../models/best.onnx ../assets/test.jpg

# 视频文件
./rm_armor_detect ../models/best.onnx ../assets/video.mp4 ../results/

# 实时摄像头
./rm_armor_detect ../models/best.onnx camera
```

### 快捷键

| 按键 | 功能 |
|------|------|
| ESC / q | 退出程序 |

## 技术细节

### 1. YOLO ONNX 推理

使用 OpenCV DNN 模块加载 YOLO11 ONNX 模型：

```
图像 → Letterbox Resize (640×640) → blobFromImage → net.forward() → 解码 → NMS → 检测结果
```

### 2. EKF 多目标跟踪

**状态向量 (8维):** `[x, y, w, h, vx, vy, vw, vh]`

**运动模型:** 匀速直线运动 (Constant Velocity)

**数据关联:** 贪心 IoU 匹配

**生命周期:**
- 新检测 → 初始化 EKF
- 连续匹配 → 维持跟踪
- 连续丢失 >30 帧 → 删除跟踪器

### 3. PnP 位姿估计

使用 `solvePnP` (IPPE_SQUARE 方法):
- 2D 点: YOLO 检测框的四个角点
- 3D 点: 装甲板在 Z=0 平面的四个角点
- 输出: 旋转向量、平移向量、欧氏距离

### 4. FPS 测量

每帧记录从检测到跟踪完成的全流程耗时，实时显示并统计平均 FPS。

## 与第一阶段的关系

| 阶段 | 内容 | 输出 |
|------|------|------|
| 阶段一 | Python YOLO 训练 + ONNX 导出 | `best.onnx` (6 分类模型) |
| 阶段二 | C++ ONNX 推理 + EKF 跟踪 + PnP 位姿 | `rm_armor_detect` (可执行文件) |

## 参考

- [RoboMaster 2026 比赛规则手册](https://www.robomaster.com)
- [Ultralytics YOLO11](https://docs.ultralytics.com)
- [OpenCV DNN Documentation](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)

---

**Deus RoboMaster 2027 — 视觉组考核阶段二**
