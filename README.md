# demo_LiuQuan_vision_yolo

RoboMaster 2027 视觉组考核 — 装甲板检测与跟踪系统

## 项目概述

本项目实现了 RoboMaster 比赛中装甲板的**检测、多目标跟踪与位姿估计**三大核心功能。

| 阶段 | 模块 | 技术栈 |
|------|------|--------|
| 阶段一 | YOLO 模型训练与导出 | Python + Ultralytics YOLO11 + ONNX |
| 阶段二 | C++ 推理 + EKF 跟踪 + PnP 位姿 | C++17 + OpenCV DNN + EKF + solvePnP |

## 目录结构

```
demo_LiuQuan_vision_yolo/
├── yolo_train.py          # YOLO11 训练脚本 (阶段一)
├── yolo_convert.py        # ONNX 导出脚本 (阶段一)
├── auto_annotate.py       # 自动标注脚本 (阶段一)
├── split_dataset.py       # 数据集划分 (阶段一)
├── models/
│   └── best.onnx          # 训练好的模型
├── detect/                # C++ 推理工程 (阶段二)
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── include/
│   │   ├── Detector.h     # YOLO 检测器
│   │   ├── KalmanFilter.h # EKF 跟踪器
│   │   ├── Tracker.h      # 多目标跟踪
│   │   └── PnPSolver.h    # 位姿估计
│   └── src/
│       ├── main.cpp       # 主程序
│       ├── Detector.cpp
│       ├── KalmanFilter.cpp
│       ├── Tracker.cpp
│       └── PnPSolver.cpp
├── assets/                # 测试图像/视频
└── results/               # 输出结果
```

## 快速开始

### 阶段一: Python 训练

```bash
pip install ultralytics torch opencv-python
python yolo_train.py       # 训练 YOLO11 模型
python yolo_convert.py     # 导出 ONNX
```

### 阶段二: C++ 推理

```bash
cd detect
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 运行
./rm_armor_detect ../models/best.onnx ../assets/test.jpg ../results/
```

详细说明见 [`detect/README.md`](detect/README.md)。

## 检测类别 (6 类)

- `red3` / `blue3` — 步兵 (3号装甲板, 230×125mm)
- `red1` / `blue1` — 英雄 (1号装甲板, 340×125mm)
- `redsb` / `bluesb` — 哨兵 (200×100mm)

## 开发环境

- Python 3.10+ / PyTorch 2.x / Ultralytics
- C++17 / CMake 3.16+ / OpenCV 4.5+
- ONNX Runtime (可选)

---

**Deus RoboMaster 2027 — 视觉组**
