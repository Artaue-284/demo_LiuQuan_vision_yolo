"""YOLO模型导出为ONNX格式"""
import os
import sys
from ultralytics import YOLO


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, "runs", "armor_detect", "weights")
    best_pt = os.path.join(weights_dir, "best.pt")

    if not os.path.exists(best_pt):
        print(f"错误: 找不到 {best_pt}")
        print("请先运行 yolo_train.py 训练模型")
        sys.exit(1)

    print(f"加载模型: {best_pt}")
    model = YOLO(best_pt)

    print("导出 ONNX...")
    model.export(format="onnx", imgsz=640, simplify=True)

    onnx_path = os.path.join(weights_dir, "best.onnx")
    if os.path.exists(onnx_path):
        print(f"ONNX 导出成功: {onnx_path}")
    else:
        print("ONNX 导出失败")


if __name__ == "__main__":
    main()
