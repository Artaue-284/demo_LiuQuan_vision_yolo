"""RoboMaster 装甲板检测 — YOLO11 训练脚本 (demo 目录备份)"""
import os
import sys
from ultralytics import YOLO


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    data_yaml = os.path.join(project_root, "dataset", "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"错误: 找不到 {data_yaml}")
        sys.exit(1)

    print(f"数据集配置: {data_yaml}")

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    print(f"使用设备: {device}")

    model = YOLO(os.path.join(project_root, "yolo11n.pt"))

    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16 if device == "cuda" else 8,
        device=device,
        workers=0,
        patience=20,
        save=True,
        save_period=10,
        project=os.path.join(script_dir, "runs"),
        name="armor_detect",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        val=True,
        plots=True,
        amp=False,
    )

    best_model = os.path.join(script_dir, "runs", "armor_detect", "weights", "best.pt")
    print(f"\n训练完成! 最佳模型: {best_model}")

    metrics = model.val()
    print(f"验证集 - mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
