"""
RoboMaster 装甲板检测 - 推理脚本
支持单张图像、目录批量、摄像头实时检测。
"""

import os
import sys
import argparse
import cv2
import numpy as np
from ultralytics import YOLO


def draw_boxes(image, results):
    img = image.copy()
    if results.boxes is None:
        return img

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = results.names.get(cls_id, f"class_{cls_id}")

        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        label = f"{class_name} {conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - label_h - baseline - 5), (x1 + label_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return img


def process_image(model, image_path, output_dir, conf_threshold=0.25):
    results = model(image_path, conf=conf_threshold)[0]
    img = cv2.imread(image_path)
    if img is None:
        from PIL import Image
        pil_img = Image.open(image_path)
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    annotated = draw_boxes(img, results)

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_detected{ext}")
    cv2.imwrite(output_path, annotated)
    print(f"  已保存: {output_path}")

    return len(results.boxes) if results.boxes is not None else 0


def main():
    parser = argparse.ArgumentParser(description="RoboMaster 装甲板检测推理")
    parser.add_argument("--model", type=str, default=None, help="模型权重路径")
    parser.add_argument("--source", type=str, required=True, help="输入图像路径或目录")
    parser.add_argument("--output", type=str, default="./results", help="输出目录")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--webcam", action="store_true", help="使用摄像头实时检测")
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    # 优先使用项目根目录的 runs，其次用 demo 目录
    default_model1 = os.path.join(PROJECT_ROOT, "runs", "armor_detect", "weights", "best.pt")
    default_model2 = os.path.join(PROJECT_ROOT, "demo_LiuQuan_vision_yolo", "runs", "armor_detect", "weights", "best.pt")
    default_model = default_model1 if os.path.exists(default_model1) else default_model2
    model_path = args.model or default_model

    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件: {model_path}")
        print("请先运行训练脚本或使用 --model 指定模型路径")
        sys.exit(1)

    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    if args.webcam:
        print("启动摄像头检测 (按 'q' 退出)...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("错误: 无法打开摄像头")
            sys.exit(1)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=args.conf)[0]
            annotated = draw_boxes(frame, results)
            cv2.imshow("RoboMaster Armor Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    total_detections = 0
    total_images = 0

    if os.path.isfile(args.source):
        print(f"处理图像: {args.source}")
        n = process_image(model, args.source, args.output, args.conf)
        total_detections += n
        total_images += 1
    elif os.path.isdir(args.source):
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        image_files = sorted([f for f in os.listdir(args.source) if f.lower().endswith(valid_exts)])
        if not image_files:
            print(f"错误: 目录 {args.source} 中没有图像文件")
            sys.exit(1)

        print(f"处理 {len(image_files)} 张图像...")
        for i, img_file in enumerate(image_files, 1):
            img_path = os.path.join(args.source, img_file)
            print(f"  [{i}/{len(image_files)}] {img_file}")
            n = process_image(model, img_path, args.output, args.conf)
            total_detections += n
            total_images += 1
    else:
        print(f"错误: 找不到输入源: {args.source}")
        sys.exit(1)

    print(f"\n===== 推理完成 =====")
    print(f"处理图像: {total_images} 张")
    print(f"检测目标: {total_detections} 个")
    print(f"结果保存: {args.output}")


if __name__ == "__main__":
    main()
