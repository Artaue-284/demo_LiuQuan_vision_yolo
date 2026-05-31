"""
RM装甲板自动标注脚本 — 6分类版本
类别: red3(0) red1(1) redsb(2) blue3(3) blue1(4) bluesb(5)
使用OpenCV检测装甲板LED灯条，根据颜色和尺寸区分机器人类型。
"""

import cv2
import numpy as np
from PIL import Image
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# 类别映射
CLASS_NAMES = {0: 'red3', 1: 'red1', 2: 'redsb', 3: 'blue3', 4: 'blue1', 5: 'bluesb'}


def imread_unicode(path):
    """支持中文路径的图像读取"""
    pil_img = Image.open(path)
    img = np.array(pil_img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def sample_color(img, x, y, bw, bh):
    """采样检测区域的主导颜色: 返回 'red' 或 'blue' 或 'unknown'"""
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img.shape[1], x + bw), min(img.shape[0], y + bh)
    if x2 <= x1 or y2 <= y1:
        return 'unknown'
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return 'unknown'
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 统计红色和蓝色像素
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 100]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([160, 50, 100]), np.array([180, 255, 255]))
    red_count = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)
    blue_mask = cv2.inRange(hsv, np.array([95, 50, 80]), np.array([130, 255, 255]))
    blue_count = cv2.countNonZero(blue_mask)
    if red_count > blue_count and red_count > roi.shape[0] * roi.shape[1] * 0.05:
        return 'red'
    elif blue_count > red_count and blue_count > roi.shape[0] * roi.shape[1] * 0.05:
        return 'blue'
    return 'unknown'


def classify_by_color_and_size(color, area_ratio, h, w):
    """根据颜色和尺寸比例确定具体类别"""
    if color == 'red':
        if area_ratio > 0.12:
            return 1  # red1 英雄 (装甲板较大)
        else:
            return 0  # red3 步兵
    elif color == 'blue':
        if area_ratio > 0.12:
            return 4  # blue1 英雄
        else:
            return 3  # blue3 步兵
    else:
        # 无法判断颜色，默认步兵
        return 0


def detect_armor_plates(img):
    """
    检测图像中的装甲板区域，返回6分类结果。
    返回: [(x_center, y_center, width, height, class_id), ...]
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    all_detections = []  # (x, y, bw, bh, class_id, confidence)

    # ===== 策略1: 亮度检测 (LED灯条非常亮) =====
    brightness_threshold = np.percentile(gray, 95)
    brightness_threshold = max(brightness_threshold, 180)
    _, bright_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bright_closed = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bright_dilated = cv2.dilate(bright_closed, kernel_dilate, iterations=1)

    contours, _ = cv2.findContours(bright_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100 or area > w * h * 0.3:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = max(bw, bh) / (min(bw, bh) + 1)
        if 1.3 < aspect < 7 and bw > 10 and bh > 10:
            color = sample_color(img, x - 5, y - 5, bw + 10, bh + 10)
            area_ratio = area / (w * h)
            cls_id = classify_by_color_and_size(color, area_ratio, h, w)
            all_detections.append((x, y, bw, bh, cls_id, 0.8))

    # ===== 策略2: 红色装甲板 =====
    lower_red1 = np.array([0, 50, 120])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 120])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    kernel_red = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    red_dilated = cv2.dilate(red_mask, kernel_red, iterations=2)
    contours_r, _ = cv2.findContours(red_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_r:
        area = cv2.contourArea(c)
        if area < 80 or area > w * h * 0.3:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = max(bw, bh) / (min(bw, bh) + 1)
        if 1.3 < aspect < 7 and bw > 8 and bh > 8:
            area_ratio = area / (w * h)
            cls_id = 1 if area_ratio > 0.12 else 0  # 红方英雄 vs 步兵
            all_detections.append((x, y, bw, bh, cls_id, 0.9))

    # ===== 策略3: 蓝色装甲板 =====
    lower_blue = np.array([95, 50, 100])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel_blue = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blue_dilated = cv2.dilate(blue_mask, kernel_blue, iterations=2)
    contours_b, _ = cv2.findContours(blue_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_b:
        area = cv2.contourArea(c)
        if area < 80 or area > w * h * 0.3:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = max(bw, bh) / (min(bw, bh) + 1)
        if 1.3 < aspect < 7 and bw > 8 and bh > 8:
            area_ratio = area / (w * h)
            cls_id = 4 if area_ratio > 0.12 else 3  # 蓝方英雄 vs 步兵
            all_detections.append((x, y, bw, bh, cls_id, 0.9))

    # ===== 策略4: 高饱和度+高亮度 =====
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    combined_mask = ((s_channel > 80) & (v_channel > 150)).astype(np.uint8) * 255
    kernel_comb = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    comb_dilated = cv2.dilate(combined_mask, kernel_comb, iterations=2)
    contours_c, _ = cv2.findContours(comb_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_c:
        area = cv2.contourArea(c)
        if area < 80 or area > w * h * 0.3:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = max(bw, bh) / (min(bw, bh) + 1)
        if 1.3 < aspect < 7 and bw > 8 and bh > 8:
            color = sample_color(img, x - 5, y - 5, bw + 10, bh + 10)
            area_ratio = area / (w * h)
            cls_id = classify_by_color_and_size(color, area_ratio, h, w)
            all_detections.append((x, y, bw, bh, cls_id, 0.5))

    # ===== 按类别分组做NMS =====
    if not all_detections:
        return []

    # 按 class_id 分组
    class_groups = {}
    for det in all_detections:
        cls_id = det[4]
        if cls_id not in class_groups:
            class_groups[cls_id] = []
        class_groups[cls_id].append(det)

    final_boxes = []
    for cls_id, dets in class_groups.items():
        boxes = np.array([[x, y, x + bw, y + bh] for (x, y, bw, bh, _, _) in dets], dtype=np.float32)
        scores = np.array([s for (_, _, _, _, _, s) in dets], dtype=np.float32)
        keep_idx = nms(boxes, scores, iou_threshold=0.3)
        for idx in keep_idx:
            x1, y1, x2, y2 = boxes[idx]
            bw = x2 - x1
            bh = y2 - y1
            cx = (x1 + bw / 2) / w
            cy = (y1 + bh / 2) / h
            nw = bw / w
            nh = bh / h
            final_boxes.append((cx, cy, nw, nh, cls_id))

    return final_boxes


def nms(boxes, scores, iou_threshold=0.3):
    """非极大值抑制"""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w_inter = np.maximum(0, xx2 - xx1)
        h_inter = np.maximum(0, yy2 - yy1)
        inter = w_inter * h_inter
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_others = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        union = area_i + area_others - inter
        iou = inter / (union + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def merge_nearby_boxes(boxes, w, h, distance_threshold=0.05):
    """合并同类别中距离很近的框"""
    if len(boxes) < 2:
        return boxes

    # 按类别分组
    class_groups = {}
    for item in boxes:
        cx, cy, nw, nh, cls_id = item
        if cls_id not in class_groups:
            class_groups[cls_id] = []
        class_groups[cls_id].append((cx, cy, nw, nh))

    merged = []
    for cls_id, items in class_groups.items():
        if len(items) < 2:
            for (cx, cy, nw, nh) in items:
                merged.append((cx, cy, nw, nh, cls_id))
            continue

        px_boxes = []
        for (cx, cy, nw, nh) in items:
            x1 = (cx - nw / 2) * w
            y1 = (cy - nh / 2) * h
            x2 = (cx + nw / 2) * w
            y2 = (cy + nh / 2) * h
            px_boxes.append([x1, y1, x2, y2])
        px_boxes = np.array(px_boxes)

        used = set()
        for i in range(len(px_boxes)):
            if i in used:
                continue
            group = [px_boxes[i]]
            used.add(i)
            changed = True
            while changed:
                changed = False
                for j in range(len(px_boxes)):
                    if j in used:
                        continue
                    for g in group:
                        dx = max(0, max(g[0], px_boxes[j][0]) - min(g[2], px_boxes[j][2]))
                        dy = max(0, max(g[1], px_boxes[j][1]) - min(g[3], px_boxes[j][3]))
                        if dx < distance_threshold * w and dy < distance_threshold * h:
                            group.append(px_boxes[j])
                            used.add(j)
                            changed = True
                            break

            group = np.array(group)
            x1 = np.min(group[:, 0])
            y1 = np.min(group[:, 1])
            x2 = np.max(group[:, 2])
            y2 = np.max(group[:, 3])
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            nw = (x2 - x1) / w
            nh = (y2 - y1) / h
            merged.append((cx, cy, nw, nh, cls_id))

    return merged


def save_yolo_label(label_path, boxes):
    """保存YOLO格式标注文件"""
    with open(label_path, 'w', encoding='utf-8') as f:
        for (cx, cy, nw, nh, cls_id) in boxes:
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")


def main():
    src_dir = '部分RM装甲板数据集'
    label_dir = 'dataset/all_labels'
    vis_dir = 'dataset/visualization'

    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort()

    stats = {'red3': 0, 'red1': 0, 'redsb': 0, 'blue3': 0, 'blue1': 0, 'bluesb': 0,
             'empty': 0, 'error': 0, 'multi': 0}

    for i, f in enumerate(files):
        try:
            path = os.path.join(src_dir, f)
            img = imread_unicode(path)
            h, w = img.shape[:2]

            boxes = detect_armor_plates(img)
            boxes = merge_nearby_boxes(boxes, w, h)

            name = os.path.splitext(f)[0]
            label_path = os.path.join(label_dir, f"{name}.txt")
            save_yolo_label(label_path, boxes)

            if boxes:
                if len(boxes) > 1:
                    stats['multi'] += 1
                cls_counts = {}
                for _, _, _, _, cls_id in boxes:
                    cls_name = CLASS_NAMES.get(cls_id, 'unknown')
                    cls_counts[cls_name] = cls_counts.get(cls_name, 0) + 1
                for cls_name, cnt in cls_counts.items():
                    stats[cls_name] = stats.get(cls_name, 0) + cnt
            else:
                stats['empty'] += 1

            if (i + 1) % 50 == 0:
                total_det = sum(stats.get(c, 0) for c in CLASS_NAMES.values())
                print(f"进度: {i + 1}/{len(files)} - 检测框: {total_det}, 空: {stats['empty']}")

        except Exception as e:
            stats['error'] += 1
            print(f"处理 {f} 时出错: {e}")

    total_det = sum(stats.get(c, 0) for c in CLASS_NAMES.values())
    print(f"\n===== 标注完成 =====")
    print(f"总图像数: {len(files)}")
    print(f"检测到装甲板: {total_det} 个框")
    print(f"  红方步兵(red3): {stats.get('red3', 0)}")
    print(f"  红方英雄(red1): {stats.get('red1', 0)}")
    print(f"  红方哨兵(redsb): {stats.get('redsb', 0)}")
    print(f"  蓝方步兵(blue3): {stats.get('blue3', 0)}")
    print(f"  蓝方英雄(blue1): {stats.get('blue1', 0)}")
    print(f"  蓝方哨兵(bluesb): {stats.get('bluesb', 0)}")
    print(f"其中多目标图像: {stats['multi']}")
    print(f"未检测到: {stats['empty']}")
    print(f"处理错误: {stats['error']}")
    print(f"\n标注文件已保存到: {label_dir}/")


if __name__ == '__main__':
    main()
