"""
数据集划分脚本
将图像和标注文件按照 train/val 比例划分到对应目录。
默认比例: 80% 训练集, 20% 验证集
"""

import os
import shutil
import random


def split_dataset(
    image_dir='部分RM装甲板数据集',
    label_dir='dataset/all_labels',
    output_dir='dataset',
    train_ratio=0.8,
    seed=42
):
    # 设置随机种子以保证可复现
    random.seed(seed)

    # 创建输出目录
    train_img_dir = os.path.join(output_dir, 'images', 'train')
    train_lbl_dir = os.path.join(output_dir, 'labels', 'train')
    val_img_dir = os.path.join(output_dir, 'images', 'val')
    val_lbl_dir = os.path.join(output_dir, 'labels', 'val')

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    print(f"总图像数: {len(image_files)}")

    # 检查哪些图像有对应的标注文件
    paired_files = []
    unlabeled_files = []
    for img_file in image_files:
        name = os.path.splitext(img_file)[0]
        label_path = os.path.join(label_dir, f"{name}.txt")
        if os.path.exists(label_path):
            paired_files.append(img_file)
        else:
            unlabeled_files.append(img_file)

    print(f"有标注的图像: {len(paired_files)}")
    print(f"无标注的图像: {len(unlabeled_files)}")

    if unlabeled_files:
        print(f"⚠ 警告: {len(unlabeled_files)} 张图像缺少标注文件，将只复制图像（不含标签）")

    # 随机打乱并划分
    random.shuffle(paired_files)
    split_idx = int(len(paired_files) * train_ratio)

    train_files = paired_files[:split_idx]
    val_files = paired_files[split_idx:]

    print(f"\n训练集: {len(train_files)} 张")
    print(f"验证集: {len(val_files)} 张")

    # 复制文件到对应目录
    def copy_files(file_list, img_dst_dir, lbl_dst_dir):
        for img_file in file_list:
            # 复制图像
            src_img = os.path.join(image_dir, img_file)
            dst_img = os.path.join(img_dst_dir, img_file)
            shutil.copy2(src_img, dst_img)

            # 复制标签
            name = os.path.splitext(img_file)[0]
            src_lbl = os.path.join(label_dir, f"{name}.txt")
            dst_lbl = os.path.join(lbl_dst_dir, f"{name}.txt")
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)

    print("\n复制训练集文件...")
    copy_files(train_files, train_img_dir, train_lbl_dir)
    print("复制验证集文件...")
    copy_files(val_files, val_img_dir, val_lbl_dir)

    # 如果有未标注的图像，也复制到训练集（不带标签）
    if unlabeled_files:
        print(f"\n复制 {len(unlabeled_files)} 张无标注图像到训练集...")
        for img_file in unlabeled_files:
            src_img = os.path.join(image_dir, img_file)
            dst_img = os.path.join(train_img_dir, img_file)
            shutil.copy2(src_img, dst_img)

    print("\n===== 数据集划分完成 =====")
    print(f"训练集图像: {len(os.listdir(train_img_dir))} 张 -> {train_img_dir}")
    print(f"训练集标签: {len(os.listdir(train_lbl_dir))} 个 -> {train_lbl_dir}")
    print(f"验证集图像: {len(os.listdir(val_img_dir))} 张 -> {val_img_dir}")
    print(f"验证集标签: {len(os.listdir(val_lbl_dir))} 个 -> {val_lbl_dir}")


if __name__ == '__main__':
    split_dataset()
