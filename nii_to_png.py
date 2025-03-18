# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: unet
File Name: nii_to_png.py
Author: yjh
Create Date: 2025/3/10
Description：
-------------------------------------------------
"""
import nibabel as nib
import numpy as np
import os
import cv2


def load_nii_gz(path):
    """加载 .nii.gz 文件并返回数据和仿射矩阵"""
    img = nib.load(path)
    return img.get_fdata(), img.affine


def process_nii_data(input_img_path, input_label_path, output_nii_dir, output_img_dir, output_label_dir):
    # 加载数据
    img_data, img_affine = load_nii_gz(input_img_path)
    label_data, _ = load_nii_gz(input_label_path)  # 标注图不需要仿射矩阵

    # 取中间切片
    middle_slice_idx = int(img_data.shape[2] / 2)

    # 处理原图切片（医学影像窗宽窗位处理）
    img_slice = img_data[:, :, middle_slice_idx]
    window_center = np.percentile(img_slice, 90)
    window_width = np.percentile(img_slice, 99) - np.percentile(img_slice, 10)
    img_slice = np.clip(img_slice, window_center - window_width / 2, window_center + window_width / 2)
    img_slice = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())) * 255
    img_slice = img_slice.astype(np.uint8)

    # 向左旋转90度
    img_slice = cv2.rotate(img_slice, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 处理标注图切片（提取较亮椎管区域）
    label_slice = label_data[:, :, middle_slice_idx]
    threshold = np.mean(label_slice)+10
    label_slice = np.where(label_slice > threshold, 255, 0).astype(np.uint8)

    # 向左旋转90度
    label_slice = cv2.rotate(label_slice, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 获取基础文件名
    base_name = os.path.splitext(os.path.basename(input_img_path))[0]

    # 保存原图
    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}.jpg"), img_slice)
    # 保存标注图
    cv2.imwrite(os.path.join(output_label_dir, f"{base_name}.png"), label_slice)


def main():
    # 配置参数
    input_dir = r"C:/Users/lenovo/Downloads/train/Mask"  # 输入目录
    output_nii_dir = r"C:/Users/lenovo/Downloads/train/newMask"  # 增强后.nii.gz保存目录
    output_img_dir = r"D:/oldD/computer/finalproject/unet_42-master/skin/Training_Images"  # 原图JPG保存目录
    output_label_dir = r"D:/oldD/computer/finalproject/unet_42-master/skin/Training_Labels"  # 标注图PNG保存目录

    # 创建输出目录
    os.makedirs(output_nii_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 遍历处理文件
    for file in os.listdir(input_dir):
        if file.endswith('.nii.gz') and file.startswith('Case'):
            img_path = os.path.join(input_dir, file)
            label_path = os.path.join(input_dir, file.replace('Case', 'mask_case'))
            if os.path.exists(label_path):
                print(f"Processing {file}...")
                process_nii_data(img_path, label_path, output_nii_dir, output_img_dir, output_label_dir)
    print("Processing completed!")


if __name__ == "__main__":
    main()