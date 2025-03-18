# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: unet
File Name: test.py
Author: yjh
Create Date: 2025/3/12
Description：
-------------------------------------------------
"""
import os
from tqdm import tqdm
from utils.utils_metrics import compute_mIoU, show_results
import glob
import numpy as np
import torch
import cv2
from model.unet_model import UNet


def cal_miou(test_dir="D:/oldD/computer/finalproject/unet_42-master/lumbar/Test_Images",
             pred_dir="D:/oldD/computer/finalproject/unet_42-master/lumbar/results", gt_dir="D:/oldD/computer/finalproject/unet_42-master/lumbar/Test_Labels"):
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    # ------------------------------#
    #   分类个数+1、如2+1
    # ------------------------------#
    num_classes = 2
    # --------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    # --------------------------------------------#
    name_classes = ["background", "lumbar"]
    # name_classes    = ["_background_","cat","dog"]
    # -------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    # -------------------------------------------------------#
    # 计算结果和gt的结果进行比对

    # 加载模型
    valid_image_ids = []

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        # 加载网络，图片单通道，分类为1。
        net = UNet(n_channels=1, n_classes=1)
        # 将网络拷贝到deivce中
        net.to(device=device)
        # 加载模型参数
        net.load_state_dict(torch.load('best_model.pth', map_location=device))  # todo
        # 测试模式
        net.eval()
        print("Load model done.")

        # 获取所有 .nii.jpg 文件
        image_paths = glob.glob(os.path.join(test_dir, "*.nii.jpg"))
        image_ids = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(test_dir, f"{image_id}.jpg")
            # 检查文件是否存在
            if not os.path.exists(image_path):
                print(f"文件 {image_path} 不存在，请检查路径。")
                continue
            img = cv2.imread(image_path)
            # 检查图像是否读取成功
            if img is None:
                print(f"无法读取文件 {image_path}，请检查文件是否损坏。")
                continue
            valid_image_ids.append(image_id)
            origin_shape = img.shape
            # 转为灰度图
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (512, 512))
            # 转为batch为1，通道为1，大小为512*512的数组
            img = img.reshape(1, 1, img.shape[0], img.shape[1])
            # 转为tensor
            img_tensor = torch.from_numpy(img)
            # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
            # 预测
            pred = net(img_tensor)
            # 提取结果
            pred = np.array(pred.data.cpu()[0])[0]
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
            # 保存预测结果，文件名包含 .nii
            cv2.imwrite(os.path.join(pred_dir, f"{image_id}.png"), pred)

        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        print(gt_dir)
        print(pred_dir)
        print(num_classes)
        print(name_classes)
        if valid_image_ids:
            hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, valid_image_ids, num_classes,
                                                            name_classes)  # 执行计算mIoU的函数
            print("Get miou done.")
            miou_out_path = "results/"
            show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
        else:
            print("没有有效的图像文件，无法计算mIoU。")


if __name__ == '__main__':
    cal_miou()