import cv2
import numpy as np
import os

# 指定要遍历的四个文件夹路径
folder_paths = [
    '/lumbar/svmdata/test/degenerative/',
    '/lumbar/svmdata/test/non_degenerative/',
    '/lumbar/svmdata/train/degenerative/',
    '/lumbar/svmdata/train/non_degenerative/'
]

# 遍历每个文件夹
for folder_path in folder_paths:
    # 遍历文件夹内的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.tif'):
                # 构建完整的图像路径
                image_path = os.path.join(root, file)
                # 读取图像
                original_image = cv2.imread(image_path)
                # 将图像转换为灰度图
                gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                # 使用 Otsu 方法自动计算阈值进行二值化
                _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # 查找轮廓
                contours, _ = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 创建一个空白掩膜
                mask = np.zeros_like(binary_image)

                # 填充轮廓内部为白色
                cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

                # 将掩膜与二值图进行融合
                result_image = cv2.bitwise_or(binary_image, mask)

                # 获取原图像文件名（不包含路径）
                filename_without_extension = os.path.splitext(os.path.basename(image_path))[0]
                # 生成保存的二值图文件名
                binary_image_filename = filename_without_extension + '_binary.tif'
                # 构建保存的二值图完整路径
                binary_image_path = os.path.join(root, binary_image_filename)
                # 保存生成的二值图
                cv2.imwrite(binary_image_path, result_image)
                print(f"已将 {image_path} 转换并保存为 {binary_image_path}")
