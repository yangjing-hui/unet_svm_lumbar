import os
import cv2
import numpy as np
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
import joblib
from tqdm import tqdm
from feature_extraction import extract_features

def main():
    model_path = '../trained_svm_model.pkl'
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在，请先运行模型训练脚本。")
        return

    try:
        loaded_model = joblib.load(model_path)

        new_binary_mask_path = '../images/tmp/single_result.jpg'
        new_original_image_path = '../images/tmp/upload_show_result.jpg'

        if not (os.path.exists(new_binary_mask_path) and os.path.exists(new_original_image_path)):
            print("图像文件路径错误或文件不存在。")
            return

        # 读取图像
        binary_mask = cv2.imread(new_binary_mask_path, 0)
        original_image = cv2.imread(new_original_image_path, 0)
        original_color = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

        # 二值图预处理：形态学操作去噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # 提取轮廓
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 过滤无效轮廓（设置最小面积）
        min_contour_area = 100  # 根据实际调整
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        for cnt in tqdm(valid_contours, desc="处理椎间盘", unit="个"):
            single_mask = np.zeros_like(binary_mask)
            cv2.drawContours(single_mask, [cnt], -1, 255, -1)

            features = extract_features(single_mask, original_image)
            prediction = loaded_model.predict([features])

            label = "Degenerative" if prediction[0] == 1 else "Normal"
            color = (0, 0, 255) if prediction[0] == 1 else (0, 255, 0)

            x, y, w, h = cv2.boundingRect(cnt)
            # 调整文字位置避免重叠和越界
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_x = x
            text_y = y - 10
            if text_y < 10:  # 位置太靠上，调整到下方
                text_y = y + h + 10
            cv2.rectangle(original_color, (x, y), (x + w, y + h), color, 2)
            cv2.putText(original_color, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imwrite("../annotated_result.jpg", original_color)

        # 显示图像
        app = QApplication([])
        window = QWidget()
        window.setWindowTitle("标注结果")
        layout = QVBoxLayout()
        label = QLabel()
        pixmap = QPixmap("../annotated_result.jpg")
        label.setPixmap(pixmap)
        layout.addWidget(label)
        window.setLayout(layout)
        window.show()
        app.exec_()

    except Exception as e:
        print(f"程序运行错误: {e}")


if __name__ == "__main__":
    main()