# -*- coding: utf-8 -*-
"""
医学影像分析系统（椎间盘分割+退变预测）
"""
import os
import cv2
import numpy as np
import sys  # 必须导入sys
import joblib
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QFileDialog, QMessageBox, QHBoxLayout,
                             QVBoxLayout, QProgressBar)
import torch
from model.unet_model import UNet
from svm_extraction.feature_extraction import extract_features

# 设备配置
device = torch.device('cpu')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('椎间盘退变分析系统')
        self.resize(1200, 800)
        self.output_size = 480  # 关键初始化
        self.setWindowIcon(QIcon("images/UI/lufei.png"))

        # 初始化模型
        self.unet_model = self.load_unet_model()
        self.svm_model = self.load_svm_model()

        # 界面组件
        self.init_ui()
        self.img_path = ""
        self.tmp_dir = "images/tmp"
        os.makedirs(self.tmp_dir, exist_ok=True)

    def load_unet_model(self):
        """加载U-Net分割模型"""
        try:
            net = UNet(n_channels=1, n_classes=1)
            net.to(device=device)
            net.load_state_dict(torch.load('best_model.pth', map_location=device))
            net.eval()
            return net
        except Exception as e:
            QMessageBox.critical(self, "模型错误", f"加载分割模型失败：{str(e)}")
            return None

    def load_svm_model(self):
        """加载SVM分类模型"""
        try:
            return joblib.load('trained_svm_model.pkl')
        except Exception:
            return None

    def init_ui(self):
        """初始化界面布局"""
        main_layout = QVBoxLayout()

        # 图像显示区域
        img_widget = QWidget()
        img_layout = QHBoxLayout()
        self.left_img = QLabel("原始图像")
        self.right_img = QLabel("处理结果")
        self.left_img.setFixedSize(self.output_size, self.output_size)
        self.right_img.setFixedSize(self.output_size, self.output_size)
        self.left_img.setStyleSheet("border: 2px solid #f0f0f0;")
        self.right_img.setStyleSheet("border: 2px solid #f0f0f0;")
        img_layout.addWidget(self.left_img)
        img_layout.addWidget(self.right_img)
        img_widget.setLayout(img_layout)

        # 按钮区域
        btn_widget = QWidget()
        btn_layout = QHBoxLayout()
        self.upload_btn = QPushButton("上传影像", clicked=self.upload_image)
        self.segment_btn = QPushButton("分割椎间盘", clicked=self.segment_disc, enabled=False)
        self.predict_btn = QPushButton("退变分析", clicked=self.predict_degeneration, enabled=False)
        btn_layout.addWidget(self.upload_btn)
        btn_layout.addWidget(self.segment_btn)
        btn_layout.addWidget(self.predict_btn)
        btn_widget.setLayout(btn_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFormat("%p%")  # 显示百分比
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 状态提示
        self.status_label = QLabel("等待操作...", alignment=Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-size: 14px;")

        # 整合布局
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        main_layout.addWidget(img_widget)
        main_layout.addWidget(btn_widget)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.progress_bar)  # 将进度条添加到主布局的最后
        self.setCentralWidget(central_widget)

    def upload_image(self):
        """处理图像上传"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择医学影像", "",
                                                   "医学影像 (*.jpg *.jpeg *.png *.tif)")
        if file_path:
            self.img_path = file_path
            self.show_original_image()
            self.status_label.setText("图像已上传，点击分割")
            self.segment_btn.setEnabled(True)
            self.predict_btn.setEnabled(False)

    def show_original_image(self):
        """显示原始图像"""
        try:
            img = cv2.imread(self.img_path)
            img = cv2.resize(img, (self.output_size, self.output_size))
            cv2.imwrite(os.path.join(self.tmp_dir, "original.jpg"), img)
            self.left_img.setPixmap(QPixmap(os.path.join(self.tmp_dir, "original.jpg")))
        except Exception as e:
            QMessageBox.warning(self, "警告", f"图像加载失败：{str(e)}")

    def segment_disc(self):
        """执行椎间盘分割"""
        if not self.unet_model:
            QMessageBox.warning(self, "警告", "分割模型未加载")
            return

        try:
            self.status_label.setText("正在分割...")
            img = cv2.imread(self.img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (512, 512))
            tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(device)

            with torch.no_grad():
                pred = self.unet_model(tensor)

            mask = (pred.cpu().numpy()[0][0] > 0.5).astype(np.uint8) * 255
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            cv2.imwrite(os.path.join(self.tmp_dir, "mask.jpg"), mask)
            self.show_result_image("mask.jpg")

            self.status_label.setText("分割完成，点击退变分析")
            self.predict_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"分割失败：{str(e)}")

    def predict_degeneration(self):
        if not self.svm_model:
            QMessageBox.warning(self, "警告", "请先训练SVM模型")
            return

        try:
            self.status_label.setText("正在分析...")
            mask = cv2.imread(os.path.join(self.tmp_dir, "mask.jpg"), 0)
            original = cv2.imread(self.img_path)

            # 形态学处理
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # 轮廓提取
            contours, _ = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            # 过滤小轮廓
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
            total = len(valid_contours)
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)

            # 标注结果
            annotated = original.copy()
            prediction_counts = {'degenerative': 0, 'normal': 0}
            predictions = []  # 存储预测结果

            for i, cnt in enumerate(valid_contours):
                single_mask = np.zeros_like(mask, dtype=np.uint8)
                cv2.drawContours(single_mask, [cnt], -1, 255, -1)

                # 特征提取（确保灰度图）
                original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                features = extract_features(single_mask, original_gray)

                # 预测
                prediction = self.svm_model.predict([features])
                predictions.append(prediction[0])  # 保存预测结果

                # 更新计数
                if prediction[0] == 1:
                    prediction_counts['degenerative'] += 1
                else:
                    prediction_counts['normal'] += 1

                # 更新进度条
                self.progress_bar.setValue(i + 1)
                QApplication.processEvents()  # 处理界面更新

            # 统一标注
            for i, cnt in enumerate(valid_contours):
                prediction = predictions[i]
                color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                cv2.putText(annotated, "Degenerative" if prediction == 1 else "Normal",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 保存并显示结果
            result_path = os.path.join(self.tmp_dir, "annotated.jpg")
            cv2.imwrite(result_path, annotated)
            self.show_result_image("annotated.jpg")
            self.status_label.setText(
                f"分析完成，发现{total}个椎间盘（退变：{prediction_counts['degenerative']}，正常：{prediction_counts['normal']}）")
            self.progress_bar.setVisible(False)

        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "错误", f"预测失败：{str(e)}")

    def show_result_image(self, filename):
        """显示处理结果图像"""
        try:
            img = cv2.imread(os.path.join(self.tmp_dir, filename))
            img = cv2.resize(img, (self.output_size, self.output_size))
            cv2.imwrite(os.path.join(self.tmp_dir, "result.jpg"), img)
            self.right_img.setPixmap(QPixmap(os.path.join(self.tmp_dir, "result.jpg")))
        except Exception as e:
            QMessageBox.warning(self, "警告", f"结果显示失败：{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)  # 必须使用sys.argv
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())  # 必须使用sys.exit