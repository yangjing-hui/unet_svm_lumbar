import os
import numpy as np
from feature_extraction_svm1 import extract_features


def load_train_data(data_dir):
    X = []
    y = []

    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            label = 1 if class_name == 'degenerative' else 0
            for image_name in os.listdir(class_dir):
                if image_name.endswith('.tif'):
                    binary_mask_path = os.path.join(class_dir, image_name)
                    # 假设原始图像路径和二值图路径有对应关系，这里需要根据实际情况修改
                    original_image_path = binary_mask_path.replace('binary', 'original')
                    features = extract_features(binary_mask_path, original_image_path)
                    X.append(features)
                    y.append(label)

    return np.array(X), np.array(y)


def load_prediction_data(binary_mask_path, original_image_path):
    features = extract_features(binary_mask_path, original_image_path)
    return np.array([features])


def load_test_data(data_dir):
    X = []
    y = []

    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            label = 1 if class_name == 'degenerative' else 0
            for image_name in os.listdir(class_dir):
                if image_name.endswith('.tif'):
                    binary_mask_path = os.path.join(class_dir, image_name)
                    # 假设原始图像路径和二值图路径有对应关系，这里需要根据实际情况修改
                    original_image_path = binary_mask_path.replace('binary', 'original')
                    features = extract_features(binary_mask_path, original_image_path)
                    X.append(features)
                    y.append(label)

    return np.array(X), np.array(y)
