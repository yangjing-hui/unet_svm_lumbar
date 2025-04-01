import os
import cv2
import numpy as np
from scipy.io import loadmat
from skimage import measure
from skimage.filters import gabor_kernel
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 加载训练数据和模型
degen_data = loadmat('Degen_feature_Train.mat')
ndegen_data = loadmat('Non_Degen_feature_Train.mat')
train = np.vstack([degen_data['featureD'][:, :-1], ndegen_data['featureND'][:, :-1]])
trainlabel = np.hstack([degen_data['featureD'][:, -1], ndegen_data['featureND'][:, -1]])

# 特征标准化
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)

# 训练SVM模型
model = SVC(kernel='rbf', gamma=0.0227, C=259.59, probability=True)
model.fit(train_scaled, trainlabel)


def create_gabor_filters():
    filters = []
    for theta in np.linspace(0, np.pi, 8, endpoint=False):
        for sigma in [1, 2, 3, 4, 5]:
            kernel = gabor_kernel(
                frequency=0.1,
                theta=theta,
                sigma_x=sigma,
                sigma_y=sigma,
                n_stds=3,
                bandwidth=1.0
            )
            filters.append(kernel)
    return filters


def extract_gabor_features(img, filters):
    height, width = img.shape
    num_blocks_h = height // 4
    num_blocks_w = width // 4
    feature_vector = []

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block = img[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4]
            block = block.astype(np.uint8)
            for kernel in filters:
                filtered = cv2.filter2D(block, -1, kernel.real)
                feature_vector.extend([np.mean(filtered), np.std(filtered)])
    return np.array(feature_vector)


def extract_hue_moments(img):
    moments = cv2.HuMoments(cv2.moments(img)).flatten()
    return moments


def analyze_disc(image_path, mask_path, output_path):
    # 读取图像
    image = cv2.imread(image_path, 0)
    mask = cv2.imread(mask_path, 0)

    if image is None or mask is None:
        print("无法读取图像文件")
        return

    # 尺寸匹配
    h, w = image.shape
    mask = cv2.resize(mask, (w, h))

    # 区域分割
    labels = measure.label(mask > 0)
    regions = measure.regionprops(labels)

    features = []
    boxes = []

    for region in regions:
        bbox = region.bbox
        y1, x1, y2, x2 = bbox

        # 边界检查
        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(h, y2)
        x2 = min(w, x2)

        if y1 >= y2 or x1 >= x2:
            continue

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # 形态学特征
        major = region.major_axis_length
        minor = region.minor_axis_length
        eccen = region.eccentricity

        # 灰度统计
        max_val = np.max(roi)
        min_val = np.min(roi)
        mean_val = np.mean(roi)

        # Hue Moments
        hue = extract_hue_moments(roi)

        # Gabor特征
        roi_resized = resize(roi, (20, 60))
        gabor_feat = extract_gabor_features(roi_resized, gabor_filters)

        # 直方图
        hist = cv2.calcHist([roi], [0], None, [256], [0, 256]).flatten()

        # 合并特征
        feature = np.hstack([
            major, minor, eccen,
            max_val, min_val, mean_val,
            hue, gabor_feat, hist
        ])
        features.append(feature)
        boxes.append([x1, y1, x2 - x1, y2 - y1])

    if not features:
        print("未检测到椎间盘区域")
        return

    # 特征标准化
    features_scaled = scaler.transform(features)

    # 预测
    probabilities = model.predict_proba(features_scaled)
    predictions = (probabilities[:, 1] > 0.5).astype(int)

    # 结果标注
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    has_degeneration = False

    for box, prob, pred in zip(boxes, probabilities[:, 1], predictions):
        x1, y1, w, h = box
        if pred == 1:
            has_degeneration = True
            cv2.rectangle(image_color, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 0, 255), 2)
            cv2.putText(image_color, f"{prob:.2f}", (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if not has_degeneration:
        cv2.putText(image_color, 'No Degeneration', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imwrite(output_path, image_color)
    print(f"结果已保存到：{output_path}")


# 初始化Gabor滤波器
gabor_filters = create_gabor_filters()

# 处理单张图像
image_path = 'images/tmp/original.jpg'
mask_path = 'images/tmp/mask.jpg'
output_path = 'images/tmp/result.jpg'

analyze_disc(image_path, mask_path, output_path)