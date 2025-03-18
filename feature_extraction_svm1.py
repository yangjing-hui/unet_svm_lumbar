import cv2
import numpy as np
from skimage.measure import moments, moments_central, moments_normalized, moments_hu
from skimage.filters import gabor

def extract_features(single_binary_mask, original_image):
    features = []

    # 1. 基本强度特征
    contours, _ = cv2.findContours(single_binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.zeros(3269)
    cnt = max(contours, key=cv2.contourArea)

    # 检查轮廓点数是否至少为 5 个
    if len(cnt) < 5:
        # 如果点数不足 5 个，将几何特征设为 0
        MA = 0
        ma = 0
        eccentricity = 0
        x, y = 0, 0
    else:
        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        eccentricity = MA / ma if ma != 0 else 0

    mask = np.zeros_like(original_image)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    mean_intensity = cv2.mean(original_image, mask=mask)[0]
    min_intensity = np.min(original_image[mask == 255])
    max_intensity = np.max(original_image[mask == 255])

    hist = cv2.calcHist([original_image], [0], mask.astype(np.uint8), [256], [0, 256])
    hist = hist.flatten()

    intensity_features = [mean_intensity, min_intensity, max_intensity, MA, ma, eccentricity]
    features.extend(intensity_features)
    features.extend(hist)

    # 2. 不变矩
    center = (y, x)  # 注意这里 y 在前，x 在后，因为 skimage 中是 (row, col) 顺序
    m = moments_hu(moments_normalized(moments_central(original_image, center=center)))
    features.extend(m)

    # 3. Gabor特征
    wavelengths = [2, 4, 6, 8, 10]
    orientations = [0, 15, 30, 45, 90, 120, 150, 180]

    for wavelength in wavelengths:
        for theta in orientations:
            filter_real, filter_imag = gabor(original_image, frequency=1 / wavelength, theta=np.deg2rad(theta))
            mean = np.mean(filter_real)
            var = np.var(filter_real)
            features.extend([mean, var])

    return np.array(features)