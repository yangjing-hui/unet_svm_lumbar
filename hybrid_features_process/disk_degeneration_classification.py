import os
import cv2
import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
from skimage.filters import gabor_kernel
from skimage.transform import resize
from skimage.measure import regionprops, label
from skimage.morphology import label as sk_label
from skimage.util import img_as_ubyte
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.segmentation import find_boundaries  # 导入 find_boundaries 函数
from tqdm import tqdm


# 生成 Gabor 滤波器组
def gaborFilterBank(scale, direction, size_x, size_y):
    filters = []
    for theta in np.linspace(0, np.pi, direction, endpoint=False):
        for sigma in np.linspace(1, scale, scale):
            kernel = gabor_kernel(0.1, theta=theta, sigma_x=sigma, sigma_y=sigma)
            filters.append(kernel)
    return filters


# 提取 Gabor 特征
def gaborFeatures(img, gaborArray, block_size_x, block_size_y):
    height, width = img.shape
    num_blocks_h = height // block_size_y
    num_blocks_w = width // block_size_x
    feature_vector = []
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block = img[i * block_size_y:(i + 1) * block_size_y, j * block_size_x:(j + 1) * block_size_x]
            for kernel in gaborArray:
                filtered = cv2.filter2D(block, -1, kernel.real)
                feature_vector.extend([np.mean(filtered), np.std(filtered)])
    return np.array(feature_vector)


# 提取 Hue Moments 特征
def feature_vec(img):
    # 这里简单返回图像的均值作为示例，需要根据实际情况实现
    return np.array([np.mean(img)])


# 加载训练数据
Degen = loadmat('Degen_feature_Train.mat')
NDegen = loadmat('Non_Degen_feature_Train.mat')

featureD = Degen['featureD']
featureND = NDegen['featureND']

train = np.vstack((featureD[:, :3268], featureND[:, :3268]))
trainlabel = np.hstack((featureD[:, 3269], featureND[:, 3269]))

# 生成 Gabor 滤波器组
gaborArray = gaborFilterBank(5, 8, 39, 39)

# 训练 SVM 模型
Model = SVC(kernel='rbf', gamma=0.0227, C=259.59, probability=True)
Model.fit(train, trainlabel)

# 预测训练数据
predictlabel = Model.predict(train)

# 处理图像
# 读取图像
image_path = 'images/tmp/original.jpg'
iv_disc_path = 'images/tmp/mask.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
iv_disc = cv2.imread(iv_disc_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"无法读取图像: {image_path}")
if iv_disc is None:
    print(f"无法读取掩码图像: {iv_disc_path}")

# 将原图缩放到二值图同样大小
if image is not None and iv_disc is not None:
    image = cv2.resize(image, (iv_disc.shape[1], iv_disc.shape[0]))

print(f"分割掩码图像形状: {iv_disc.shape}")
print(f"分割掩码图像最大值: {np.max(iv_disc)}")
print(f"分割掩码图像最小值: {np.min(iv_disc)}")

# 显示分割后的图像
plt.figure(1)
plt.imshow(iv_disc, cmap='gray')
plt.show()

# 连通区域分析
labeled_iv_disc = sk_label(iv_disc)
regions = regionprops(labeled_iv_disc)
print(f"找到的连通区域数量: {len(regions)}")

centroids = np.array([region.centroid for region in regions])
sorted_indices = np.argsort(centroids[:, 1])
scentroid = centroids[sorted_indices]

# 显示原始图像并绘制边界
plt.figure()
plt.imshow(image, cmap='gray')
boundaries = []
for region in tqdm(regions, desc="绘制边界"):
    region_mask = (labeled_iv_disc == region.label)  # 创建当前区域的掩码
    boundary = find_boundaries(region_mask, mode='outer')  # 找到边界
    boundary_coords = np.argwhere(boundary)  # 获取边界坐标
    if len(boundary_coords) > 0:
        plt.plot(boundary_coords[:, 1], boundary_coords[:, 0], 'r', linewidth=1)
        boundaries.append(boundary_coords)
plt.show()

# 提取特征
Major = []
Minor = []
Eccen = []
maxID = []
minID = []
meanID = []
M_f = []
gaborD = []
hist_D = []
boxPoint = []

# 定义统一的裁剪图像大小
target_size = (20, 60)

for k in tqdm(range(len(regions)), desc="提取特征"):
    location = sorted_indices[k]
    region = regions[location]
    bbox = region.bbox
    t = [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]]
    rect = Rectangle((t[0], t[1]), t[2], t[3], edgecolor='b', linewidth=2, facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)

    cropped_image = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    # 将裁剪图像调整为统一大小
    cropped_image = resize(cropped_image, target_size)
    Major.append(region.major_axis_length)
    Minor.append(region.minor_axis_length)
    Eccen.append(region.eccentricity)
    maxID.append(np.max(cropped_image))
    minID.append(np.min(cropped_image))
    meanID.append(np.mean(cropped_image))
    M_f.append(feature_vec(cropped_image))
    gaborD.append(gaborFeatures(cropped_image, gaborArray, 4, 4))
    hist = cv2.calcHist([cropped_image], [0], None, [256], [0, 256]).flatten()
    hist_D.append(hist)
    boxPoint.append(t)

# 组合特征
Major = np.array(Major)
Minor = np.array(Minor)
Eccen = np.array(Eccen)
maxID = np.array(maxID)
minID = np.array(minID)
meanID = np.array(meanID)
M_f = np.array(M_f)
gaborD = np.array(gaborD)
hist_D = np.array(hist_D)
Feature = np.hstack((Major.reshape(-1, 1), Minor.reshape(-1, 1), Eccen.reshape(-1, 1),
                     maxID.reshape(-1, 1), minID.reshape(-1, 1), meanID.reshape(-1, 1),
                     M_f, gaborD, hist_D))

# 预测
predictlabeltest = Model.predict(Feature)
predictscore = Model.predict_proba(Feature)

# 打印预测概率值
print("预测概率值分布:")
print(f"最小值: {np.min(predictscore[:, 1])}")
print(f"最大值: {np.max(predictscore[:, 1])}")
print(f"平均值: {np.mean(predictscore[:, 1])}")

get_detect = predictscore[:, 1] * ((predictscore[:, 1] < 3) & (predictscore[:, 1] > 0.1))
r = np.nonzero(get_detect)[0]

print(f"预测标签: {predictlabeltest}")
print(f"预测概率: {predictscore}")
print(f"筛选出的可能退化区域数量: {len(r)}")

if len(r) == 0:
    # 无退化
    text = 'No Degeneration'
    I2 = cv2.putText(image.copy(), text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    plt.figure(4)
    # 设置 matplotlib 显示参数
    plt.rcParams['figure.dpi'] = 300
    plt.imshow(I2, cmap='gray')
    output_path = 'images/tmp/result.jpg'

    # 检查保存路径是否存在，不存在则创建
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 确保图像数据类型是 uint8
    I2 = cv2.convertScaleAbs(I2)
    # 设置保存参数以提高图像质量
    cv2.imwrite(output_path, I2, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"无退化，保存结果图像到: {output_path}")
else:
    # 有退化
    rects = [boxPoint[idx] for idx in r]
    boundbox = np.array(rects)
    # 这里 selectStrongestBbox 函数需要根据实际情况实现
    selectedBbox = boundbox
    selectedScore = get_detect[r]
    label = 'Degeneration'
    I2 = image.copy()
    for bbox in tqdm(selectedBbox, desc="绘制标注框"):
        x, y, w, h = bbox.astype(int)
        cv2.rectangle(I2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(I2, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    plt.figure(4)
    # 设置 matplotlib 显示参数
    plt.rcParams['figure.dpi'] = 300
    plt.imshow(I2, cmap='gray')
    output_path = 'images/tmp/result.jpg'

    # 检查保存路径是否存在，不存在则创建
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 确保图像数据类型是 uint8
    I2 = cv2.convertScaleAbs(I2)
    # 设置保存参数以提高图像质量
    cv2.imwrite(output_path, I2, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"有退化，保存结果图像到: {output_path}")
