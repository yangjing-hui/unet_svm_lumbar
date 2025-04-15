import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

# 加载数据
Degen = loadmat('Degen_feature.mat')
NDegen = loadmat('Non_Degen_feature.mat')

featureD = Degen['featureD']
featureND = NDegen['featureND']

# 提取训练特征和标签
train = np.vstack((featureD[:, :3269], featureND[:, :3269]))
trainlabel = np.hstack((featureD[:, 3269], featureND[:, 3269]))

# 初始化变量
confusionMatrix = []
errorMat = []
True_negative = []
False_Positive = []
False_Negative = []
True_Positive = []
accuracy = []
precision = []
recall = []
specificity = []
f_score = []
TNR = []

# 10 折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)
for train_index, test_index in kf.split(train):
    X_train, X_test = train[train_index], train[test_index]
    y_train, y_test = trainlabel[train_index], trainlabel[test_index]

    # 训练 SVM 模型
    svm = SVC(kernel='rbf', gamma=0.0227, C=259.59)
    svm.fit(X_train, y_train)

    # 预测
    y_pred = svm.predict(X_test)

    # 计算准确率
    accuracy_fold = np.mean(y_pred == y_test)
    errorMat.append(accuracy_fold)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    confusionMatrix.append(cm)

    # 提取 TN、FP、FN、TP
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    True_negative.append(TN)
    False_Positive.append(FP)
    False_Negative.append(FN)
    True_Positive.append(TP)

    # 计算评估指标
    accuracy_fold = (TP + TN) / (TP + FP + FN + TN) * 100
    precision_fold = TP / (TP + FP) * 100 if (TP + FP) != 0 else 0
    recall_fold = TP / (TP + FN) * 100 if (TP + FN) != 0 else 0
    specificity_fold = TN / (FP + TN) * 100 if (FP + TN) != 0 else 0
    f_score_fold = 2 * TP / (2 * TP + FP + FN) * 100 if (2 * TP + FP + FN) != 0 else 0
    TNR_fold = TN / (TN + FP) * 100 if (TN + FP) != 0 else 0

    accuracy.append(accuracy_fold)
    precision.append(precision_fold)
    recall.append(recall_fold)
    specificity.append(specificity_fold)
    f_score.append(f_score_fold)
    TNR.append(TNR_fold)

# 输出结果
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)
print("F1 Score:", f_score)
print("True Negative Rate:", TNR)