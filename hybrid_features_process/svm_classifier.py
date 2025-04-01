import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import joblib

# 加载数据
degen_data = loadmat('Degen_feature.mat')
non_degen_data = loadmat('Non_Degen_feature.mat')

# 提取特征和标签
X_degen = degen_data['featureD'][:, :-1]  # 前3269列特征
y_degen = degen_data['featureD'][:, -1]  # 标签（1表示退化）

X_non_degen = non_degen_data['featureND'][:, :-1]
y_non_degen = non_degen_data['featureND'][:, -1]  # 标签（0表示非退化）

# 合并数据集
X_train = np.vstack((X_degen, X_non_degen))
y_train = np.hstack((y_degen, y_non_degen))

# 初始化模型参数
model_params = {
    'kernel': 'rbf',
    'gamma': 0.0227,
    'C': 259.59,
    'probability': True  # 用于概率输出
}

# 10折交叉验证
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'specificity': [],
    'f1_score': []
}

models = []  # 保存每个fold的模型

for fold, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train)):
    X_fold_train, X_fold_test = X_train[train_idx], X_train[test_idx]
    y_fold_train, y_fold_test = y_train[train_idx], y_train[test_idx]

    # 训练模型
    svm = SVC(**model_params)
    svm.fit(X_fold_train, y_fold_train)
    models.append(svm)

    # 预测
    y_pred = svm.predict(X_fold_test)
    cm = confusion_matrix(y_fold_test, y_pred)

    # 计算指标
    TN, FP, FN, TP = cm.ravel()

    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
    precision = TP / (TP + FP) * 100 if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) * 100 if (TN + FP) != 0 else 0
    f1_score = 2 * TP / (2 * TP + FP + FN) * 100 if (2 * TP + FP + FN) != 0 else 0

    cv_metrics['accuracy'].append(accuracy)
    cv_metrics['precision'].append(precision)
    cv_metrics['recall'].append(recall)
    cv_metrics['specificity'].append(specificity)
    cv_metrics['f1_score'].append(f1_score)

# 输出交叉验证结果
print("Cross-Validation Results:")
for metric, values in cv_metrics.items():
    print(f"{metric}: Mean = {np.mean(values):.2f}%, Std = {np.std(values):.2f}%")

# 用全部数据训练最终模型并保存
final_model = SVC(**model_params)
final_model.fit(X_train, y_train)
joblib.dump(final_model, 'svm_model.pkl')
print("\nFinal model saved as 'svm_model.pkl'")