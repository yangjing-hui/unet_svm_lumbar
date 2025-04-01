from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from data_loader_svm2 import load_train_data, load_test_data

# 加载训练数据
train_data_dir = '../svmdata/train'
X_train, y_train = load_train_data(train_data_dir)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 创建 SVM 模型
svm_model = SVC(kernel='rbf', C=0.259, gamma=0.0227)

# 训练模型
svm_model.fit(X_train, y_train)

# 在验证集上进行预测
y_pred_val = svm_model.predict(X_val)

# 计算验证集评估指标
accuracy_val = accuracy_score(y_val, y_pred_val)
precision_val = precision_score(y_val, y_pred_val)
recall_val = recall_score(y_val, y_pred_val)
f1_val = f1_score(y_val, y_pred_val)

print(f"验证集 Accuracy: {accuracy_val}")
print(f"验证集 Precision: {precision_val}")
print(f"验证集 Recall: {recall_val}")
print(f"验证集 F1-score: {f1_val}")

# 保存训练好的模型
joblib.dump(svm_model, '../trained_svm_model.pkl')

# 加载测试数据
test_data_dir = '../svmdata/test'
X_test, y_test = load_test_data(test_data_dir)

# 在测试集上进行预测
y_pred_test = svm_model.predict(X_test)

# 计算测试集评估指标
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)

print(f"测试集 Accuracy: {accuracy_test}")
print(f"测试集 Precision: {precision_test}")
print(f"测试集 Recall: {recall_test}")
print(f"测试集 F1-score: {f1_test}")