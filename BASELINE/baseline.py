import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# =========================
# 1. 读取数据
# =========================
df = pd.read_csv("patient_clinical_filt.csv")

# 假设第一列是 patient_id
id_col = df.columns[0]
label_col = "outcome_21d"
feature_cols = [c for c in df.columns if c not in [id_col, label_col]]

X = df[feature_cols].values
y = df[label_col].values

# =========================
# 2. 划分训练集和测试集（保持正负比例相同）
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y  # 分层抽样，保证正负比例一致
)

# =========================
# 3. 标准化（逻辑回归 & SVM 需要）
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 4. 定义模型
# =========================
models = {
    "SVM": SVC(kernel="rbf", probability=True),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
}

# =========================
# 5. 模型训练与预测
# =========================
results = []

for name, model in models.items():
    # 选择标准化特征或原始特征
    if name in ["SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:,1]
    else:  # RandomForest
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]

    # 计算指标
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # 计算敏感性（Sensitivity）和特异性（Specificity）
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # 正类召回率
    specificity = tn / (tn + fp)  # 负类预测正确率

    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1-score": f1,
        "Precision": prec,
        "Recall": rec,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "AUC": auc
    })

# =========================
# 6. 输出结果
# =========================
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("baseline_model_results_stratified.csv", index=False)
print("✅ baseline 结果已保存为 baseline_model_results_stratified.csv")
