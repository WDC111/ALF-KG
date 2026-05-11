import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV

# =========================
# 1. 读取数据
# =========================
input_path = "cohort_final_static.csv"
output_cleaned_path = "patient_clinical.csv"
df = pd.read_csv(input_path)

# =========================
# 2. 字段定义
# =========================
id_cols = ["subject_id_x", "hadm_id"]
label_col = "outcome_21d"

categorical_cols = ["insurance", "language", "marital_status", "race"]
feature_cols = [c for c in df.columns if c not in id_cols + [label_col]]

# =========================
# 3. 删除缺失率 >80% 的特征
# =========================
missing_ratio = df[feature_cols].isnull().mean()
selected_features = missing_ratio[missing_ratio <= 0.8].index.tolist()
print(f"删除缺失率>80%前特征数: {len(feature_cols)}")
print(f"删除缺失率>80%后特征数: {len(selected_features)}")
print(feature_cols)
print(selected_features)
# =========================
# 4. 缺失值填补
# =========================
# 区分连续和分类变量
continuous_cols = [c for c in selected_features if c not in categorical_cols]

# 连续变量：中位数填补
imputer_cont = SimpleImputer(strategy="median")
df[continuous_cols] = imputer_cont.fit_transform(df[continuous_cols])

# 分类变量：固定值填补
for col in categorical_cols:
    if col in selected_features:
        df[col] = df[col].fillna("UNKNOWN")

# =========================
# 5. 分类变量数值化
# =========================
label_encoders = {}
for col in categorical_cols:
    if col in selected_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# =========================
# 6. 连续变量标准化
# =========================
scaler = StandardScaler()
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

# =========================
# 7. 输出处理后的全数值化文件
# =========================
df.to_csv(output_cleaned_path, index=False)
print(f"✅ 数据清理、填补、标准化完成，保存为: {output_cleaned_path}")

# =========================
# 8. L1 Logistic Regression 特征选择
# =========================
X = df[selected_features]
y = df[label_col]

l1_model = LogisticRegressionCV(
    penalty="l1",
    solver="saga",
    Cs=10,
    cv=5,
    scoring="roc_auc",
    max_iter=5000,
    n_jobs=-1,
    refit=True
)
l1_model.fit(X, y)

# 提取非零系数及特征
coef = l1_model.coef_.flatten()
selected_mask = coef != 0

selected_features_l1 = np.array(selected_features)[selected_mask]
selected_coefs_l1 = coef[selected_mask]

df_selected = pd.DataFrame({
    "feature": selected_features_l1,
    "coef": selected_coefs_l1
})
df_selected.to_csv("selected_features_l1_with_coef.csv", index=False)

print(f"非零 L1 特征数量: {len(selected_features_l1)}")
print(selected_features_l1)
print("文件已保存为 selected_features_l1_with_coef2.csv")
