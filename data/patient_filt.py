import pandas as pd

# =========================
# 1. 读取原始患者数据
# =========================
df = pd.read_csv("patient_clinical.csv")

# =========================
# 2. 读取 L1 筛选出的特征
# =========================
df_selected = pd.read_csv("selected_features_l1_with_coef.csv")
selected_features = df_selected["feature"].tolist()

# =========================
# 3. 构建新的数据表（保留 L1 特征 + label）
# =========================
label_col = "outcome_21d"
new_feature_cols = selected_features + [label_col]

df_reduced = df[new_feature_cols]

# =========================
# 4. 保存新的患者数据
# =========================
output_path = "patient_clinical_filt.csv"
df_reduced.to_csv(output_path, index=False)

print(f"✅ 新的患者数据已构建，只包含 {len(selected_features)} 个 L1 特征 + label")
print("文件保存为:", output_path)
