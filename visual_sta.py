import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------
# 1. 读取 SHAP 值和患者结局
# -------------------------
shap_df = pd.read_csv("shap_value/shap_last_hour_49_named.csv")
patient_df = pd.read_csv("shap_value/cohort_final_static_named.csv")

# 从 patient_df 提取 hadm_id 和 outcome_21d
patient_outcome = patient_df[['hadm_id', 'outcome_21d']]

# 将 outcome 对应到 shap_df
shap_df = shap_df.merge(patient_outcome, on='hadm_id', how='left')
shap_df.rename(columns={'outcome_21d': 'outcome'}, inplace=True)

# -------------------------
# 2. 提取特征列
# -------------------------
feature_cols = [c for c in shap_df.columns if c not in ['hadm_id', 'outcome']]
shap_values = shap_df[feature_cols]

# -------------------------
# 3. 平均绝对值排序，Top20特征
# -------------------------
mean_abs_shap = shap_values.abs().mean().sort_values(ascending=False)
top_features = mean_abs_shap.head(20).index

# 平均特征重要性柱状图
plt.figure(figsize=(12, 6))
mean_abs_shap[top_features].plot(kind='bar', color='skyblue')
plt.ylabel("Mean |SHAP value|")
plt.title("Top 20 Feature Importance based on SHAP values")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# -------------------------
# 4. Top20特征按 outcome 分组均值柱状图
# -------------------------
shap_top = shap_df[top_features.tolist() + ['outcome']]

mean_by_outcome = shap_top.groupby('outcome').mean()
mean_outcome1 = mean_by_outcome.loc[1]
mean_outcome0 = mean_by_outcome.loc[0]

plt.figure(figsize=(12, 6))
x = np.arange(len(top_features))
plt.bar(x - 0.2, mean_outcome1[top_features], width=0.4, color='red', label='Outcome=1')
plt.bar(x + 0.2, mean_outcome0[top_features], width=0.4, color='blue', label='Outcome=0')
plt.xticks(x, top_features, rotation=90)
plt.ylabel("Mean SHAP value")
plt.title("Top 20 Features: Mean SHAP value by Outcome")
plt.legend()
plt.tight_layout()
plt.show()


# -------------------------
# 5. 单特征依赖图（Dependence Plot）
# -------------------------
def plot_dependence(shap_df, patient_df, feature, outcome_col='outcome'):
    """
    绘制单特征依赖图，x轴使用真实特征取值，颜色表示 outcome
    shap_df: 包含 hadm_id, 特征SHAP值, outcome
    patient_df: 原始患者数据，包含 hadm_id 和特征值
    feature: 要分析的特征列名（必须在 patient_df 和 shap_df 中存在）
    """
    # 匹配患者特征值
    feature_values = patient_df.set_index('hadm_id').loc[shap_df['hadm_id'], feature].values
    shap_values = shap_df[feature].values
    outcome = shap_df[outcome_col].values

    plt.figure(figsize=(8,5))
    scatter = plt.scatter(feature_values, shap_values, c=outcome, cmap='bwr', alpha=0.6)
    plt.colorbar(scatter, label='Outcome')
    plt.xlabel(f"{feature} value")
    plt.ylabel(f"SHAP value of {feature}")
    plt.title(f"Dependence Plot for feature {feature}")
    plt.tight_layout()
    plt.show()



# 示例：绘制 Top1 特征依赖图
# Top1特征
top_feature = top_features[0]
plot_dependence(shap_df, patient_df, top_feature, outcome_col='outcome')
top_feature = top_features[1]
plot_dependence(shap_df, patient_df, top_feature, outcome_col='outcome')
top_feature = top_features[2]
plot_dependence(shap_df, patient_df, top_feature, outcome_col='outcome')
top_feature = top_features[3]
plot_dependence(shap_df, patient_df, top_feature, outcome_col='outcome')
top_feature = top_features[4]
plot_dependence(shap_df, patient_df, top_feature, outcome_col='outcome')
top_feature = top_features[5]
plot_dependence(shap_df, patient_df, top_feature, outcome_col='outcome')
top_feature = top_features[6]
plot_dependence(shap_df, patient_df, top_feature, outcome_col='outcome')
top_feature = top_features[7]
plot_dependence(shap_df, patient_df, top_feature, outcome_col='outcome')
top_feature = top_features[8]
plot_dependence(shap_df, patient_df, top_feature, outcome_col='outcome')
top_feature = top_features[9]
plot_dependence(shap_df, patient_df, top_feature, outcome_col='outcome')

# -------------------------
# 6. 热力图展示患者 SHAP 值模式（按 outcome 分组）
# -------------------------
# 提取 Top20 特征
top_features_list = top_features.tolist()

# outcome=1
shap_outcome1 = shap_df[shap_df['outcome'] == 1]
plt.figure(figsize=(15, 8))
sns.heatmap(shap_outcome1[top_features_list].iloc[:100], cmap="coolwarm", center=0)
plt.xlabel("Feature")
plt.ylabel("Patient index (first 50, Outcome=1)")
plt.title("SHAP values heatmap (Outcome=1, first 100 patients)")
plt.xticks(rotation=90)  # 特征名竖着显示
plt.tight_layout()
plt.show()

# outcome=0
shap_outcome0 = shap_df[shap_df['outcome'] == 0]
plt.figure(figsize=(15, 8))
sns.heatmap(shap_outcome0[top_features_list].iloc[:100], cmap="coolwarm", center=0)
plt.xlabel("Feature")
plt.ylabel("Patient index (first 50, Outcome=0)")
plt.title("SHAP values heatmap (Outcome=0, first 100 patients)")
plt.xticks(rotation=90)  # 特征名竖着显示
plt.tight_layout()
plt.show()

