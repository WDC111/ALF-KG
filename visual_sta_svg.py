import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib

# 设置 SVG 文字为可编辑模式（不将文字转为路径）
matplotlib.rcParams['svg.fonttype'] = 'none'

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
plt.savefig("feature_importance_top20.svg", format="svg")  # 保存为 SVG
# plt.show()  # 注释掉显示

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
plt.savefig("mean_shap_by_outcome.svg", format="svg")  # 保存为 SVG


# plt.show()

# -------------------------
# 5. 单特征依赖图（Dependence Plot）
# -------------------------
def plot_dependence(shap_df, patient_df, feature, outcome_col='outcome', save_path=None):
    """
    绘制单特征依赖图，x轴使用真实特征取值，颜色表示 outcome
    如果 save_path 提供，则保存图形到该路径，否则显示图形。
    """
    # 匹配患者特征值
    feature_values = patient_df.set_index('hadm_id').loc[shap_df['hadm_id'], feature].values
    shap_values = shap_df[feature].values
    outcome = shap_df[outcome_col].values

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(feature_values, shap_values, c=outcome, cmap='bwr', alpha=0.6)
    plt.colorbar(scatter, label='Outcome')
    plt.xlabel(f"{feature} value")
    plt.ylabel(f"SHAP value of {feature}")
    plt.title(f"Dependence Plot for feature {feature}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg")
        plt.close()  # 关闭图形以释放内存
    else:
        plt.show()


# 为前10个特征绘制依赖图并保存
for i, feat in enumerate(top_features[:10]):
    filename = f"dependence_{feat}.svg"
    plot_dependence(shap_df, patient_df, feat, outcome_col='outcome', save_path=filename)

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
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("heatmap_outcome1.svg", format="svg")  # 保存为 SVG
# plt.show()

# outcome=0
shap_outcome0 = shap_df[shap_df['outcome'] == 0]
plt.figure(figsize=(15, 8))
sns.heatmap(shap_outcome0[top_features_list].iloc[:100], cmap="coolwarm", center=0)
plt.xlabel("Feature")
plt.ylabel("Patient index (first 50, Outcome=0)")
plt.title("SHAP values heatmap (Outcome=0, first 100 patients)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("heatmap_outcome0.svg", format="svg")  # 保存为 SVG
# plt.show()