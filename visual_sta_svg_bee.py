import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# ========== 1. 读取数据 ==========
# SHAP 值文件（包含特征列和 outcome 列）
shap_df = pd.read_csv("shap_value/shap_static_values_named.csv")
# 患者原始特征文件（用于获取特征值颜色）
patient_df = pd.read_csv("shap_value/cohort_final_static_named.csv")

# 将 outcome 合并到 shap_df（若文件中已有 outcome 则无需此步）
patient_outcome = patient_df[['hadm_id', 'outcome_21d']]
shap_df = shap_df.merge(patient_outcome, on='hadm_id', how='left')
shap_df.rename(columns={'outcome_21d': 'outcome'}, inplace=True)

# ========== 2. 提取特征列和 SHAP 值矩阵 ==========
feature_cols = [c for c in shap_df.columns if c not in ['hadm_id', 'outcome']]
shap_values = shap_df[feature_cols].values  # (样本数, 特征数)

# ========== 3. 计算平均绝对 SHAP 值并排序，选择 Top20 ==========
mean_abs_shap = np.abs(shap_values).mean(axis=0)  # 按特征计算平均绝对值
sorted_idx = np.argsort(mean_abs_shap)[::-1]      # 从大到小排序
top20_idx = sorted_idx[:20]                       # 前20个索引
top20_features = [feature_cols[i] for i in top20_idx]
top20_shap_values = shap_values[:, top20_idx]

# ========== 4. 准备原始特征矩阵 X（顺序与 top20_features 一致） ==========
# 确保 X 的样本顺序与 shap_df 完全一致
X_df = patient_df.set_index('hadm_id').loc[shap_df['hadm_id']]
X = X_df[feature_cols].values
# 提取 Top20 特征对应的原始特征值
X_top20 = X[:, top20_idx]

# ========== 5. 绘制蜂群图 ==========
plt.figure(figsize=(12, 8))
shap.summary_plot(top20_shap_values, X_top20, feature_names=top20_features,
                  plot_type='dot', show=False)   # 关闭自动显示
plt.tight_layout()
plt.savefig("beeswarm_plot_top20.svg", format="svg")
plt.close()

print("✅ SHAP Beeswarm plot (Top20 by mean |SHAP|) saved as beeswarm_plot_top20.svg")