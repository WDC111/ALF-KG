import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 数据准备（与你的原始代码一致）
# =========================
# 读取数据
shap_df = pd.read_csv("shap_last_hour_49_named.csv")
#shap_df = pd.read_csv("shap_static_values_named.csv")
patient_df = pd.read_csv("cohort_final_static_named.csv")

# 合并 outcome
shap_df = shap_df.merge(
    patient_df[['hadm_id', 'outcome_21d']],
    on='hadm_id',
    how='left'
).rename(columns={'outcome_21d': 'outcome'})

# 指定患者
TARGET_HADM_ID = 28834003
TOP_K = 8

patient_shap = shap_df[shap_df['hadm_id'] == TARGET_HADM_ID]
assert len(patient_shap) == 1, "该 hadm_id 不唯一或不存在"

feature_cols = [c for c in shap_df.columns if c not in ['hadm_id', 'outcome']]
shap_values = patient_shap[feature_cols].iloc[0]

patient_features = (
    patient_df.set_index('hadm_id')
    .loc[TARGET_HADM_ID, feature_cols]
)

# 构建解释 DataFrame
explain_df = pd.DataFrame({
    'Feature': feature_cols,
    'SHAP_value': shap_values.values,
    'Feature_value': patient_features.values
})

# 删除特征值为 NaN 的行
explain_df = explain_df.dropna(subset=['Feature_value'])

# 取绝对值最大的 TOP_K 个特征
explain_df['abs_SHAP'] = explain_df['SHAP_value'].abs()
explain_df = explain_df.sort_values('abs_SHAP', ascending=False).head(TOP_K)

# 可选：打印查看
print(explain_df[['Feature', 'SHAP_value', 'Feature_value']])

# =========================
# 2. 雷达图绘制（无径向刻度，特征旁标注原始 SHAP 值）
# =========================
# 提取绘图数据
labels = explain_df['Feature'].tolist()
shap_vals = explain_df['SHAP_value'].values   # 原始 SHAP 值（可能全负）

N = len(labels)

# 角度（等分圆周，从正上方开始顺时针）
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

# 闭合数据（首尾相连）
angles_closed = np.concatenate([angles, [angles[0]]])
shap_vals_closed = np.concatenate([shap_vals, [shap_vals[0]]])
labels_closed = labels + [labels[0]]

# -------------------------
# 2.1 平移 SHAP 值，使最小值为 0（雷达图半径不能为负）
# -------------------------
min_val = shap_vals_closed.min()
offset = abs(min_val)           # 平移量
radius_vals = shap_vals_closed + offset   # 平移后的半径（全 ≥ 0）
zero_radius = offset            # SHAP=0 对应的半径

# -------------------------
# 2.2 创建极坐标子图
# -------------------------
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 绘制主轮廓
ax.plot(angles_closed, radius_vals, linewidth=2, color='black')
ax.fill(angles_closed, radius_vals, alpha=0.2, color='steelblue')

# 绘制 SHAP=0 的参考圆（虚线）
theta = np.linspace(0, 2 * np.pi, 200)
ax.plot(theta, [zero_radius] * len(theta), linestyle='--', linewidth=1.5, color='gray', label='SHAP = 0')

# -------------------------
# 2.3 隐藏径向刻度（不显示圈上的数值）
# -------------------------
ax.set_yticklabels([])          # 隐藏径向数字
ax.set_yticks([])               # 可选：同时隐藏刻度线

# 也可以保留刻度线但隐藏标签，上面已经做了。若连刻度线都不要：
# ax.set_yscale('function', functions=(lambda x: x, lambda x: x))
# 但保持简洁即可。

# -------------------------
# 2.4 设置角度轴（特征名称）
# -------------------------
ax.set_theta_offset(np.pi / 2)   # 从正上方（12点钟）开始
ax.set_theta_direction(-1)       # 顺时针方向

ax.set_xticks(angles)
ax.set_xticklabels(labels, fontsize=10)

# 微调角度标签的对齐方式，避免重叠
for label, angle in zip(ax.get_xticklabels(), angles):
    label.set_horizontalalignment('center')
    # 可选：根据角度旋转标签，使阅读更舒服
    # if 0 <= angle < np.pi/2 or 3*np.pi/2 <= angle < 2*np.pi:
    #     label.set_rotation(angle * 180/np.pi - 90)
    # else:
    #     label.set_rotation(angle * 180/np.pi + 90)
    label.set_rotation(1)   # 简单保持水平

# -------------------------
# 2.5 在每个特征轴外侧标注原始 SHAP 值
# -------------------------
# 确定一个放置文本的半径：略大于最大平移半径，放在图形外侧
max_radius = radius_vals.max()
text_radius = 1 * (zero_radius)   # 向外偏移15%的幅度

for i, (angle, shap_val) in enumerate(zip(angles, shap_vals)):
    # 计算文本坐标（极坐标转直角坐标，matplotlib 自动处理，直接使用极坐标）
    # 在极坐标中，我们可以直接用 ax.text() 指定半径和角度
    color = 'red' if shap_val > 0 else 'blue'   # 正红负蓝
    ax.text(angle, text_radius,
            f'{shap_val:.3f}',
            ha='center', va='center',
            fontsize=9, color=color,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

# -------------------------
# 2.6 标题和图例
# -------------------------
ax.set_title(
    f"Patient-level SHAP Radar Plot (hadm_id={TARGET_HADM_ID})\nInner: Decrease risk | Outer: Increase risk",
    fontsize=13, pad=20
)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
ax.grid(True)

plt.tight_layout()

# 保存为 SVG（可选）
plt.savefig(f'patient_{TARGET_HADM_ID}_shap_radar_no_ticks.svg', format='svg', bbox_inches='tight')

plt.show()