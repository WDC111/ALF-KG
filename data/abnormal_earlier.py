import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
from collections import defaultdict

# =========================
# 1. 读取首次异常数据
# =========================
INPUT_FILE = "first_abnormal_hour_l1_features.csv"
OUTPUT_FILE = "feature_pair_abnormal_order_counts.csv"

df = pd.read_csv(INPUT_FILE)

# 确保类型
df["hadm_id"] = df["hadm_id"].astype(int)
df["itemid"] = df["itemid"].astype(int)
df["first_abnormal_hour"] = df["first_abnormal_hour"].astype(int)

print("✅ 输入数据行数:", len(df))
print("✅ 患者数:", df["hadm_id"].nunique())
print("✅ 特征数:", df["itemid"].nunique())

# =========================
# 2. 按患者分组
# =========================
grouped = df.groupby("hadm_id")

# =========================
# 3. 结果容器
# key = (f1, f2)
# value = [count1, count2, count3]
# =========================
pair_counts = defaultdict(lambda: [0, 0, 0])

# =========================
# 4. 核心统计逻辑
# =========================
print("🚀 开始统计特征对的异常先后顺序")

for hadm_id, g in tqdm(grouped, desc="Processing patients"):
    # 当前患者发生过异常的特征及其时间
    feats = g[["itemid", "first_abnormal_hour"]].values

    if len(feats) < 2:
        continue  # 至少要两个特征才有“顺序”

    # 枚举特征对（组合）
    for (f1, t1), (f2, t2) in combinations(feats, 2):
        # 统一 key 顺序，避免 (A,B) 和 (B,A) 重复
        if f1 < f2:
            key = (f1, f2)
            if t1 < t2:
                pair_counts[key][0] += 1  # f1 先于 f2
            elif t1 > t2:
                pair_counts[key][1] += 1  # f1 后于 f2
            else:
                pair_counts[key][2] += 1  # 同时
        else:
            key = (f2, f1)
            if t2 < t1:
                pair_counts[key][0] += 1
            elif t2 > t1:
                pair_counts[key][1] += 1
            else:
                pair_counts[key][2] += 1

# =========================
# 5. 转为 DataFrame
# =========================
records = []
for (f1, f2), (c1, c2, c3) in pair_counts.items():
    records.append({
        "feature1": f1,
        "feature2": f2,
        "count1": c1,  # feature1 先于 feature2
        "count2": c2,  # feature1 后于 feature2
        "count3": c3   # 同时发生
    })

df_out = pd.DataFrame(records)

# =========================
# 6. 保存结果
# =========================
df_out.to_csv(OUTPUT_FILE, index=False)

print("✅ 统计完成")
print("📄 输出文件:", OUTPUT_FILE)
print("📊 特征对数量:", len(df_out))
