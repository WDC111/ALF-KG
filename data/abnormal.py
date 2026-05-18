import pandas as pd
import numpy as np
from tqdm import tqdm

# =========================
# 1. 文件路径
# =========================
LABEVENTS_FILE = "labevents.csv"
ADMISSION_FILE = "admissions.csv"
COHORT_FILE = "cohort_final_static.csv"
FEATURE_FILE = "selected_features_l1_with_coef1.csv"

OUTPUT_FILE = "first_abnormal_hour_l1_features.csv"

# =========================
# 2. 读取 cohort hadm_id
# =========================
cohort = pd.read_csv(COHORT_FILE, usecols=["hadm_id"])
hadm_set = set(cohort["hadm_id"].dropna().astype(int))
print(f"✅ cohort hadm_id 数量: {len(hadm_set)}")

# =========================
# 3. 读取 L1 特征 itemid
# =========================
features = pd.read_csv(FEATURE_FILE)
itemids = set(features["feature"].astype(int))
print(f"✅ L1 特征数量: {len(itemids)}")

# =========================
# 4. 读取 admissions（入院时间）
# =========================
admission = pd.read_csv(
    ADMISSION_FILE,
    usecols=["hadm_id", "admittime"],
    parse_dates=["admittime"]
)

admit_time_map = dict(
    zip(admission["hadm_id"], admission["admittime"])
)

# =========================
# 5. 用于记录首次异常
# =========================
# key = (hadm_id, itemid) -> first_hour
first_abnormal = {}

# =========================
# 6. 分块扫描 labevents（核心）
# =========================
chunksize = 1_000_000

reader = pd.read_csv(
    LABEVENTS_FILE,
    usecols=["hadm_id", "itemid", "charttime", "flag"],
    parse_dates=["charttime"],
    chunksize=chunksize
)

print("🚀 开始扫描 labevents.csv")

for chunk in tqdm(reader, desc="Scanning labevents"):
    # 基础过滤
    chunk = chunk[
        chunk["hadm_id"].isin(hadm_set) &
        chunk["itemid"].isin(itemids) &
        (chunk["flag"] == "abnormal")
    ]

    if chunk.empty:
        continue

    # 计算 hour_from_admit
    chunk["admittime"] = chunk["hadm_id"].map(admit_time_map)
    chunk = chunk.dropna(subset=["admittime"])

    chunk["hour_from_admit"] = (
        (chunk["charttime"] - chunk["admittime"])
        .dt.total_seconds() // 3600
    ).astype(int)

    # 限制 0–47 小时
    chunk = chunk[
        (chunk["hour_from_admit"] >= 0) &
        (chunk["hour_from_admit"] < 48)
    ]

    # 按时间排序，确保“首次”
    chunk = chunk.sort_values("hour_from_admit")

    # 逐行更新首次异常
    for row in chunk.itertuples(index=False):
        key = (row.hadm_id, row.itemid)
        if key not in first_abnormal:
            first_abnormal[key] = row.hour_from_admit

# =========================
# 7. 输出 CSV
# =========================
records = [
    {
        "hadm_id": k[0],
        "itemid": k[1],
        "first_abnormal_hour": v
    }
    for k, v in first_abnormal.items()
]

df_out = pd.DataFrame(records)
df_out.to_csv(OUTPUT_FILE, index=False)

print("✅ 首次异常特征时间已生成")
print(f"📄 文件: {OUTPUT_FILE}")
print(f"📊 记录条数: {len(df_out)}")
