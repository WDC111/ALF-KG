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

OUT_LSTM = "lstm_48h_l1_features.csv"

# =========================
# 2. 读取 cohort hadm_id
# =========================
cohort = pd.read_csv(COHORT_FILE, usecols=["hadm_id"])
hadm_set = set(cohort["hadm_id"].astype(int))
print(f"✅ cohort hadm_id 数量: {len(hadm_set)}")

# =========================
# 3. 读取 admission 时间
# =========================
admission = pd.read_csv(
    ADMISSION_FILE,
    usecols=["hadm_id", "admittime"],
    parse_dates=["admittime"]
)
admission = admission[admission["hadm_id"].isin(hadm_set)]
admit_time_dict = dict(
    zip(admission["hadm_id"], admission["admittime"])
)

# =========================
# 4. 读取 L1 特征 itemid
# =========================
features = pd.read_csv(FEATURE_FILE)
itemids = features["feature"].astype(int).tolist()
itemid_to_idx = {itemid: i for i, itemid in enumerate(itemids)}
n_features = len(itemids)

print(f"✅ L1 特征数量: {n_features}")

# =========================
# 5. 初始化结果结构（稀疏更新）
# hadm_id -> (48, n_features)
# =========================
lstm_data = {}

def get_patient_matrix(hadm_id):
    if hadm_id not in lstm_data:
        lstm_data[hadm_id] = np.zeros((48, n_features), dtype=np.float32)
    return lstm_data[hadm_id]

# =========================
# 6. 顺序扫描 labevents（关键提速）
# =========================
chunksize = 2_000_000  # 可根据内存调整

usecols = ["hadm_id", "itemid", "charttime", "valuenum"]

for chunk in tqdm(
    pd.read_csv(
        LABEVENTS_FILE,
        usecols=usecols,
        parse_dates=["charttime"],
        chunksize=chunksize
    ),
    desc="Scanning labevents"
):
    # 只保留 cohort + L1 特征
    chunk = chunk[
        chunk["hadm_id"].isin(hadm_set) &
        chunk["itemid"].isin(itemids)
    ]

    if chunk.empty:
        continue

    # 计算 hour_from_admit
    chunk["admittime"] = chunk["hadm_id"].map(admit_time_dict)
    chunk = chunk.dropna(subset=["admittime"])

    chunk["hour"] = (
        (chunk["charttime"] - chunk["admittime"])
        .dt.total_seconds() // 3600
    ).astype(int)

    # 只保留 0–47 小时
    chunk = chunk[(chunk["hour"] >= 0) & (chunk["hour"] < 48)]

    # === 核心：逐条更新矩阵（只做赋值，不查表）
    for hadm_id, itemid, hour, val in zip(
        chunk["hadm_id"],
        chunk["itemid"],
        chunk["hour"],
        chunk["valuenum"]
    ):
        if pd.isna(val):
            continue

        mat = get_patient_matrix(hadm_id)
        mat[hour, itemid_to_idx[itemid]] = float(val)

# =========================
# 7. 前向填充（0 → 上一小时）
# =========================
rows = []

for hadm_id, mat in tqdm(lstm_data.items(), desc="Forward filling"):
    last = np.zeros(n_features)

    for h in range(48):
        mask = mat[h] == 0
        mat[h][mask] = last[mask]
        last = mat[h].copy()

        row = {
            "hadm_id": hadm_id,
            "hour": h
        }
        for i, itemid in enumerate(itemids):
            row[f"feat_{itemid}"] = mat[h, i]

        rows.append(row)

# =========================
# 8. 输出 CSV
# =========================
df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_LSTM, index=False)

print(f"✅ LSTM 输入数据已生成: {OUT_LSTM}")
print(f"Shape ≈ ({df_out.shape[0]}, {df_out.shape[1]})")
