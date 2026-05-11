import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from collections import defaultdict
from tqdm import tqdm
import gc
import os

# =========================
# 1. 读取文件
# =========================
patient_df = pd.read_csv("../patient_clinical_filt.csv")
mapping_df = pd.read_csv("../mapping.csv")
relations_df = pd.read_csv(
    "../ONE-HOP/Processed_Relations.txt",
    sep="\t",
    names=["head", "relation", "tail"],
    header=0
)
abnormal_df = pd.read_csv("./first_abnormal_hour_l1_features.csv")

# =========================
# 2. 构建映射字典
# =========================
feature2cid = dict(zip(mapping_df["feature"].astype(str), mapping_df["mapping"]))

# =========================
# 3. 构建 BIOS 邻接表
# =========================
bios_adj = defaultdict(set)
bios_edges = []

for _, row in relations_df.iterrows():
    h, r, t = row["head"], row["relation"], row["tail"]
    bios_adj[h].add(t)
    bios_adj[t].add(h)  # 无向
    bios_edges.append((h, r, t))


# =========================
# 4. 辅助函数
# =========================
def get_patient_active_cids(patient_row, feature2cid):
    active = {}
    for feat, cid in feature2cid.items():
        if feat not in patient_row:
            continue
        val = patient_row[feat]
        if pd.isna(val):
            continue
        active[cid] = float(val)
    return active  # cid -> value


def build_patient_subgraph(active_cids, bios_adj, bios_edges, hop=1):
    nodes = set(active_cids.keys())
    frontier = set(nodes)
    for _ in range(hop):
        new_nodes = set()
        for cid in frontier:
            new_nodes |= bios_adj.get(cid, set())
        new_nodes -= nodes
        nodes |= new_nodes
        frontier = new_nodes
    # 边
    edges = []
    for h, r, t in bios_edges:
        if h in nodes and t in nodes:
            edges.append((h, r, t))
    return nodes, edges


def create_pyg_data(hadm_id, nodes, edges, active_cids, abnormal_hour, feature2cid):
    """
    nodes: set(cid)
    edges: list of (h, r, t)
    active_cids: dict(cid -> value)
    abnormal_hour: dict(feature -> first_abnormal_hour)
    """
    cid2idx = {cid: idx for idx, cid in enumerate(nodes)}

    # 节点特征矩阵 x [num_nodes, 1]
    x = []
    for cid in nodes:
        if cid in active_cids:
            x.append([active_cids[cid]])
        else:
            x.append([0.0])
    x = torch.tensor(x, dtype=torch.float)

    # 边索引 edge_index [2, num_edges]
    edge_index = []
    edge_weight = []
    for h, r, t in edges:
        edge_index.append([cid2idx[h], cid2idx[t]])
        # 默认权重 1.0
        w = 1.0
        # 异常小时增强权重
        h_feat = next((k for k, v in feature2cid.items() if v == h), None)
        t_feat = next((k for k, v in feature2cid.items() if v == t), None)
        if h_feat in abnormal_hour and t_feat in abnormal_hour:
            diff = abnormal_hour[h_feat] - abnormal_hour[t_feat]
            if diff < 0:
                w += 0.5
            elif diff > 0:
                w += 0.5
            else:
                w += 1.0
        edge_weight.append(w)
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.tensor([], dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    data.hadm_id = hadm_id
    return data


# =========================
# 5. 生成患者子图
# =========================
output_dir = "./patient_graph_data"
os.makedirs(output_dir, exist_ok=True)

for _, row in tqdm(patient_df.iterrows(), total=len(patient_df), desc="Building patient graphs"):
    hadm_id = row["hadm_id"]
    active_cids = get_patient_active_cids(row, feature2cid)
    if len(active_cids) == 0:
        continue

    patient_abnormal = abnormal_df[abnormal_df["hadm_id"] == hadm_id]
    abnormal_hour = dict(zip(patient_abnormal["itemid"].astype(str), patient_abnormal["first_abnormal_hour"]))

    nodes, edges = build_patient_subgraph(active_cids, bios_adj, bios_edges, hop=1)
    if len(nodes) == 0:
        continue

    data = create_pyg_data(hadm_id, nodes, edges, active_cids, abnormal_hour, feature2cid)

    # 保存为单个文件 per patient，节省内存
    torch.save(data, os.path.join(output_dir, f"{hadm_id}.pt"))

    del nodes, edges, active_cids, patient_abnormal, abnormal_hour, data
    gc.collect()

print("✅ 患者子图嵌入生成完成")
print("文件保存在:", output_dir)
