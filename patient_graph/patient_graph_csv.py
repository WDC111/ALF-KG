import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import gc

# 读取文件
patient_df = pd.read_csv("../patient_clinical_filt.csv")
mapping_df = pd.read_csv("../mapping.csv")
relations_df = pd.read_csv("../ONE-HOP/Processed_Relations.txt", sep="\t", names=["head","relation","tail"], header=0)
first_abnormal_df = pd.read_csv("./first_abnormal_hour_l1_features.csv")

# 构建映射和邻接表
feature2cid = dict(zip(mapping_df["feature"].astype(str), mapping_df["mapping"]))
bios_adj = defaultdict(set)
bios_edges = []
for _, row in relations_df.iterrows():
    h, r, t = row["head"], row["relation"], row["tail"]
    bios_adj[h].add(t)
    bios_adj[t].add(h)
    bios_edges.append((h, r, t))

# 输出文件
node_file = "patient_graph_nodes.csv"
edge_file = "patient_graph_edges.csv"

# 初始化 CSV，写入列名
pd.DataFrame(columns=["hadm_id","cid","node_type","value"]).to_csv(node_file, index=False)
pd.DataFrame(columns=["hadm_id","head","relation","tail","weight"]).to_csv(edge_file, index=False)

def get_patient_active_cids(patient_row, feature2cid):
    active = {}
    for feat, cid in feature2cid.items():
        if feat in patient_row and not pd.isna(patient_row[feat]):
            active[cid] = float(patient_row[feat])
    return active

def build_patient_subgraph(active_cids, bios_adj, bios_edges, hop=1):
    nodes = set(active_cids.keys())
    frontier = set(nodes)
    for _ in range(hop):
        new_nodes = set()
        for cid in frontier:
            new_nodes |= bios_adj.get(cid,set())
        new_nodes -= nodes
        nodes |= new_nodes
        frontier = new_nodes
    edges = []
    for h,r,t in bios_edges:
        if h in nodes and t in nodes:
            edges.append((h,r,t))
    return nodes, edges

# 按患者处理
for _, row in tqdm(patient_df.iterrows(), total=len(patient_df), desc="Building patient graphs"):
    hadm_id = row["hadm_id"]
    active_cids = get_patient_active_cids(row, feature2cid)
    if len(active_cids)==0:
        continue
    nodes, edges = build_patient_subgraph(active_cids, bios_adj, bios_edges, hop=1)

    # 获取异常小时信息
    patient_abnormal = first_abnormal_df[first_abnormal_df["hadm_id"]==hadm_id]
    abnormal_dict = dict(zip(patient_abnormal["itemid"].astype(str), patient_abnormal["first_abnormal_hour"]))

    # 节点写入
    node_rows = []
    for cid in nodes:
        if cid in active_cids:
            node_type = "clinical"
            value = active_cids[cid]
        else:
            node_type = "knowledge"
            value = 0.0
        node_rows.append({"hadm_id":hadm_id,"cid":cid,"node_type":node_type,"value":value})
    pd.DataFrame(node_rows).to_csv(node_file, mode="a", header=False, index=False)

    # 边写入
    edge_rows = []
    for h,r,t in edges:
        # 默认权重 1.0
        w = 1.0
        # 如果两端都是临床节点且有异常小时信息
        h_feat = {k:v for k,v in feature2cid.items() if v==h}
        t_feat = {k:v for k,v in feature2cid.items() if v==t}
        if len(h_feat)>0 and len(t_feat)>0:
            h_item = list(h_feat.keys())[0]
            t_item = list(t_feat.keys())[0]
            if h_item in abnormal_dict and t_item in abnormal_dict:
                if abnormal_dict[h_item] < abnormal_dict[t_item]:
                    w += 0.5  # h 先于 t
                elif abnormal_dict[h_item] > abnormal_dict[t_item]:
                    w -= 0.3  # h 后于 t
                else:
                    w += 0.2  # 同时
        edge_rows.append({"hadm_id":hadm_id,"head":h,"relation":r,"tail":t,"weight":w})
    pd.DataFrame(edge_rows).to_csv(edge_file, mode="a", header=False, index=False)

    # 清理内存
    del nodes, edges, node_rows, edge_rows, patient_abnormal, abnormal_dict, active_cids
    gc.collect()
