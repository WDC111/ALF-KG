#!/usr/bin/env python3
# coding: utf-8

"""
patient_subgraph_node2vec_pt.py

用Node2Vec生成每个患者的子图嵌入，并保存为单独 pt 文件
每个患者一个 pt 文件，文件名为 hadm_id.pt
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
import torch

# -----------------------------
# 文件路径
# -----------------------------
GRAPH_FILE = "target_connectivity_graph.edgelist"
PATIENT_FILE = "patient_clinical_filt.csv"
MAPPING_FILE = "mapping.csv"
OUTCOME_NODE = "CN36909145"

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 1️⃣ 读取图和患者数据
# -----------------------------
print("读取图数据...")
G = nx.Graph()

with open(GRAPH_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 4:
            u, v, weight, extra = parts[:4]
            G.add_edge(u, v, weight=float(weight), extra=float(extra))
        else:
            print("警告: 边格式异常:", line)

print(f"图中节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

print("读取患者数据...")
patient_df = pd.read_csv(PATIENT_FILE)
mapping_df = pd.read_csv(MAPPING_FILE)

nodes = list(G.nodes())

# -----------------------------
# 2️⃣ 为每个患者赋值节点
# -----------------------------
print("构建患者节点值矩阵...")
patient_values = pd.DataFrame(0, index=patient_df['hadm_id'], columns=nodes)

for _, row in mapping_df.iterrows():
    feature = row['feature']
    node = row['mapping']
    if feature in patient_df.columns and node in patient_values.columns:
        patient_values[node] = patient_df[feature].values

# outcome节点赋值（可选）
if 'outcome_21d' in patient_df.columns:
    patient_values[OUTCOME_NODE] = patient_df['outcome_21d'].values


# -----------------------------
# 2.5 统计患者子图规模
# -----------------------------
print("开始统计患者子图规模...")

subgraph_nodes = []
subgraph_edges = []

threshold = 5   # 子图过小阈值（你论文可以写这个）

for idx, row in patient_values.iterrows():

    # 找到该患者激活的节点（有值的节点）
    active_nodes = row[row != 0].index.tolist()

    if len(active_nodes) == 0:
        subgraph_nodes.append(0)
        subgraph_edges.append(0)
        continue

    # 一阶邻居子图
    nodes_1hop = set(active_nodes)
    for n in active_nodes:
        nodes_1hop.update(G.neighbors(n))

    subG = G.subgraph(nodes_1hop).copy()

    # 如果子图太小，扩展到二阶邻居
    if subG.number_of_nodes() < threshold:
        nodes_2hop = set(nodes_1hop)
        for n in nodes_1hop:
            nodes_2hop.update(G.neighbors(n))
        subG = G.subgraph(nodes_2hop).copy()

    subgraph_nodes.append(subG.number_of_nodes())
    subgraph_edges.append(subG.number_of_edges())

# 保存统计
stats_df = pd.DataFrame({
    "hadm_id": patient_values.index,
    "nodes": subgraph_nodes,
    "edges": subgraph_edges
})

stats_df.to_csv("subgraph_statistics.csv", index=False)

# -----------------------------
# 统计指标
# -----------------------------
nodes_arr = np.array(subgraph_nodes)
edges_arr = np.array(subgraph_edges)

print("\n子图统计结果：")
print("平均节点数:", round(nodes_arr.mean(), 2))
print("平均边数:", round(edges_arr.mean(), 2))

print("\n节点数分布：")
print("P25:", np.percentile(nodes_arr, 25))
print("P50:", np.percentile(nodes_arr, 50))
print("P75:", np.percentile(nodes_arr, 75))

print("\n最大节点数:", nodes_arr.max())
print("最小节点数:", nodes_arr.min())
# -----------------------------
# 3️⃣ 训练 Node2Vec 节点嵌入
# -----------------------------
print("训练 Node2Vec 节点嵌入...")
node2vec = Node2Vec(
    G,
    dimensions=128,
    walk_length=10,
    num_walks=100,
    workers=4,
    weight_key="weight",
    seed=42
)
model = node2vec.fit(window=5, min_count=1, batch_words=4)

# -----------------------------
# 4️⃣ 生成每个患者的嵌入并保存
# -----------------------------
print("生成并保存每个患者的子图嵌入...")
for idx, row in patient_values.iterrows():
    emb_list = []
    for node in nodes:
        node_vec = model.wv[node]
        node_value = row[node]
        emb_list.append(node_vec * node_value)  # 节点值加权
    patient_emb = np.mean(emb_list, axis=0)
    patient_emb_tensor = torch.tensor(patient_emb, dtype=torch.float)

    pt_file = os.path.join(OUT_DIR, f"{idx}.pt")
    torch.save(patient_emb_tensor, pt_file)

print(f"完成 {len(patient_values)} 个患者的嵌入生成并保存到 {OUT_DIR}")
