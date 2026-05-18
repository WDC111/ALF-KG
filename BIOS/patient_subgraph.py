#!/usr/bin/env python3
# coding: utf-8

"""
quick_gnn_explainer.py - 简化版本
"""

import os
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from torch_geometric.explain import GNNExplainer
import pickle

# 1. 加载图和患者数据
G = nx.read_edgelist("target_connectivity_graph.edgelist", data=[('weight', float), ('extra', float)])
patient_df = pd.read_csv("patient_clinical_filt.csv")


# 2. 创建简单的GNN模型
class SimpleGNN(nn.Module):
    def __init__(self):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 8)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# 3. 转换为PyG格式
pyg_data = from_networkx(G)


# 4. 为每个患者创建特征
def create_patient_features(patient_row, nodes):
    # 假设patient_row是患者的特征行
    features = torch.zeros(len(nodes), 1)
    # 这里根据您的数据逻辑填充特征
    return features


# 5. 初始化解释器
def explain_patient_gnn(model, patient_features, edge_index):
    # 设置模型为评估模式
    model.eval()

    # 创建GNNExplainer
    explainer = GNNExplainer(model, epochs=100, return_type='raw')

    # 生成解释
    node_feat_mask, edge_mask = explainer.explain_graph(
        patient_features,
        edge_index
    )

    return node_feat_mask, edge_mask


# 6. 使用示例
if __name__ == "__main__":
    # 初始化模型
    model = SimpleGNN()

    # 选择第一个患者
    patient_features = create_patient_features(patient_df.iloc[0], list(G.nodes()))

    # 生成解释
    node_mask, edge_mask = explain_patient_gnn(
        model,
        patient_features,
        pyg_data.edge_index
    )

    print("节点重要性掩码形状:", node_mask.shape)
    print("边重要性掩码形状:", edge_mask.shape)

    # 保存解释结果
    explanation = {
        'patient_id': patient_df.iloc[0]['hadm_id'],
        'node_importance': node_mask.detach().numpy(),
        'edge_importance': edge_mask.detach().numpy(),
        'important_nodes': list(G.nodes())[:10],  # 示例
        'important_edges': list(G.edges())[:10]  # 示例
    }

    with open("patient_explanation.pkl", "wb") as f:
        pickle.dump(explanation, f)

    print("解释结果已保存到 patient_explanation.pkl")