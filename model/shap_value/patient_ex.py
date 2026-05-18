import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# 1. 读取子图
# -------------------------
graph_file = "target_connectivity_graph.edgelist"
G = nx.read_edgelist(graph_file, delimiter='\t', data=[('weight', float), ('length', int)])

# -------------------------
# 2. 读取 mapping.csv（已替换指标名）
# -------------------------
mapping_df = pd.read_csv("mapping_named.csv")
feature2node = dict(zip(mapping_df['feature'].str.strip(), mapping_df['mapping'].str.strip()))

# -------------------------
# 3. 读取患者 SHAP 值
# -------------------------
hadm_id = 29297163
target_node = 'CN36909145'

shap_static_all = pd.read_csv("shap_static_values_named.csv")
shap_time_all   = pd.read_csv("shap_last_hour_49_named.csv")

shap_static = shap_static_all[shap_static_all['hadm_id']==hadm_id].copy()
shap_time   = shap_time_all[shap_time_all['hadm_id']==hadm_id].copy()

# -------------------------
# 4. 转长表 + 删除NaN和0
# -------------------------
static_cols = [c for c in shap_static.columns if c not in ['hadm_id','outcome']]
shap_static_long = shap_static.melt(id_vars=['hadm_id'], value_vars=static_cols,
                                    var_name='Feature', value_name='SHAP_static')
shap_static_long = shap_static_long.dropna(subset=['SHAP_static'])
shap_static_long = shap_static_long[shap_static_long['SHAP_static'] != 0]

time_cols = [c for c in shap_time.columns if c not in ['hadm_id','hour']]
shap_time_long = shap_time.melt(id_vars=['hadm_id'], value_vars=time_cols,
                                var_name='Feature', value_name='SHAP_time')
shap_time_long = shap_time_long.dropna(subset=['SHAP_time'])
shap_time_long = shap_time_long[shap_time_long['SHAP_time'] != 0]

shap_long = pd.merge(shap_static_long, shap_time_long, on=['hadm_id','Feature'], how='outer')

# -------------------------
# 5. 映射 Feature -> Node
# -------------------------
shap_long['Feature'] = shap_long['Feature'].str.strip()
shap_long['node'] = shap_long['Feature'].map(feature2node)
shap_long = shap_long.dropna(subset=['node'])

# -------------------------
# 6. 过滤掉静态特征值为空的节点
# -------------------------
patient_static_all = pd.read_csv("cohort_final_static_named.csv").set_index('hadm_id')
def check_feature_value(row):
    feature = row['Feature']
    if feature in patient_static_all.columns:
        val = patient_static_all.at[hadm_id, feature]
        return pd.notna(val)
    else:
        return True
shap_long = shap_long[shap_long.apply(check_feature_value, axis=1)]

# -------------------------
# 7. 构造节点信息
# -------------------------
node_static = dict(zip(shap_long['node'], shap_long['SHAP_static'].fillna(0)))
node_time   = dict(zip(shap_long['node'], shap_long['SHAP_time'].fillna(0)))
node_featval = {}
for node, feature in zip(shap_long['node'], shap_long['Feature']):
    if feature in patient_static_all.columns:
        node_featval[node] = patient_static_all.at[hadm_id, feature]
    else:
        node_featval[node] = np.nan

if target_node not in node_static:
    node_static[target_node] = 1.0
    node_time[target_node] = 1.0
    node_featval[target_node] = 1.0

# -------------------------
# 8. 只保留有边的节点
# -------------------------
nodes_in_edges = set([n for e in G.edges() for n in e])
nodes_to_keep = [n for n in node_static if n in nodes_in_edges or n==target_node]
G_sub = G.subgraph(nodes_to_keep).copy()

# ======================= 关键路径分析（仅保留最短路径） =======================
# 8.1 选择 SHAP 绝对值最大的前 10 个节点（排除目标节点）
nodes_with_shap = [(n, abs(node_static.get(n, 0))) for n in G_sub.nodes() if n != target_node]
nodes_with_shap.sort(key=lambda x: x[1], reverse=True)
top10_nodes = [n for n, _ in nodes_with_shap[:10] if n in G_sub.nodes()]

print(f"Top-10 SHAP absolute value nodes (excluding target): {top10_nodes}")

# 8.2 定义路径有效性检查（中间节点 SHAP 非零）
def is_valid_node_for_path(node):
    if node == target_node:
        return True
    return abs(node_static.get(node, 0)) > 1e-6

def find_simple_paths(G, source, target, cutoff=5):
    """返回所有长度 <= cutoff 的简单路径，且中间节点 SHAP 非零"""
    all_paths = []
    for path in nx.all_simple_paths(G, source, target, cutoff=cutoff):
        if all(is_valid_node_for_path(n) for n in path[1:-1]):
            all_paths.append(path)
    return all_paths

# 8.3 对每个 top-10 节点，只保留总长度最短的路径（若多条等长，取第一条）
path_scores_raw = []  # 每个元素: (path, semantic, temporal, contrib)
for src in top10_nodes:
    all_paths = find_simple_paths(G_sub, src, target_node, cutoff=5)
    if not all_paths:
        continue
    # 计算每条路径的总长度（边 length 之和）
    path_lengths = []
    for path in all_paths:
        total_len = sum(G_sub[path[i]][path[i+1]]['length'] for i in range(len(path)-1))
        path_lengths.append((path, total_len))
    # 按总长度排序，取最短的（如果等长，取第一个）
    path_lengths.sort(key=lambda x: x[1])
    shortest_path = path_lengths[0][0]
    # 计算该最短路径的语义、时间、贡献分数
    sem = sum(G_sub[shortest_path[i]][shortest_path[i+1]]['weight'] for i in range(len(shortest_path)-1))
    total_len = sum(G_sub[shortest_path[i]][shortest_path[i+1]]['length'] for i in range(len(shortest_path)-1))
    tmp = 1.0 / (total_len + 1)
    contrib = sum(abs(node_static.get(n, 0)) for n in shortest_path[:-1])  # 不包含目标节点
    path_scores_raw.append((shortest_path, sem, tmp, contrib))

if not path_scores_raw:
    print("警告：未找到任何从 top-10 节点到目标节点的有效路径。")
else:
    # 8.4 归一化并计算综合得分（基于所有最短路径）
    semantic_vals = [p[1] for p in path_scores_raw]
    temporal_vals = [p[2] for p in path_scores_raw]
    contrib_vals  = [p[3] for p in path_scores_raw]

    def normalize(arr):
        arr = np.array(arr)
        min_v, max_v = arr.min(), arr.max()
        if max_v - min_v < 1e-9:
            return np.ones_like(arr)
        return (arr - min_v) / (max_v - min_v)

    semantic_norm = normalize(semantic_vals)
    temporal_norm = normalize(temporal_vals)
    contrib_norm  = normalize(contrib_vals)

    w_s = w_t = w_c = 1/3
    # 直接为每条路径计算综合得分（每个源节点只有一条）
    results = []
    highlight_edges = []
    print("\n===== 关键路径（每个源节点的最短路径）=====")
    for idx, (path, sem_raw, tmp_raw, contrib_raw) in enumerate(path_scores_raw):
        combined = w_s * semantic_norm[idx] + w_t * temporal_norm[idx] + w_c * contrib_norm[idx]
        src = path[0]
        # 构建实体链
        entity_chain = []
        for node in path:
            feat_row = shap_long[shap_long['node'] == node]
            if not feat_row.empty:
                feat_name = feat_row['Feature'].values[0]
            else:
                feat_name = node
            entity_chain.append(f"{feat_name}({node})")
        chain_str = " → ".join(entity_chain)
        print(f"源节点 {src}: 得分 = {combined:.4f} (总长度 = {sum(G_sub[path[i]][path[i+1]]['length'] for i in range(len(path)-1))})")
        print(f"  路径: {chain_str}\n")
        results.append({
            'source_node': src,
            'path_nodes': ' -> '.join(path),
            'entity_chain': chain_str,
            'combined_score': combined,
            'semantic_raw': sem_raw,
            'temporal_raw': tmp_raw,
            'contrib_raw': contrib_raw,
            'semantic_norm': semantic_norm[idx],
            'temporal_norm': temporal_norm[idx],
            'contrib_norm': contrib_norm[idx]
        })
        # 收集高亮边
        for i in range(len(path)-1):
            highlight_edges.append((path[i], path[i+1]))

    # 保存结果到 CSV
    if results:
        df_paths = pd.DataFrame(results)
        df_paths.to_csv("key_paths_scores.csv", index=False)
        print("关键路径及得分已保存到 key_paths_scores.csv")

# ======================= 绘图（与之前相同，仅使用 highlight_edges） =======================
node_colors = ['yellow' if n==target_node else 'lightgray' for n in G_sub.nodes()]

edge_styles = []
edge_widths = []
for u,v,d in G_sub.edges(data=True):
    edge_widths.append(d['weight']*2)
    edge_styles.append('dashed' if d['weight'] != 1.0 else 'solid')

labels = {}
for node in G_sub.nodes():
    if node == target_node:
        feat_name = shap_long.loc[shap_long['node']==node, 'Feature'].values
        feat_name = feat_name[0] if len(feat_name)>0 else node
        labels[node] = f"{feat_name}\nVal:0"
    else:
        feat_name = shap_long.loc[shap_long['node']==node, 'Feature'].values
        feat_name = feat_name[0] if len(feat_name)>0 else node
        shap_val = node_static.get(node,0)
        feat_val = node_featval.get(node,'NA')
        labels[node] = f"{feat_name}\nSHAP:{shap_val:.3f}\nVal:{feat_val}"

pos = nx.spring_layout(G_sub, seed=42, k=0.8)

plt.figure(figsize=(14,9.1))
nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, node_size=800, linewidths=1.5)

# 所有边（灰色）
for (u,v,d), style, width in zip(G_sub.edges(data=True), edge_styles, edge_widths):
    nx.draw_networkx_edges(G_sub, pos, edgelist=[(u,v)],
                           style=style, width=width, alpha=0.5,
                           edge_color='gray', arrows=True,
                           arrowstyle='-|>', arrowsize=20,
                           connectionstyle='arc3,rad=0.2')

# 高亮关键路径上的边（红/蓝，加粗）
if highlight_edges:
    for (u,v) in set(highlight_edges):
        shap_u = node_static.get(u, 0)
        if shap_u > 0:
            color = 'red'
        elif shap_u < 0:
            color = 'blue'
        else:
            continue
        weight = G_sub[u][v]['weight']
        style = 'dashed' if weight != 1.0 else 'solid'
        nx.draw_networkx_edges(G_sub, pos, edgelist=[(u,v)],
                               style=style, width=weight*3 + 1,
                               edge_color=color, alpha=0.9,
                               arrows=True, arrowstyle='-|>', arrowsize=20,
                               connectionstyle='arc3,rad=0.2')

nx.draw_networkx_labels(G_sub, pos, labels=labels, font_size=7)
plt.title(f"Patient {hadm_id} SHAP Mapped on Knowledge Subgraph\nTarget node: {target_node} highlighted in yellow\nKey paths: red edges (positive SHAP), blue edges (negative SHAP)", fontsize=12)
plt.axis('off')

svg_filename = f"patient_{hadm_id}_subgraph_shortest_paths.svg"
plt.savefig(svg_filename, format='svg', bbox_inches='tight')
print(f"带最短路径高亮的 SVG 已保存为：{svg_filename}")

plt.show()

# 输出节点信息（不变）
node_info = pd.DataFrame({
    'node': list(G_sub.nodes()),
    'feature': [shap_long.loc[shap_long['node']==n, 'Feature'].values[0] if len(shap_long.loc[shap_long['node']==n, 'Feature'].values)>0 else n for n in G_sub.nodes()],
    'SHAP': [node_static.get(n, np.nan) for n in G_sub.nodes()],
    'Value': [node_featval.get(n, np.nan) if n!=target_node else 1 for n in G_sub.nodes()]
})
node_info.to_csv("G_sub_nodes.csv", index=False)

edge_info = pd.DataFrame({
    'source': [u for u,v in G_sub.edges()],
    'target': [v for u,v in G_sub.edges()],
    'weight': [G_sub.edges[u,v]['weight'] for u,v in G_sub.edges()],
    'length': [G_sub.edges[u,v]['length'] for u,v in G_sub.edges()]
})
edge_info.to_csv("G_sub_edges.csv", index=False)

print("节点信息保存到 G_sub_nodes.csv，边信息保存到 G_sub_edges.csv")