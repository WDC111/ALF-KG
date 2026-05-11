import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
import time
from tqdm import tqdm

print("开始构建目标节点连通图...")
start_time = time.time()

# -----------------------------
# Step 1: 读取目标节点
# -----------------------------
print("1. 读取目标节点...")
mapping_df = pd.read_csv("mapping.csv")
target_nodes = list(mapping_df['mapping'].unique())
print(f"   找到 {len(target_nodes)} 个目标节点")
print(f"   目标节点: {target_nodes}")

# -----------------------------
# Step 2: 读取完整知识图谱
# -----------------------------
print("2. 加载完整知识图谱...")
relations_df = pd.read_csv("ONE-HOP/Processed_Relations.txt", sep="\t")

# 创建完整图
G_full = nx.DiGraph()
for _, row in tqdm(relations_df.iterrows(), total=len(relations_df), desc="构建完整图"):
    G_full.add_edge(row['head.cid'], row['tail.cid'], relation=row['relation'])

print(f"   完整图包含 {G_full.number_of_nodes():,} 个节点和 {G_full.number_of_edges():,} 条边")

# -----------------------------
# Step 3: 创建目标节点连通图（有向）
# -----------------------------
print("3. 创建目标节点连通图 (有向)...")

# 使用有向图
G_target = nx.DiGraph()

# 添加所有目标节点
for node in target_nodes:
    if node in G_full:
        G_target.add_node(node)
    else:
        print(f"   警告: 目标节点 {node} 不在完整图中，已跳过")

print(f"   连通图初始节点数: {G_target.number_of_nodes()}")

# -----------------------------
# Step 4: 计算有向最短路径并添加边
# -----------------------------
print("4. 计算目标节点间有向最短路径并设置边权重...")

available_targets = list(G_target.nodes())
n = len(available_targets)
print(f"   需要计算 {n} 个节点之间的有向连通性")

shortest_paths = {}
path_lengths = {}

total_pairs = n * (n - 1)  # 有向图中节点对数量
processed_pairs = 0

with tqdm(total=total_pairs, desc="计算有向最短路径") as pbar:
    for source in available_targets:
        try:
            # 单源最短路径（有向）
            lengths = nx.single_source_shortest_path_length(G_full, source, cutoff=20)

            for target in available_targets:
                if source == target:
                    processed_pairs += 1
                    pbar.update(1)
                    continue

                if target in lengths:
                    path_len = lengths[target]
                    if path_len > 0 and path_len < 4:
                        weight = 1.0 / path_len
                        G_target.add_edge(source, target, weight=weight, path_length=path_len)
                        shortest_paths[(source, target)] = path_len
                        path_lengths[(source, target)] = weight

                processed_pairs += 1
                pbar.update(1)

        except Exception as e:
            print(f"   计算节点 {source} 的最短路径时出错: {e}")
            for target in available_targets:
                processed_pairs += 1
                pbar.update(1)

print(f"   有向连通图最终节点数: {G_target.number_of_nodes()}")
print(f"   有向连通图边数: {G_target.number_of_edges()}")


# -----------------------------
# Step 5: 保存连通图
# -----------------------------
print("5. 保存连通图...")

# 保存边列表（带权重）
output_file = "target_connectivity_graph.edgelist"
with open(output_file, 'w') as f:
    for u, v, data in G_target.edges(data=True):
        weight = data.get('weight', 0)
        path_len = data.get('path_length', 0)
        f.write(f"{u}\t{v}\t{weight}\t{path_len}\n")

print(f"   连通图已保存: {output_file}")

# 保存节点信息
node_info = pd.DataFrame({
    'node': list(G_target.nodes()),
    'degree': [G_target.degree(node) for node in G_target.nodes()]
})
node_info.to_csv("target_nodes_info.csv", index=False)
print(f"   节点信息已保存: target_nodes_info.csv")

# 保存边信息
edge_info = []
for u, v, data in G_target.edges(data=True):
    edge_info.append({
        'source': u,
        'target': v,
        'weight': data.get('weight', 0),
        'path_length': data.get('path_length', 0)
    })
edge_df = pd.DataFrame(edge_info)
edge_df.to_csv("target_edges_info.csv", index=False)
print(f"   边信息已保存: target_edges_info.csv")

# -----------------------------
# Step 6: 可视化连通图
# -----------------------------
print("\n6. 可视化连通图...")

# 创建可视化
plt.figure(figsize=(16, 12))

# 使用多种布局算法
print("   计算节点布局...")
try:
    # 尝试使用spring布局
    pos = nx.spring_layout(G_target, seed=42, k=0.8, iterations=100)
    layout_name = "spring布局"
except:
    try:
        # 如果spring布局失败，尝试kamada_kawai布局
        pos = nx.kamada_kawai_layout(G_target)
        layout_name = "kamada_kawai布局"
    except:
        # 最后使用circular布局
        pos = nx.circular_layout(G_target)
        layout_name = "circular布局"

print(f"   使用 {layout_name}")

# 准备节点颜色和大小（基于节点度）
node_degrees = dict(G_target.degree())
max_degree = max(node_degrees.values()) if node_degrees else 1

node_colors = []
node_sizes = []
for node in G_target.nodes():
    degree = node_degrees[node]
    # 节点颜色：度越高颜色越深
    color_intensity = 0.3 + 0.7 * (degree / max_degree)
    node_colors.append((color_intensity, 0.5, 0.7))
    # 节点大小：度越高节点越大
    node_sizes.append(300 + 700 * (degree / max_degree))

# 绘制节点
nx.draw_networkx_nodes(G_target, pos,
                       node_color=node_colors,
                       node_size=node_sizes,
                       alpha=0.9,
                       edgecolors='black',
                       linewidths=1.5)

# 绘制边（根据权重调整宽度）
edge_weights = [G_target.edges[u, v].get('weight', 0.1) for u, v in G_target.edges()]
if edge_weights:
    max_weight = max(edge_weights)
    min_weight = min(edge_weights) if min(edge_weights) > 0 else 0.001
else:
    max_weight = 1
    min_weight = 0.001

edge_widths = []
for u, v in G_target.edges():
    weight = G_target.edges[u, v].get('weight', 0.1)
    # 归一化权重到[2, 10]范围作为线宽
    if max_weight > min_weight:
        normalized_weight = (weight - min_weight) / (max_weight - min_weight)
        width = 2 + 8 * normalized_weight
    else:
        width = 5
    edge_widths.append(width)

# 边的颜色也根据权重变化
edge_colors = []
for u, v in G_target.edges():
    weight = G_target.edges[u, v].get('weight', 0.1)
    # 权重越高颜色越深
    if max_weight > min_weight:
        color_intensity = 0.3 + 0.7 * ((weight - min_weight) / (max_weight - min_weight))
    else:
        color_intensity = 0.5
    edge_colors.append((0.2, color_intensity, 0.5, 0.7))

nx.draw_networkx_edges(G_target, pos,
                       width=edge_widths,
                       alpha=0.7,
                       edge_color=edge_colors,
                       style='solid')

# 绘制节点标签
nx.draw_networkx_labels(G_target, pos,
                        font_size=9,
                        font_weight='bold',
                        font_color='black',
                        verticalalignment='center',
                        horizontalalignment='center')

# 绘制边的权重标签
edge_labels = {}
for u, v, data in G_target.edges(data=True):
    weight = data.get('weight', 0)
    if weight > 0.01:  # 只显示权重大于0.01的边
        edge_labels[(u, v)] = f"{weight:.3f}"

if edge_labels:
    nx.draw_networkx_edge_labels(G_target, pos,
                                 edge_labels=edge_labels,
                                 font_size=7,
                                 font_color='darkred',
                                 font_weight='bold',
                                 label_pos=0.5,
                                 bbox=dict(boxstyle="round,pad=0.3",
                                           facecolor="white",
                                           edgecolor="lightgray",
                                           alpha=0.8))

plt.title(
    f"目标节点连通图\n节点: {G_target.number_of_nodes()}, 边: {G_target.number_of_edges()}\n边权重 = 1 / 最短路径长度",
    fontsize=16, fontweight='bold', pad=20)
plt.axis('off')

# 添加图例
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [
    Patch(facecolor=(0.3, 0.5, 0.7), edgecolor='black', label='节点 (颜色深浅表示度)'),
    Line2D([0], [0], color=(0.2, 0.5, 0.5, 0.7), lw=5, label='边 (宽度/颜色表示权重)'),
    Patch(facecolor='white', edgecolor='lightgray', label='权重标签')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=10,
           bbox_to_anchor=(1.0, 1.0), framealpha=0.9)

plt.tight_layout()
plt.savefig("target_connectivity_visualization.png", dpi=300, bbox_inches='tight')
print("   连通图可视化已保存为: target_connectivity_visualization.png")
plt.show()

# -----------------------------
# Step 7: 创建统计分析图
# -----------------------------
print("\n7. 创建统计分析图...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1: 节点度分布
degrees = [G_target.degree(node) for node in G_target.nodes()]
axes[0, 0].hist(degrees, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].set_xlabel('节点度', fontsize=10)
axes[0, 0].set_ylabel('节点数量', fontsize=10)
axes[0, 0].set_title('节点度分布', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 子图2: 边权重分布
weights = [G_target.edges[u, v].get('weight', 0) for u, v in G_target.edges()]
if weights:
    axes[0, 1].hist(weights, bins=20, alpha=0.7, color='darkorange', edgecolor='black')
    axes[0, 1].set_xlabel('边权重 (1/最短路径长度)', fontsize=10)
    axes[0, 1].set_ylabel('边数量', fontsize=10)
    axes[0, 1].set_title('边权重分布', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
else:
    axes[0, 1].text(0.5, 0.5, '无边数据', ha='center', va='center', fontsize=12)
    axes[0, 1].set_title('边权重分布', fontsize=12, fontweight='bold')

# 子图3: 最短路径长度分布
path_lengths = [G_target.edges[u, v].get('path_length', 0) for u, v in G_target.edges()]
if path_lengths and max(path_lengths) > 0:
    # 过滤掉0值（表示没有路径或同一节点）
    filtered_lengths = [l for l in path_lengths if l > 0]
    if filtered_lengths:
        axes[1, 0].hist(filtered_lengths, bins=15, alpha=0.7, color='seagreen', edgecolor='black')
        axes[1, 0].set_xlabel('最短路径长度', fontsize=10)
        axes[1, 0].set_ylabel('边数量', fontsize=10)
        axes[1, 0].set_title('最短路径长度分布', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, '无路径数据', ha='center', va='center', fontsize=12)
        axes[1, 0].set_title('最短路径长度分布', fontsize=12, fontweight='bold')
else:
    axes[1, 0].text(0.5, 0.5, '无路径数据', ha='center', va='center', fontsize=12)
    axes[1, 0].set_title('最短路径长度分布', fontsize=12, fontweight='bold')

# 子图4: 节点连接数排名
if node_degrees:
    # 获取前15个度最高的节点
    top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:15]
    node_names = [node[0][:8] + "..." if len(node[0]) > 11 else node[0] for node in top_nodes]
    node_degrees_vals = [node[1] for node in top_nodes]

    axes[1, 1].barh(range(len(node_names)), node_degrees_vals, alpha=0.7, color='purple')
    axes[1, 1].set_yticks(range(len(node_names)))
    axes[1, 1].set_yticklabels(node_names, fontsize=8)
    axes[1, 1].set_xlabel('连接数', fontsize=10)
    axes[1, 1].set_title('节点连接数排名 (Top 15)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
else:
    axes[1, 1].text(0.5, 0.5, '无节点度数据', ha='center', va='center', fontsize=12)
    axes[1, 1].set_title('节点连接数排名', fontsize=12, fontweight='bold')

plt.suptitle('连通图统计分析', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("connectivity_statistical_analysis.png", dpi=300, bbox_inches='tight')
print("   统计分析图已保存为: connectivity_statistical_analysis.png")
plt.show()

# -----------------------------
# Step 8: 输出图的基本信息
# -----------------------------
print("\n8. 输出图的基本信息...")

# 计算图的密度
density = nx.density(G_target)
print(f"   图密度: {density:.6f}")

# 计算平均聚类系数
try:
    avg_clustering = nx.average_clustering(G_target)
    print(f"   平均聚类系数: {avg_clustering:.4f}")
except:
    print("   无法计算平均聚类系数")

# 检查连通性
if nx.is_connected(G_target):
    print("   图是连通的")

    # 计算平均最短路径长度
    try:
        avg_path_length = nx.average_shortest_path_length(G_target, weight='weight')
        print(f"   平均最短路径长度(加权): {avg_path_length:.4f}")
    except:
        print("   无法计算平均最短路径长度")
else:
    print(f"   图不是连通的，包含 {nx.number_connected_components(G_target)} 个连通分量")

# 计算权重统计
if weights:
    weight_stats = pd.Series(weights).describe()
    print(f"\n   权重统计:")
    print(f"     最小值: {weight_stats['min']:.6f}")
    print(f"     最大值: {weight_stats['max']:.6f}")
    print(f"     平均值: {weight_stats['mean']:.6f}")
    print(f"     中位数: {weight_stats['50%']:.6f}")

total_time = time.time() - start_time
print("\n" + "=" * 60)
print("【连通图构建完成】")
print("=" * 60)
print(f"总耗时: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")
print(f"目标节点数: {len(target_nodes)}")
print(f"连通图统计:")
print(f"  - 节点数: {G_target.number_of_nodes()}")
print(f"  - 边数: {G_target.number_of_edges()}")
print(f"  - 平均节点度: {sum(node_degrees.values()) / len(node_degrees):.2f}")
print(f"生成的文件:")
print(f"1. 连通图文件:")
print(f"   - {output_file} (图数据)")
print(f"   - target_nodes_info.csv (节点信息)")
print(f"   - target_edges_info.csv (边信息)")
print(f"2. 可视化文件:")
print(f"   - target_connectivity_visualization.png (连通图可视化)")
print(f"   - connectivity_statistical_analysis.png (统计分析图)")
print("=" * 60)