import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict, deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
import warnings
# 在代码开头添加
import torch_geometric.data.data as pg_data

# 添加安全全局变量
torch.serialization.add_safe_globals([pg_data.DataEdgeAttr])
warnings.filterwarnings('ignore')


# =========================
# 安全加载图数据
# =========================

def safe_load_graph(file_path):
    """安全加载图数据"""
    try:
        # 尝试加载为字典格式
        saved_data = torch.load(file_path, map_location='cpu', weights_only=False)

        # 检查是否是PyG Data对象
        if hasattr(saved_data, 'x') and hasattr(saved_data, 'edge_index'):
            # 已经是Data对象
            data = saved_data
        elif isinstance(saved_data, dict):
            # 是字典格式，转换为Data对象
            from torch_geometric.data import Data

            # 提取必要字段
            x = saved_data.get('x')
            edge_index = saved_data.get('edge_index')
            hadm_id = saved_data.get('hadm_id')

            # 确保x和edge_index是张量
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            if isinstance(edge_index, np.ndarray):
                edge_index = torch.from_numpy(edge_index).long()

            # 创建Data对象
            data = Data(x=x, edge_index=edge_index)

            # 添加额外属性
            if hadm_id is not None:
                data.hadm_id = hadm_id
            if 'edge_weight' in saved_data:
                data.edge_weight = saved_data['edge_weight']

        else:
            # 其他格式，直接使用
            data = saved_data

        # 确保必要的属性存在
        if not hasattr(data, 'x'):
            print(f"警告: 文件 {file_path} 缺少x属性")
            return None

        if not hasattr(data, 'edge_index'):
            print(f"警告: 文件 {file_path} 缺少edge_index属性")
            return None

        # 确保数据类型正确
        if data.x.dtype != torch.float32:
            data.x = data.x.float()
        if data.edge_index.dtype != torch.int64:
            data.edge_index = data.edge_index.long()

        # 添加hadm_id如果不存在
        if not hasattr(data, 'hadm_id'):
            filename = os.path.basename(file_path)
            hadm_id = filename.split('.')[0]
            data.hadm_id = hadm_id

        # 添加edge_weight如果不存在
        if not hasattr(data, 'edge_weight') and hasattr(data, 'edge_index'):
            data.edge_weight = torch.ones(data.edge_index.shape[1], dtype=torch.float)

        return data

    except Exception as e:
        print(f"加载文件 {file_path} 失败: {e}")
        return None


# =========================
# 1. 优化的路径感知GNN模型
# =========================

class PathAwareGNN(nn.Module):
    """优化的路径感知GNN模型，处理大型图"""

    def __init__(self,
                 input_dim: int = 1,
                 hidden_dim: int = 32,
                 num_heads: int = 2,
                 num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # 使用较小的模型结构
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, edge_index, edge_weight=None, batch=None, return_attention=True):
        """前向传播"""
        # 第一层GAT
        x1, att1 = self.conv1(x, edge_index, return_attention_weights=True)
        x1 = F.elu(x1)

        # 第二层GAT
        x2, att2 = self.conv2(x1, edge_index, return_attention_weights=True)
        x2 = F.elu(x2)

        # 全局池化
        if batch is not None:
            graph_embedding = global_mean_pool(x2, batch)
        else:
            graph_embedding = x2.mean(dim=0, keepdim=True)

        # 分类
        logits = self.classifier(graph_embedding)

        if return_attention:
            # 合并注意力权重
            combined_attention = att1[1] + att2[1]  # 简单相加
            return logits, combined_attention, att1, att2

        return logits

    def extract_key_paths_fast(self, data, top_k=3, max_path_length=2):
        """快速的关键路径提取，适用于大型图"""
        self.eval()

        # 获取有值的节点
        active_nodes = torch.where(data.x.squeeze() > 0)[0].tolist()

        if len(active_nodes) < 2:
            return []

        print(f"活跃节点数: {len(active_nodes)}")
        print(f"活跃节点: {active_nodes}")

        # 获取注意力权重
        with torch.no_grad():
            try:
                _, edge_attention, _, _ = self.forward(
                    data.x, data.edge_index, return_attention=True
                )
            except Exception as e:
                print(f"获取注意力失败: {e}")
                edge_attention = torch.ones(data.edge_index.shape[1])

        # 转换为CPU和numpy
        edge_index = data.edge_index.cpu().numpy()

        # 构建活跃节点的邻接表（只包含与活跃节点相关的边）
        print("构建邻接表...")
        adj_list = defaultdict(list)
        edge_weights = {}

        # 限制检查的边数
        max_edges_to_check = min(50000, edge_index.shape[1])

        for i in range(max_edges_to_check):
            src, dst = edge_index[0, i], edge_index[1, i]

            # 如果src或dst是活跃节点
            if src in active_nodes or dst in active_nodes:
                adj_list[src].append(dst)
                adj_list[dst].append(src)

                # 获取边权重
                if i < edge_attention.shape[0]:
                    weight = edge_attention[i].mean().item()
                else:
                    weight = 1.0

                edge_weights[(src, dst)] = weight
                edge_weights[(dst, src)] = weight

        # 在活跃节点间寻找重要路径
        important_paths = []
        checked_pairs = set()

        print("寻找路径...")
        # 只检查每个活跃节点的前几个邻居
        for src_idx in active_nodes:
            if src_idx not in adj_list:
                continue

            # 获取邻居
            neighbors = adj_list[src_idx]

            # 只检查前几个邻居
            for neighbor in neighbors[:5]:
                if neighbor in active_nodes and neighbor > src_idx:
                    pair = (src_idx, neighbor)
                    if pair in checked_pairs:
                        continue

                    checked_pairs.add(pair)

                    # 检查是否直接连接
                    if (src_idx, neighbor) in edge_weights:
                        # 直接连接
                        path = [src_idx, neighbor]
                        weight = edge_weights[(src_idx, neighbor)]

                        important_paths.append({
                            'path': path,
                            'score': weight,
                            'src': src_idx,
                            'dst': neighbor,
                            'length': 2
                        })

        # 按分数排序
        important_paths.sort(key=lambda x: x['score'], reverse=True)

        return important_paths[:top_k]


# =========================
# 2. 高效数据加载器
# =========================

class PatientGraphDataset:
    def __init__(self, graph_dir):
        self.graph_dir = graph_dir
        self.graph_files = []
        self.graphs = []
        self.hadm_to_index = {}

    def load_graphs(self, limit=None):
        """加载图数据"""
        import glob

        # 查找所有.pt文件
        self.graph_files = glob.glob(os.path.join(self.graph_dir, "*.pt"))

        if not self.graph_files:
            print(f"在 {self.graph_dir} 中没有找到.pt文件")
            return

        if limit:
            self.graph_files = self.graph_files[:limit]

        print(f"找到 {len(self.graph_files)} 个图文件")

        loaded_count = 0

        for file_path in tqdm(self.graph_files, desc="加载图数据"):
            try:
                data = safe_load_graph(file_path)
                if data is None:
                    continue

                # 确保有hadm_id属性
                if not hasattr(data, 'hadm_id'):
                    filename = os.path.basename(file_path)
                    hadm_id = filename.split('.')[0]
                    data.hadm_id = hadm_id

                # 确保edge_index是long类型
                if data.edge_index.dtype != torch.long:
                    data.edge_index = data.edge_index.long()

                # 确保x是float类型
                if data.x.dtype != torch.float:
                    data.x = data.x.float()

                # 添加edge_weight如果不存在
                if not hasattr(data, 'edge_weight'):
                    data.edge_weight = torch.ones(data.edge_index.shape[1], dtype=torch.float)

                self.graphs.append(data)
                self.hadm_to_index[str(data.hadm_id)] = len(self.graphs) - 1
                loaded_count += 1

                if limit and loaded_count >= limit:
                    break

            except Exception as e:
                print(f"加载 {file_path} 失败: {e}")

        print(f"成功加载 {len(self.graphs)} 个图")

    def get_graph_by_id(self, hadm_id):
        """根据ID获取图"""
        hadm_id_str = str(hadm_id)
        idx = self.hadm_to_index.get(hadm_id_str)
        if idx is not None:
            return self.graphs[idx]
        return None


# =========================
# 3. 高效关键路径分析器
# =========================

class KeyPathAnalyzer:
    def __init__(self, model, mapping_df_path=None):
        self.model = model
        self.model.eval()  # 设置为评估模式

        # 加载映射关系
        self.cid2feature = {}
        if mapping_df_path and os.path.exists(mapping_df_path):
            self.load_mapping(mapping_df_path)
        else:
            print("警告：未提供映射文件或文件不存在，将使用节点索引作为特征名")

    def load_mapping(self, mapping_df_path):
        """加载映射关系"""
        try:
            mapping_df = pd.read_csv(mapping_df_path)
            print(f"映射文件列名: {mapping_df.columns.tolist()}")
            print(f"映射文件前几行:\n{mapping_df.head()}")

            # 确保列存在
            required_cols = ['feature', 'mapping']
            if not all(col in mapping_df.columns for col in required_cols):
                print(f"映射文件缺少必要列，需要: {required_cols}")
                return

            # 清理数据
            mapping_df = mapping_df.dropna(subset=['feature', 'mapping'])
            mapping_df["feature"] = mapping_df["feature"].astype(str).str.strip()
            mapping_df["mapping"] = pd.to_numeric(mapping_df["mapping"], errors='coerce')
            mapping_df = mapping_df.dropna(subset=['mapping'])

            # 创建映射
            self.feature2cid = dict(zip(mapping_df["feature"], mapping_df["mapping"]))
            self.cid2feature = {int(v): k for k, v in self.feature2cid.items()}

            print(f"成功加载 {len(self.cid2feature)} 个映射关系")
            print(f"CID示例: {list(self.cid2feature.keys())[:5]}")
            print(f"特征名示例: {list(self.cid2feature.values())[:5]}")

        except Exception as e:
            print(f"加载映射文件失败: {e}")
            import traceback
            traceback.print_exc()

    def get_feature_name(self, node_idx):
        """获取特征名"""
        if node_idx in self.cid2feature:
            return self.cid2feature[node_idx]
        else:
            # 尝试查找近似值
            if self.cid2feature:
                # 转换为整数
                try:
                    node_int = int(node_idx)
                    # 查找最接近的CID
                    closest_cid = min(self.cid2feature.keys(), key=lambda x: abs(x - node_int))
                    if abs(closest_cid - node_int) <= 100:  # 允许100以内的偏差
                        return f"{self.cid2feature[closest_cid]}(近似:{node_int})"
                except:
                    pass
            return f"Node_{node_idx}"

    def analyze_patient_fast(self, data, top_k=3):
        """快速分析患者关键路径"""
        if data is None:
            print("错误：数据为空")
            return None

        print(f"\n分析患者 {getattr(data, 'hadm_id', 'unknown')}:")
        print(f"  节点数: {data.x.shape[0]}")
        print(f"  边数: {data.edge_index.shape[1]}")

        # 获取活跃节点
        active_mask = data.x.squeeze() > 0
        active_indices = torch.where(active_mask)[0].tolist()
        active_values = data.x[active_mask].squeeze().tolist()

        print(f"  有值节点数: {len(active_indices)}")
        if active_indices:
            print(f"  有值节点索引: {active_indices[:10]}{'...' if len(active_indices) > 10 else ''}")
            print(f"  有值节点值: {active_values[:10]}{'...' if len(active_values) > 10 else ''}")

        # 如果活跃节点太少，直接返回
        if len(active_indices) < 2:
            print("  活跃节点不足，无法分析路径")
            return {
                'hadm_id': getattr(data, 'hadm_id', 'unknown'),
                'num_nodes': data.x.shape[0],
                'num_edges': data.edge_index.shape[1],
                'active_nodes': len(active_indices),
                'key_paths': [],
                'note': '活跃节点不足'
            }

        # 使用快速路径提取
        try:
            key_paths = self.model.extract_key_paths_fast(data, top_k=top_k)
        except Exception as e:
            print(f"提取关键路径失败: {e}")
            key_paths = []

        # 转换为可解释格式
        interpretable_paths = []

        for i, path_info in enumerate(key_paths):
            path = path_info['path']
            score = path_info['score']

            interpretable_path = {
                'rank': i + 1,
                'score': float(score),
                'nodes': [],
                'features': [],
                'values': [],
                'length': len(path)
            }

            # 添加节点信息
            for node_idx in path:
                node_idx_int = int(node_idx)
                feature_name = self.get_feature_name(node_idx_int)
                node_value = data.x[node_idx_int].item() if node_idx_int < data.x.shape[0] else 0.0

                interpretable_path['nodes'].append(node_idx_int)
                interpretable_path['features'].append(feature_name)
                interpretable_path['values'].append(float(node_value))

            interpretable_paths.append(interpretable_path)

        # 如果没有找到路径，尝试直接连接分析
        if not interpretable_paths and len(active_indices) >= 2:
            print("  未找到路径，尝试直接连接分析...")
            # 创建简单的直接连接
            for i in range(min(top_k, len(active_indices) - 1)):
                src = active_indices[i]
                dst = active_indices[i + 1]

                interpretable_path = {
                    'rank': i + 1,
                    'score': 0.5,  # 默认分数
                    'nodes': [src, dst],
                    'features': [self.get_feature_name(src), self.get_feature_name(dst)],
                    'values': [float(data.x[src].item()), float(data.x[dst].item())],
                    'length': 2,
                    'note': '直接连接'
                }
                interpretable_paths.append(interpretable_path)

        return {
            'hadm_id': getattr(data, 'hadm_id', 'unknown'),
            'num_nodes': data.x.shape[0],
            'num_edges': data.edge_index.shape[1],
            'active_nodes': len(active_indices),
            'key_paths': interpretable_paths,
            'analysis_method': 'fast'
        }

    def visualize_path_bar(self, path_info, title=None, save_path=None):
        """使用条形图可视化路径"""
        if not path_info or 'features' not in path_info:
            print("无效的路径信息")
            return

        features = path_info['features']
        values = path_info['values']

        if len(features) != len(values):
            print(f"特征和值数量不匹配: {len(features)} != {len(values)}")
            return

        # 创建条形图
        plt.figure(figsize=(max(8, len(features) * 1.5), 6))

        x_pos = np.arange(len(features))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))

        # 绘制条形
        bars = plt.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='black')

        # 添加值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=10)

        # 设置x轴标签
        plt.xticks(x_pos, features, rotation=45, ha='right', fontsize=10)

        plt.ylabel('节点值', fontsize=12)
        plt.title(title or f"关键路径 (分数: {path_info.get('score', 0):.3f})", fontsize=14, fontweight='bold')

        # 添加网格
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def save_analysis_results(self, results, output_file="gnn_path_importance/key_paths_analysis.json"):
        """保存分析结果"""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"结果保存到 {output_file}")
        except Exception as e:
            print(f"保存结果失败: {e}")


# =========================
# 4. 主程序
# =========================

def main():
    """主程序"""
    print("=" * 60)
    print("路径感知GNN关键路径分析")
    print("=" * 60)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载数据
    print("\n1. 加载患者图数据...")

    # 检查数据目录
    graph_dir = "../../model/data/patient_graph_data"
    if not os.path.exists(graph_dir):
        print(f"错误: 目录不存在: {graph_dir}")
        # 尝试相对路径
        graph_dir = "../../model/data/patient_graph_data"
        if not os.path.exists(graph_dir):
            print(f"错误: 目录不存在: {graph_dir}")
            return

    # 加载数据集
    dataset = PatientGraphDataset(graph_dir)
    dataset.load_graphs(limit=5)  # 只加载5个图用于测试

    if not dataset.graphs:
        print("错误: 没有加载到任何图数据")
        # 尝试直接加载单个文件
        sample_file = os.path.join(graph_dir, "20000235.0.pt")
        if os.path.exists(sample_file):
            print(f"尝试加载单个文件: {sample_file}")
            data = safe_load_graph(sample_file)
            if data:
                dataset.graphs = [data]
                dataset.hadm_to_index[str(data.hadm_id)] = 0
            else:
                return
        else:
            # 列出目录中的文件
            import glob
            files = glob.glob(os.path.join(graph_dir, "*.pt"))
            if files:
                print(f"目录中的文件: {files[:5]}...")
                # 尝试加载第一个文件
                data = safe_load_graph(files[0])
                if data:
                    dataset.graphs = [data]
                    dataset.hadm_to_index[str(data.hadm_id)] = 0
                else:
                    return
            else:
                print(f"目录 {graph_dir} 中没有.pt文件")
                return

    # 2. 创建模型
    print("\n2. 创建路径感知GNN模型...")
    model = PathAwareGNN(
        input_dim=1,
        hidden_dim=16,  # 使用更小的维度以提高速度
        num_heads=2,
        num_classes=2
    ).to(device)

    # 3. 加载映射关系
    mapping_path = "../mapping.csv"
    analyzer = KeyPathAnalyzer(model, mapping_path)

    # 4. 分析每个患者
    print("\n3. 分析患者关键路径...")
    all_results = []

    for i, data in enumerate(dataset.graphs):
        print(f"\n--- 分析患者 {i + 1}/{len(dataset.graphs)} ---")

        # 移动到设备
        data_device = data.to(device)

        # 分析患者
        result = analyzer.analyze_patient_fast(data_device, top_k=3)

        if result:
            all_results.append(result)

            # 显示结果
            print(f"\n患者 {result['hadm_id']} 分析结果:")
            print(f"  总节点数: {result['num_nodes']:,}")
            print(f"  边数: {result['num_edges']:,}")
            print(f"  活跃节点: {result['active_nodes']}")
            print(f"  分析方法: {result['analysis_method']}")

            if result['key_paths']:
                print(f"  找到 {len(result['key_paths'])} 条关键路径:")
                for path in result['key_paths']:
                    print(f"\n    路径 {path['rank']} (分数: {path['score']:.3f}):")
                    for j, (feature, value) in enumerate(zip(path['features'], path['values'])):
                        print(f"      {j}. {feature}: {value:.4f}")

                # 可视化第一条路径
                analyzer.visualize_path_bar(
                    result['key_paths'][0],
                    title=f"患者 {result['hadm_id']} 的关键路径"
                )
            else:
                print("  未找到关键路径")

    # 5. 保存结果
    print("\n4. 保存分析结果...")
    if all_results:
        analyzer.save_analysis_results(all_results, "key_paths_results.json")

        # 生成摘要
        print("\n5. 分析摘要:")
        print(f"  分析患者数: {len(all_results)}")

        total_paths = sum(len(r['key_paths']) for r in all_results)
        print(f"  总关键路径数: {total_paths}")

        if total_paths > 0:
            avg_paths = total_paths / len(all_results)
            print(f"  平均每患者关键路径数: {avg_paths:.2f}")

            # 收集所有特征
            all_features = []
            for result in all_results:
                for path in result['key_paths']:
                    all_features.extend(path['features'])

            # 统计特征出现频率
            from collections import Counter
            feature_counts = Counter(all_features)

            if feature_counts:
                print(f"\n  常见特征 (前10):")
                for feature, count in feature_counts.most_common(10):
                    print(f"    {feature}: {count}次")
    else:
        print("  没有分析结果可保存")

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


# =========================
# 直接分析特定患者
# =========================

def analyze_specific_patient(hadm_id="20000235.0"):
    """分析特定患者"""
    print("=" * 60)
    print(f"分析特定患者: {hadm_id}")
    print("=" * 60)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载患者图
    file_path = f"../../model/data/patient_graph_data/{hadm_id}.pt"
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        # 尝试其他扩展名
        import glob
        files = glob.glob(f"../../model/data/patient_graph_data/{hadm_id}*.pt")
        if files:
            file_path = files[0]
            print(f"使用文件: {file_path}")
        else:
            print("未找到患者文件")
            return

    print(f"加载患者文件: {file_path}")
    data = safe_load_graph(file_path)

    if data is None:
        print("无法加载患者数据")
        return

    print(f"患者图信息:")
    print(f"  节点数: {data.x.shape[0]:,}")
    print(f"  边数: {data.edge_index.shape[1]:,}")

    # 2. 创建模型
    model = PathAwareGNN(
        input_dim=1,
        hidden_dim=32,
        num_heads=2,
        num_classes=2
    ).to(device)

    # 3. 加载映射关系
    mapping_path = "../mapping.csv"
    analyzer = KeyPathAnalyzer(model, mapping_path)

    # 4. 分析患者
    data_device = data.to(device)
    result = analyzer.analyze_patient_fast(data_device, top_k=5)

    if result:
        print(f"\n患者 {result['hadm_id']} 分析结果:")
        print(f"  总节点数: {result['num_nodes']:,}")
        print(f"  边数: {result['num_edges']:,}")
        print(f"  活跃节点: {result['active_nodes']}")

        if result['key_paths']:
            print(f"  找到 {len(result['key_paths'])} 条关键路径:")

            # 保存详细结果
            detailed_file = f"patient_{hadm_id}_paths.json"
            with open(detailed_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"  详细结果保存到: {detailed_file}")

            # 显示每条路径
            for path in result['key_paths']:
                print(f"\n    路径 {path['rank']} (分数: {path['score']:.3f}):")
                for j, (feature, value) in enumerate(zip(path['features'], path['values'])):
                    print(f"      {j}. {feature}: {value:.4f}")

            # 可视化
            analyzer.visualize_path_bar(
                result['key_paths'][0],
                title=f"患者 {hadm_id} 的关键路径",
                save_path=f"patient_{hadm_id}_path.png"
            )
        else:
            print("  未找到关键路径")

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


# =========================
# 运行程序
# =========================

if __name__ == "__main__":
    print("选择分析模式:")
    print("1. 批量分析多个患者")
    print("2. 分析特定患者 (20000235.0)")

    choice = input("请输入选择 (1 或 2): ").strip()

    if choice == "2":
        # 询问患者ID
        hadm_id = input("请输入患者ID (默认为20000235.0): ").strip()
        if not hadm_id:
            hadm_id = "20000235.0"
        analyze_specific_patient(hadm_id)
    else:
        main()