import os
import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from tqdm import tqdm

# ========== GNN 定义 ==========
class GraphEncoder(torch.nn.Module):
    def __init__(self, in_dim=1, hid_dim=64, out_dim=128):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return global_mean_pool(x, batch)  # [1, out_dim]


# ========== 路径 ==========
GRAPH_DIR = "patient_graph_data"   # 你的 pt 子图目录
OUT_DIR   = "graph_embeddings"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphEncoder().to(device)
model.eval()

# ========== 离线计算 ==========
with torch.no_grad():
    for fname in tqdm(os.listdir(GRAPH_DIR)):
        if not fname.endswith(".pt"):
            continue

        hadm_id = fname.replace(".pt", "")
        data = torch.load(os.path.join(GRAPH_DIR, fname), weights_only=False)
        data = data.to(device)

        emb = model(data)  # [1, 128]
        torch.save(emb.cpu(), os.path.join(OUT_DIR, f"{hadm_id}.pt"))

print("✅ Graph embeddings precomputed.")
