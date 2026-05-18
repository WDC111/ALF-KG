import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import random


# ================= Reproducibility =================
def set_seed(seed=64):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(64)


# ================= Dataset =================
class PatientDataset(Dataset):
    def __init__(self, static_csv, ts_csv, graph_emb_dir):
        self.static_df = pd.read_csv(static_csv)
        self.ts_df = pd.read_csv(ts_csv)
        self.graph_emb_dir = graph_emb_dir

        self.hadm_ids = self.static_df["hadm_id"].values
        self.static_feats = self.static_df.drop(
            columns=["hadm_id", "outcome_21d"]
        ).values.astype(np.float32)
        self.labels = self.static_df["outcome_21d"].values.astype(np.float32)
        self.ts_dim = self.ts_df.shape[1] - 2

    def __len__(self):
        return len(self.hadm_ids)

    def __getitem__(self, idx):
        hadm_id = str(int(self.hadm_ids[idx]))
        graph_emb = torch.load(f"{self.graph_emb_dir}/{hadm_id}.0.pt").squeeze(0)
        static_feat = torch.tensor(self.static_feats[idx])
        label = torch.tensor(self.labels[idx])
        ts = self.ts_df[self.ts_df["hadm_id"] == float(hadm_id)]
        ts_feat = torch.tensor(
            ts.drop(columns=["hadm_id", "hour"]).values,
            dtype=torch.float32
        )
        return graph_emb, static_feat, ts_feat, label


# ================= Encoders =================
class StaticEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.relu(self.fc(x))


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]


# ================= Models =================
# 修改MultimodalModel，去掉attention
class MultimodalModel(nn.Module):
    def __init__(self, static_dim, ts_dim, graph_dim=128, embed_dim=64):
        super().__init__()
        self.graph_proj = nn.Linear(graph_dim, embed_dim)
        self.static_enc = StaticEncoder(static_dim, embed_dim)
        self.temporal_enc = TemporalEncoder(ts_dim, embed_dim)
        # 去掉attention，使用简单融合
        self.cls = nn.Linear(embed_dim, 1)

    def forward(self, g, s, t):
        g = self.graph_proj(g)
        s = self.static_enc(s)
        t = self.temporal_enc(t)
        # 简单拼接后使用线性层
        fused = torch.cat([g, s, t], dim=1)
        # 通过一个额外的线性层进行降维
        fused = nn.functional.relu(nn.Linear(192, 64).to(fused.device)(fused))
        return self.cls(fused).squeeze(1), None  # 返回None代替attention权重


class MeanFusionModel(MultimodalModel):
    def forward(self, g, s, t):
        g = self.graph_proj(g)
        s = self.static_enc(s)
        t = self.temporal_enc(t)
        fused = (g + s + t) / 3
        return self.cls(fused).squeeze(1), None


class NoKGModel(MultimodalModel):
    def forward(self, g, s, t):
        s = self.static_enc(s)
        t = self.temporal_enc(t)
        fused = (s + t) / 2
        return self.cls(fused).squeeze(1), None


class NoTSModel(MultimodalModel):
    def forward(self, g, s, t):
        g = self.graph_proj(g)
        s = self.static_enc(s)
        fused = (g + s) / 2
        return self.cls(fused).squeeze(1), None


# ================= Evaluation =================
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for g, s, t, y in loader:
            g, s, t = g.to(device), s.to(device), t.to(device)
            logit, _ = model(g, s, t)
            prob = torch.sigmoid(logit)
            y_true.extend(y.numpy())
            y_pred.extend(prob.cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred > 0.5),
        "F1": f1_score(y_true, y_pred > 0.5),
        "Precision": precision_score(y_true, y_pred > 0.5),
        "Recall": recall_score(y_true, y_pred > 0.5),
        "AUC": roc_auc_score(y_true, y_pred)
    }
    pos, neg = int(y_true.sum()), len(y_true) - int(y_true.sum())
    return metrics, pos, neg


# ================= Training + Selection =================
def train_experiment(model_class, name):
    print(f"\n========== {name} ==========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PatientDataset(
        "patient_clinical_filt.csv",
        "time_series.csv",
        "graph_embeddings"
    )

    # 7:3 split
    train_len = int(0.7 * len(dataset))
    test_len = len(dataset) - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    model = model_class(
        static_dim=dataset.static_feats.shape[1],
        ts_dim=dataset.ts_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = 0
    best_epoch = -1
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, 21):
        model.train()
        total_loss = 0
        for g, s, t, y in tqdm(train_loader, desc=f"{name} Epoch {epoch}/20"):
            g, s, t, y = g.to(device), s.to(device), t.to(device), y.to(device)
            logit, _ = model(g, s, t)
            loss = loss_fn(logit, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # evaluate
        metrics, pos, neg = evaluate(model, test_loader, device)
        auc = metrics["AUC"]

        torch.save(model.state_dict(), f"checkpoints/{name}_epoch{epoch}.pt")
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            torch.save(model.state_dict(), f"checkpoints/{name}_best.pt")

        print(f"Epoch {epoch} | Loss={avg_loss:.4f} | AUC={auc:.4f}")

    # Final report
    print(f"\n{name} BEST Epoch {best_epoch}")
    best_model = model_class(
        static_dim=dataset.static_feats.shape[1],
        ts_dim=dataset.ts_dim
    ).to(device)
    best_model.load_state_dict(torch.load(f"checkpoints/{name}_best.pt"))
    metrics, pos, neg = evaluate(best_model, test_loader, device)

    print(f"Test set positive: {pos}, negative: {neg}")
    print(f"Metrics: {metrics}")
    return name, metrics, pos, neg


# ================= Run All =================
if __name__ == "__main__":
    results = []
    for model_class, name in [
        (MultimodalModel, "Full_Attention"),  # 改名为Full_Model更合适，但保持您原来的命名
        (MeanFusionModel, "No_Attention"),
        (NoKGModel, "No_KG"),
        (NoTSModel, "No_TS")
    ]:
        res = train_experiment(model_class, name)
        results.append(res)

    # 输出表格
    df = pd.DataFrame([{
        "Model": r[0],
        "Accuracy": r[1]["Accuracy"],
        "F1-score": r[1]["F1"],
        "Precision": r[1]["Precision"],
        "Recall": r[1]["Recall"],
        "AUC": r[1]["AUC"],
        "Test_Pos": r[2],
        "Test_Neg": r[3]
    } for r in results])
    print("\n===== Summary Table =====")
    print(df)
    df.to_csv("results_summary.csv", index=False)