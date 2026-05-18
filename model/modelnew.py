import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import random

def set_seed(seed=64):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(64)

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

class MultimodalModel(nn.Module):
    def __init__(self, static_dim, ts_dim, graph_dim=128, embed_dim=64, dropout_rate=0.5):
        super().__init__()
        self.graph_proj = nn.Linear(graph_dim, embed_dim)
        self.static_enc = StaticEncoder(static_dim, embed_dim)
        self.temporal_enc = TemporalEncoder(ts_dim, embed_dim)

        self.fusion_fc = nn.Linear(embed_dim * 3, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.cls = nn.Linear(embed_dim, 1)

    def forward(self, g, s, t):
        g = self.graph_proj(g)
        s = self.static_enc(s)
        t = self.temporal_enc(t)

        fused = torch.cat([g, s, t], dim=1)
        fused = torch.relu(self.fusion_fc(fused))
        fused = self.dropout(fused)
        return self.cls(fused).squeeze(1)

def evaluate(model, loader, device):
    """返回指标、正负样本数、以及真实标签和预测概率（用于后续绘图）"""
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for g, s, t, y in loader:
            g, s, t = g.to(device), s.to(device), t.to(device)
            logit = model(g, s, t)
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
    return metrics, pos, neg, y_true, y_pred

def plot_calibration_curve(y_true, y_pred, save_path="calibration_curve.png"):
    """绘制校准曲线，并保存图片"""
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy='uniform')
    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Full_Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Calibration curve saved to {save_path}")

def plot_decision_curve(y_true, y_pred, save_path="decision_curve.png"):
    """绘制决策曲线 (DCA)"""
    thresholds = np.linspace(0.01, 0.99, 100)
    n = len(y_true)
    pos_rate = y_true.mean()
    net_benefit = []

    for thresh in thresholds:
        y_pred_binary = (y_pred >= thresh).astype(int)
        tp = np.sum((y_pred_binary == 1) & (y_true == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit.append(nb)

    # Treat All 策略
    nb_all = pos_rate - (1 - pos_rate) * (thresholds / (1 - thresholds))

    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, net_benefit, linewidth=2, label='Full_Model')
    plt.plot(thresholds, nb_all, linestyle='--', color='red', label='Treat All')
    plt.axhline(0, color='gray', linestyle='--', label='Treat None')
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Decision curve saved to {save_path}")

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PatientDataset(
        "patient_clinical_filt.csv",
        "time_series.csv",
        "graph_embeddings"
    )

    train_len = int(0.7 * len(dataset))
    test_len = len(dataset) - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    model = MultimodalModel(
        static_dim=dataset.static_feats.shape[1],
        ts_dim=dataset.ts_dim,
        dropout_rate=0.5
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    patience = 5
    best_auc = 0
    best_epoch = -1
    epochs_no_improve = 0
    best_model_state = None

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, 101):
        model.train()
        total_loss = 0

        for g, s, t, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            g, s, t, y = g.to(device), s.to(device), t.to(device), y.to(device)
            logit = model(g, s, t)
            loss = loss_fn(logit, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 验证
        metrics, pos, neg, _, _ = evaluate(model, test_loader, device)
        auc = metrics["AUC"]

        torch.save(model.state_dict(), f"checkpoints/model_epoch{epoch}.pt")

        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            torch.save(model.state_dict(), "checkpoints/model_best.pt")
            print(f"Epoch {epoch} | Loss={avg_loss:.4f} | LR={current_lr:.2e} | AUC={auc:.4f} ★ NEW BEST")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch} | Loss={avg_loss:.4f} | LR={current_lr:.2e} | AUC={auc:.4f} (no improvement for {epochs_no_improve})")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs (no improvement for {patience} consecutive epochs).")
            break

    print(f"\nBEST Epoch: {best_epoch} with AUC={best_auc:.4f}")

    # 加载最佳模型
    best_model = MultimodalModel(
        static_dim=dataset.static_feats.shape[1],
        ts_dim=dataset.ts_dim,
        dropout_rate=0.5
    ).to(device)
    best_model.load_state_dict(best_model_state if best_model_state is not None else torch.load("checkpoints/model_best.pt"))

    # 获取测试集上的预测概率和真实标签
    metrics, pos, neg, y_true, y_pred = evaluate(best_model, test_loader, device)
    print(f"Test set positive: {pos}, negative: {neg}")
    print(f"Metrics: {metrics}")

    # 计算 Brier 分数
    brier = brier_score_loss(y_true, y_pred)
    print(f"Brier Score: {brier:.4f}")

    # 绘制校准曲线
    plot_calibration_curve(y_true, y_pred, save_path="calibration_curve.png")

    # 绘制决策曲线
    plot_decision_curve(y_true, y_pred, save_path="decision_curve.png")

    # 将 Brier 分数加入结果并保存
    metrics['Brier'] = brier
    df = pd.DataFrame([metrics])
    df.to_csv("results_single_model.csv", index=False)
    print("Results saved to results_single_model.csv")

if __name__ == "__main__":
    train_model()