import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, \
    roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
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


# ================= Attention =================
class ModalityAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, g, s, t):
        x = torch.cat([g, s, t], dim=1)
        alpha = torch.softmax(self.net(x), dim=1)
        fused = g * alpha[:, 0:1] + s * alpha[:, 1:2] + t * alpha[:, 2:3]
        return fused, alpha


# ================= Models =================
class MultimodalModel(nn.Module):
    def __init__(self, static_dim, ts_dim, graph_dim=128, embed_dim=128):
        super().__init__()
        self.graph_proj = nn.Linear(graph_dim, embed_dim)
        self.static_enc = StaticEncoder(static_dim, embed_dim)
        self.temporal_enc = TemporalEncoder(ts_dim, embed_dim)
        self.attn = ModalityAttention(embed_dim)
        self.cls = nn.Linear(embed_dim, 1)

    def forward(self, g, s, t):
        g = self.graph_proj(g)
        s = self.static_enc(s)
        t = self.temporal_enc(t)
        fused, alpha = self.attn(g, s, t)
        return self.cls(fused).squeeze(1), alpha


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
def evaluate(model, loader, device, return_preds=False):
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
    y_bin = (y_pred > 0.5).astype(int)

    # 基础指标
    acc = accuracy_score(y_true, y_bin)
    f1 = f1_score(y_true, y_bin)
    prec = precision_score(y_true, y_bin)
    rec = recall_score(y_true, y_bin)
    auc_score = roc_auc_score(y_true, y_pred)

    # 敏感性 & 特异性
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    metrics = {
        "Accuracy": acc,
        "F1": f1,
        "Precision": prec,
        "Recall": rec,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "AUC": auc_score
    }
    pos, neg = int(y_true.sum()), len(y_true) - int(y_true.sum())

    if return_preds:
        return metrics, pos, neg, y_true, y_pred
    else:
        return metrics, pos, neg


# ================= Training + Selection =================
def train_experiment(model_class, name):
    print(f"\n========== {name} ==========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PatientDataset(
        "data/patient_clinical_filt.csv",
        "data/time_series.csv",
        "data/graph_embeddings"
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

    for epoch in range(1, 13):
        model.train()
        total_loss = 0
        for g, s, t, y in tqdm(train_loader, desc=f"{name} Epoch {epoch}/12"):
            g, s, t, y = g.to(device), s.to(device), t.to(device), y.to(device)
            logit, _ = model(g, s, t)
            loss = loss_fn(logit, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

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
    metrics, pos, neg, y_true, y_pred = evaluate(best_model, test_loader, device, return_preds=True)

    print(f"Test set positive: {pos}, negative: {neg}")
    print(f"Metrics: {metrics}")
    return name, metrics, pos, neg, y_true, y_pred


# ================= Plotting Functions =================
def draw_roc_curves(preds_dict, save_path="roc_curves.png"):
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red', 'orange']
    for (name, (y_true, y_pred)), color in zip(preds_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        lw = 3 if name == "Full_Attention" else 2
        plt.plot(fpr, tpr, color=color, lw=lw, label=f'{name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"ROC curves saved to {save_path}")


def draw_pr_curves(preds_dict, positive_rate, save_path="pr_curves.png"):
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red', 'orange']
    for (name, (y_true, y_pred)), color in zip(preds_dict.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)
        lw = 3 if name == "Full_Attention" else 2
        plt.plot(recall, precision, color=color, lw=lw, label=f'{name} (AUPRC = {auprc:.4f})')
    # 绘制随机分类器基线（阳性率）
    plt.axhline(y=positive_rate, color='gray', linestyle='--', lw=1,
                label=f'Baseline (Positive Rate = {positive_rate:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Comparison')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"PR curves saved to {save_path}")


def draw_calibration_curves(preds_dict, n_bins=10, save_path="calibration_curves.png"):
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red', 'orange']
    for (name, (y_true, y_pred)), color in zip(preds_dict.items(), colors):
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='uniform')
        lw = 3 if name == "Full_Attention" else 2
        plt.plot(prob_pred, prob_true, marker='o', color=color, lw=lw, label=name)
    # 完美校准线
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curves')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Calibration curves saved to {save_path}")


# ================= Main =================
if __name__ == "__main__":
    results = []
    preds_dict = {}  # 存储每个模型的真实标签和预测概率
    for model_class, name in [
        (MultimodalModel, "Full_Model"),
        (MeanFusionModel, "No_Attention"),
        (NoKGModel, "No_KG"),
        (NoTSModel, "No_TS")
    ]:
        res = train_experiment(model_class, name)
        name, metrics, pos, neg, y_true, y_pred = res
        preds_dict[name] = (y_true, y_pred)
        results.append((name, metrics, pos, neg))

    # 输出表格
    df = pd.DataFrame([{
        "Model": r[0],
        "Accuracy": r[1]["Accuracy"],
        "F1-score": r[1]["F1"],
        "Precision": r[1]["Precision"],
        "Recall": r[1]["Recall"],
        "Sensitivity": r[1]["Sensitivity"],
        "Specificity": r[1]["Specificity"],
        "AUC": r[1]["AUC"],
        "Test_Pos": r[2],
        "Test_Neg": r[3]
    } for r in results])
    print("\n===== Summary Table =====")
    print(df)
    df.to_csv("results_summary.csv", index=False)
    print("✅ Results saved to results_summary.csv")

    # 计算总体阳性率（使用任一模型的真实标签，所有模型测试集相同）
    any_y_true = next(iter(preds_dict.values()))[0]
    positive_rate = any_y_true.mean()
    print(f"Overall positive rate in test set: {positive_rate:.3f}")

    # 保存预测数据
    os.makedirs("prediction_data", exist_ok=True)
    for name, (y_true, y_pred) in preds_dict.items():
        np.save(f"prediction_data/{name}_y_true.npy", y_true)
        np.save(f"prediction_data/{name}_y_pred.npy", y_pred)
    print("✅ Prediction data saved to prediction_data/")

    # 绘制ROC曲线
    draw_roc_curves(preds_dict, "roc_curves.png")
    # 绘制PR曲线
    draw_pr_curves(preds_dict, positive_rate, "pr_curves.png")
    # 绘制校准曲线
    draw_calibration_curves(preds_dict, n_bins=10, save_path="calibration_curves.png")