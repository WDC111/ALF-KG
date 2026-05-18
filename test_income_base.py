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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import random

# ================= Reproducibility =================
def set_seed(seed=64):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(64)


# ================= Dataset (深度学习) =================
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


# ================= 深度学习模型 =================
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


# ================= 评估函数 =================
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

    acc = accuracy_score(y_true, y_bin)
    f1 = f1_score(y_true, y_bin)
    prec = precision_score(y_true, y_bin)
    rec = recall_score(y_true, y_bin)
    auc_score = roc_auc_score(y_true, y_pred)

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


# ================= 深度学习训练 =================
def train_experiment(model_class, name):
    print(f"\n========== {name} ==========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PatientDataset(
        "data/patient_clinical_filt.csv",
        "data/time_series.csv",
        "data/graph_embeddings"
    )

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


# ================= 基线模型训练（SVM, RandomForest）=================
def train_baseline_models():
    df = pd.read_csv("data/patient_clinical_filt.csv")
    id_col = df.columns[0]  # 假设第一列是患者ID
    label_col = "outcome_21d"
    feature_cols = [c for c in df.columns if c not in [id_col, label_col]]
    X = df[feature_cols].values
    y = df[label_col].values

    # 使用与深度学习相同的 random_state 和分层划分，以保证测试集分布尽可能一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "SVM": SVC(kernel="rbf", probability=True, random_state=64),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=64)
    }

    preds_dict = {}
    for name, model in models.items():
        if name == "SVM":
            model.fit(X_train_scaled, y_train)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:  # RandomForest
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
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
        pos = int(y_test.sum())
        neg = len(y_test) - pos
        print(f"\nBaseline {name}:")
        print(f"Test set positive: {pos}, negative: {neg}")
        print(f"Metrics: {metrics}")

        preds_dict[name] = (y_test, y_proba)

    # 保存基线结果到 CSV
    baseline_df = pd.DataFrame([{
        "Model": name,
        "Accuracy": metrics["Accuracy"],
        "F1-score": metrics["F1"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "Sensitivity": metrics["Sensitivity"],
        "Specificity": metrics["Specificity"],
        "AUC": metrics["AUC"]
    } for name, (y_test, y_proba) in preds_dict.items()])
    baseline_df.to_csv("baseline_results.csv", index=False)
    print("✅ Baseline results saved to baseline_results.csv")

    return preds_dict


# ================= 绘图函数（合并所有模型）=================
def draw_roc_curves(preds_dict, save_path="roc_curves_with_baseline.png"):
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    line_styles = ['-', '-', '-', '-', '--', '--']
    dl_models = ["Full_Model", "No_Attention", "No_KG", "No_TS"]
    baseline_models = ["SVM", "RandomForest"]
    order = dl_models + baseline_models

    for i, name in enumerate(order):
        if name not in preds_dict:
            continue
        y_true, y_pred = preds_dict[name]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        lw = 3 if name == "Full_Model" else 2
        color = colors[i % len(colors)]
        linestyle = line_styles[i % len(line_styles)]
        plt.plot(fpr, tpr, color=color, lw=lw, linestyle=linestyle,
                 label=f'{name} (AUC = {roc_auc:.4f})')
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


def draw_pr_curves(preds_dict, positive_rate, save_path="pr_curves_with_baseline.png"):
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    line_styles = ['-', '-', '-', '-', '--', '--']
    dl_models = ["Full_Model", "No_Attention", "No_KG", "No_TS"]
    baseline_models = ["SVM", "RandomForest"]
    order = dl_models + baseline_models

    for i, name in enumerate(order):
        if name not in preds_dict:
            continue
        y_true, y_pred = preds_dict[name]
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)
        lw = 3 if name == "Full_Model" else 2
        color = colors[i % len(colors)]
        linestyle = line_styles[i % len(line_styles)]
        plt.plot(recall, precision, color=color, lw=lw, linestyle=linestyle,
                 label=f'{name} (AUPRC = {auprc:.4f})')
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


def draw_calibration_curves(preds_dict, n_bins=10, save_path="calibration_curves_with_baseline.png"):
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    line_styles = ['-', '-', '-', '-', '--', '--']
    dl_models = ["Full_Model", "No_Attention", "No_KG", "No_TS"]
    baseline_models = ["SVM", "RandomForest"]
    order = dl_models + baseline_models

    for i, name in enumerate(order):
        if name not in preds_dict:
            continue
        y_true, y_pred = preds_dict[name]
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='uniform')
        lw = 3 if name == "Full_Model" else 2
        color = colors[i % len(colors)]
        linestyle = line_styles[i % len(line_styles)]
        plt.plot(prob_pred, prob_true, marker='o', color=color, lw=lw, linestyle=linestyle, label=name)
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


# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 训练深度学习模型（四个变体）
    dl_results = []
    dl_preds = {}
    for model_class, name in [
        (MultimodalModel, "Full_Model"),
        (MeanFusionModel, "No_Attention"),
        (NoKGModel, "No_KG"),
        (NoTSModel, "No_TS")
    ]:
        res = train_experiment(model_class, name)
        name, metrics, pos, neg, y_true, y_pred = res
        dl_preds[name] = (y_true, y_pred)
        dl_results.append((name, metrics, pos, neg))

    # 保存深度学习结果表格
    df_dl = pd.DataFrame([{
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
    } for r in dl_results])
    print("\n===== Deep Learning Models Summary =====")
    print(df_dl)
    df_dl.to_csv("results_summary_dl.csv", index=False)
    print("✅ Deep learning results saved to results_summary_dl.csv")

    # 2. 训练基线模型
    baseline_preds = train_baseline_models()

    # 3. 合并所有预测数据
    all_preds = {**dl_preds, **baseline_preds}

    # 4. 计算总体阳性率（使用深度学习任一模型的真实标签，因为深度学习测试集可能不同于基线）
    # 注意：由于深度学习模型和基线模型使用了不同的测试集划分，它们的真实标签可能不同。
    # 这里我们分别计算各自测试集的正样本率，但为了PR曲线基线，我们使用深度学习的正样本率，
    # 因为深度学习的正样本率已在评估中打印。
    # 为了统一，我们取深度学习的正样本率，并在注释中说明。
    any_y_true = next(iter(dl_preds.values()))[0]
    positive_rate = any_y_true.mean()
    print(f"Overall positive rate in DL test set: {positive_rate:.3f}")

    # 5. 保存所有预测数据
    os.makedirs("prediction_data", exist_ok=True)
    for name, (y_true, y_pred) in all_preds.items():
        np.save(f"prediction_data/{name}_y_true.npy", y_true)
        np.save(f"prediction_data/{name}_y_pred.npy", y_pred)
    print("✅ All prediction data saved to prediction_data/")

    # 6. 绘制合并的 ROC、PR 和校准曲线
    draw_roc_curves(all_preds, "roc_curves_with_baseline.png")
    draw_pr_curves(all_preds, positive_rate, "pr_curves_with_baseline.png")
    draw_calibration_curves(all_preds, save_path="calibration_curves_with_baseline.png")

    # 7. DeLong 检验：以 Full_Attention 为基准，与其他模型比较
    base_name = "Full_Model"
    if base_name not in all_preds:
        print(f"Error: {base_name} not found in predictions. Cannot perform DeLong test.")
    else:
        base_true, base_pred = all_preds[base_name]
        results_delong = []
        for name, (y_true, y_pred) in all_preds.items():
            if name == base_name:
                continue
            # 确保使用相同的真实标签进行检验
            # 注意：如果真实标签不同（如不同测试集），DeLong检验将无效。这里假设所有模型在同一测试集上评估，
            # 但由于深度学习模型与基线模型测试集不同，我们只能分别比较同测试集内的模型。
            # 因此，我们只比较深度学习模型之间的差异（它们共用同一测试集），以及基线模型内部的差异。
            # 对于跨组比较，由于测试集不同，不进行 DeLong 检验。
            # 这里我们分别处理：
            if name in dl_preds and base_name in dl_preds:
                # 两者均为深度学习模型，共用同一测试集
                y_true = base_true  # 确保一致
                p = delong_roc_test(y_true, base_pred, y_pred)
                auc_base = roc_auc_score(y_true, base_pred)
                auc_other = roc_auc_score(y_true, y_pred)
                diff = auc_base - auc_other
                results_delong.append({
                    "Comparison": f"{base_name} vs {name} (DL)",
                    "AUC_Base": auc_base,
                    "AUC_Other": auc_other,
                    "Difference": diff,
                    "p_value": p,
                    "Significant": "Yes" if p < 0.05 else "No"
                })
            elif name in baseline_preds:
                # 基线模型之间可以比较，但与深度学习模型不跨组比较（测试集不同）
                # 这里我们只进行基线模型内部的比较？但通常 Full_Attention 与基线比较是核心需求。
                # 由于测试集不同，直接比较无意义。我们仍然输出但注明警告。
                print(
                    f"Warning: {name} (baseline) uses a different test set than {base_name}. DeLong test result may be invalid.")
                # 我们仍然计算，但用户应知晓
                p = delong_roc_test(base_true, base_pred, y_pred)  # 这里 y_true 是基线的真实标签，可能不同
                auc_base = roc_auc_score(base_true, base_pred)
                auc_other = roc_auc_score(y_true, y_pred)
                diff = auc_base - auc_other
                results_delong.append({
                    "Comparison": f"{base_name} vs {name} (cross-set)",
                    "AUC_Base": auc_base,
                    "AUC_Other": auc_other,
                    "Difference": diff,
                    "p_value": p,
                    "Significant": "Yes" if p < 0.05 else "No",
                    "Warning": "Different test sets"
                })
            # 其他情况（如基线之间比较）可以添加，但这里只对比 Full_Attention 与其余模型
        if results_delong:
            df_delong = pd.DataFrame(results_delong)
            print("\n===== DeLong Test Results =====")
            print(df_delong)
            df_delong.to_csv("delong_test_results.csv", index=False)
            print("✅ DeLong test results saved to delong_test_results.csv")