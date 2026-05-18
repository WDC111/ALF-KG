import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm


class LiverDataset(Dataset):
    def __init__(self, ts_csv, clinical_csv):
        self.ts_df = pd.read_csv(ts_csv)
        self.clinical_df = pd.read_csv(clinical_csv)

        self.hadm_ids = self.clinical_df['hadm_id'].values
        self.labels = self.clinical_df['outcome_21d'].values.astype(np.float32)

        self.feature_cols = [
            c for c in self.ts_df.columns if c not in ['hadm_id', 'hour']
        ]

        self.seq_len = self.ts_df['hour'].nunique()

    def __len__(self):
        return len(self.hadm_ids)

    def __getitem__(self, idx):
        hadm_id = self.hadm_ids[idx]
        ts = self.ts_df[self.ts_df['hadm_id'] == hadm_id].sort_values('hour')
        ts_feat = ts[self.feature_cols].values.astype(np.float32)
        return torch.tensor(ts_feat), torch.tensor(self.labels[idx])


class LiverLSTM(nn.Module):
    def __init__(self, input_dim=49, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 只用最后一个时间步
        return torch.sigmoid(out).squeeze(1)


def train_model(model, train_loader, val_loader, device, epochs=20):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                p = model(x)
                preds.extend(p.cpu().numpy())
                labels.extend(y.cpu().numpy())

        auc = roc_auc_score(labels, preds)
        acc = accuracy_score(labels, np.round(preds))
        print(f"Epoch {ep+1}: AUC={auc:.4f}, Acc={acc:.4f}")

    return model


def explain_shap_last_hour(model, dataset, device):
    model.eval()

    bg_idx = np.random.choice(len(dataset), 50, replace=False)
    background = []
    for i in bg_idx:
        ts, _ = dataset[i]
        background.append(ts[-1].numpy())  # (49for 49 features

    background = np.array(background)  # (50, 49)

    def model_forward_last_hour(x):
        x = torch.tensor(x, dtype=torch.float32).to(device)
        x = x.unsqueeze(1)  # (N,1,49)
        with torch.no_grad():
            out = model(x)
        return out.cpu().numpy()

    explainer = shap.KernelExplainer(model_forward_last_hour, background)

    out_csv = "shap_ts.csv"
    header_written = False

    for i in tqdm(range(len(dataset)), desc="SHAP patients"):
        ts_feat, _ = dataset[i]
        last_hour = ts_feat[-1].numpy()  # (49,)

        shap_values = explainer.shap_values(
            last_hour.reshape(1, -1), nsamples=100
        )[0]

        df = pd.DataFrame(
            [shap_values],
            columns=dataset.feature_cols
        )
        df["hadm_id"] = dataset.hadm_ids[i]
        df["hour"] = dataset.seq_len - 1

        if not header_written:
            df.to_csv(out_csv, index=False)
            header_written = True
        else:
            df.to_csv(out_csv, mode="a", header=False, index=False)


# =========================
# 5. Main
# =========================
if __name__ == "__main__":
    TS_CSV = "data/time_series.csv"
    CLINICAL_CSV = "data/patient_clinical_filt.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = LiverDataset(TS_CSV, CLINICAL_CSV)

    idx_train, idx_val = train_test_split(
        np.arange(len(dataset)), test_size=0.2, random_state=42
    )

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, idx_train),
        batch_size=16, shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, idx_val),
        batch_size=16
    )

    model = LiverLSTM(input_dim=len(dataset.feature_cols)).to(device)
    model = train_model(model, train_loader, val_loader, device)

    explain_shap_last_hour(model, dataset, device)

