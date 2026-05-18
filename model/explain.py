import torch
import pandas as pd
import numpy as np
import shap
from torch.utils.data import DataLoader
from train import PatientDataset, MultimodalModel


def extract_modality_attention(model, loader, device):
    model.eval()
    attn_list = []
    with torch.no_grad():
        for g, s, t, y in loader:
            g, s, t = g.to(device), s.to(device), t.to(device)
            _, alpha = model(g, s, t)
            if alpha is not None:
                attn_list.append(alpha.cpu())
    if attn_list:
        return torch.cat(attn_list, dim=0).numpy()
    else:
        return np.array([])


def compute_shap_static_incremental(
    model,
    dataset,
    device,
    batch_size=100,
    nsamples=100,
    csv_path="shap_static_values.csv"
):

    model.eval()

    feature_names = dataset.static_df.drop(
        columns=["hadm_id", "outcome_21d"]
    ).columns.tolist()

    background = torch.tensor(
        dataset.static_feats[:batch_size],
        dtype=torch.float32
    ).to(device)

    def model_forward(x_numpy):
        x = torch.tensor(x_numpy, dtype=torch.float32).to(device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        B = x.shape[0]

        g_dummy = torch.zeros((B, 128), device=device)
        t_dummy = torch.zeros((B, 1, dataset.ts_dim), device=device)

        g_emb = model.graph_proj(g_dummy)
        s_emb = model.static_enc(x)
        t_emb = model.temporal_enc(t_dummy)

        fused, _ = model.attn(g_emb, s_emb, t_emb)
        logit = model.cls(fused).squeeze(1)

        return torch.sigmoid(logit).detach().cpu().numpy()

    explainer = shap.KernelExplainer(
        model_forward,
        background.detach().cpu().numpy()
    )

    n = len(dataset)
    first_write = True

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        static_batch = torch.tensor(
            dataset.static_feats[start:end],
            dtype=torch.float32
        ).to(device)

        hadm_ids = dataset.static_df.iloc[start:end]["hadm_id"].values

        shap_vals = explainer.shap_values(
            static_batch.detach().cpu().numpy(),
            nsamples=nsamples
        )

        df = pd.DataFrame(shap_vals, columns=feature_names)
        df.insert(0, "hadm_id", hadm_ids)

        df.to_csv(
            csv_path,
            mode="w" if first_write else "a",
            header=first_write,
            index=False
        )

        first_write = False
        print(f"SHAP Static processed: {end}/{n}")



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PatientDataset(
        "data/patient_clinical_filt.csv",
        "data/time_series.csv",
        "data/graph_embeddings"
    )
    loader = DataLoader(dataset, batch_size=32)

    model = MultimodalModel(
        static_dim=dataset.static_feats.shape[1],
        ts_dim=dataset.ts_dim
    ).to(device)
    model.load_state_dict(torch.load("checkpoints/Full_Attention_best.pt", map_location=device))
    model.eval()

    # 1. Modality Attention
    attn = extract_modality_attention(model, loader, device)
    df_attn = pd.DataFrame(attn, columns=["KG", "Static", "Temporal"])
    df_attn.to_csv("modality_attention.csv", index=False)
    print(df_attn.describe())

    # 2. SHAP Static
    compute_shap_static_incremental(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=100,
        nsamples=100,
        csv_path="shap_static_values.csv"
    )

