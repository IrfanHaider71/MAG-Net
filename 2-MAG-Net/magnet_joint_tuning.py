
import os, random, warnings, copy
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn

SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

EXACT_INDEX_COLS = [
    "M1 (First Zagreb)", "M2 (Second Zagreb)", "mM2 (Modified Second Zagreb)",
    "ReM3 (Redefined M3)", "Harmonic Index", "Forgotten Index",
    "AZ (Augmented Zagreb)", "ISI (Inverse Sum Indeg)", "SDD (Symmetric Division Degree)"
]
INDEX_SHORT_NAMES = ["M1", "M2", "mM2", "ReM3", "H", "F", "AZ", "ISI", "SDD"]

N_CONFIGS = 25
SEARCH_EPOCHS, SEARCH_PATIENCE = 200, 25
FINAL_EPOCHS = 500

def clean_drug(df):
    df.columns = df.columns.astype(str).str.strip()
    if "Drug" not in df.columns:
        cand = [c for c in df.columns if c.lower().strip() == "drug"]
        df = df.rename(columns={cand[0]: "Drug"})
    df["Drug"] = df["Drug"].astype(str).str.strip()
    if "SMILES" in df.columns:
        df["SMILES"] = df["SMILES"].astype(str).str.strip()
    return df

def coerce_numeric_except(df, skip_cols):
    for c in df.columns:
        if c in skip_cols: continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def rmse_fn(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

def standardize(Xtr, Xte=None):
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-8
    Xtr_z = (Xtr - mu) / sd
    if Xte is None: return Xtr_z, mu, sd
    return Xtr_z, (Xte - mu) / sd, mu, sd

def mask_aware_target_stats(Ytr, Mtr):
    y_mu = Ytr.sum(0) / (Mtr.sum(0) + 1e-8)
    y_sd = torch.sqrt((((Ytr - y_mu) ** 2 * Mtr).sum(0) / (Mtr.sum(0) + 1e-8))) + 1e-8
    return y_mu, y_sd

class MAGNet(nn.Module):

    def __init__(self, din, dout, hidden=256, p_drop=0.10, n_layers=2):
        super().__init__()
        layers, d = [], din
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(p_drop)]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, dout)
        self.gate_logits = nn.Parameter(torch.zeros(dout, din))
        self.alpha = nn.Parameter(torch.ones(dout))
    def forward(self, x):
        h = self.trunk(x); base = self.out(h)
        W = torch.softmax(self.gate_logits, dim=1)
        z = x @ W.t()
        return base + self.alpha * z

def batcher(X, Y, M, bs, rng):
    n = X.shape[0]; idx = rng.permutation(n)
    for i in range(0, n, bs):
        j = idx[i:i+bs]; yield X[j], Y[j], M[j]

def train_model(Xtr_z, Ytr_z, Mtr, Xval, Yval_raw, Mval, y_mu, y_sd, din, dout,
                epochs, batch, lr, patience, hidden, p_drop, n_layers, weight_decay, seed=SEED):
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    model = MAGNet(din, dout, hidden=hidden, p_drop=p_drop, n_layers=n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.7, patience=10, min_lr=5e-5)
    best, best_state, bad = 1e30, None, 0
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb, mb in batcher(Xtr_z, Ytr_z, Mtr, batch, rng):
            opt.zero_grad()
            yhat = model(xb)
            loss = ((yhat-yb)**2*mb).sum()/(mb.sum()+1e-8)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        model.eval()
        with torch.no_grad():
            yv_hat = model(Xval)*y_sd+y_mu
            val_rmse = torch.sqrt((((yv_hat-Yval_raw)**2*Mval).sum())/(Mval.sum()+1e-8)).item()
        sched.step(val_rmse)
        if val_rmse+1e-8 < best:
            best_state = copy.deepcopy(model.state_dict()); best = val_rmse; bad = 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    return model, best

def predict(model, Xz, y_mu, y_sd):
    model.eval()
    with torch.no_grad():
        return (model(Xz)*y_sd+y_mu).numpy()

def sample_config(rng):
    return {
        "hidden":       int(rng.choice([128, 192, 256, 384, 512])),
        "n_layers":     int(rng.choice([2, 3])),
        "p_drop":       float(rng.choice([0.05, 0.10, 0.15, 0.20, 0.30])),
        "lr":           float(rng.choice([1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3])),
        "weight_decay": float(rng.choice([1e-5, 1e-4, 3e-4, 1e-3])),
        "batch":        int(rng.choice([16, 32, 48])),
    }

smiles   = clean_drug(pd.read_csv(F_SMILES))
exact_df = clean_drug(pd.read_csv(F_EXACT))
props    = clean_drug(pd.read_csv(F_PROPS))
exact_df = exact_df.rename(columns=dict(zip(EXACT_INDEX_COLS, INDEX_SHORT_NAMES)))[["Drug"] + INDEX_SHORT_NAMES]

df = smiles.merge(exact_df, on="Drug", how="inner").merge(props, on="Drug", how="inner")
df = df.drop_duplicates(subset=["Drug", "SMILES"]).reset_index(drop=True)
id_cols = ["Drug", "SMILES", "Class"]; pred_cols = INDEX_SHORT_NAMES
df = coerce_numeric_except(df, skip_cols=id_cols + pred_cols).dropna(subset=pred_cols).reset_index(drop=True)

desired_order = ["Molecular Weight (MW)", "Heavy Atom Count (HAC)", "Exact Mass (EM)",
                  "Complexity (C)", "Polarizability (P)", "Molar Volume (MV)", "Molar refractivity (MR)"]
prop_cols = [c for c in desired_order if c in df.columns]

idx_all = np.arange(len(df))
train_idx, test_idx = train_test_split(idx_all, test_size=0.2, random_state=SEED, shuffle=True)
df_train = df.iloc[train_idx].reset_index(drop=True)
df_test  = df.iloc[test_idx].reset_index(drop=True)
saved_test = pd.read_csv("cv_test_split.csv")
assert set(df_test["Drug"]) == set(saved_test["Drug"]), "Test split mismatch!"
print(f"Confirmed identical test split (N_test={len(df_test)}).")

X_full = df_train[pred_cols].values.astype(np.float32)
Y_full = df_train[prop_cols].values.astype(np.float32)
Xte_np = df_test[pred_cols].values.astype(np.float32)
Yte_np = df_test[prop_cols].values.astype(np.float32)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

print(f"\nSearching {N_CONFIGS} joint-model configs via 5-fold CV (this will take a while)...")
rng = np.random.RandomState(SEED)
search_results = []
for cfg_i in range(N_CONFIGS):
    cfg = sample_config(rng)
    fold_rmses = []
    for tr_i, va_i in kf.split(X_full):
        Xtr_t = torch.from_numpy(X_full[tr_i]); Ytr_t = torch.from_numpy(Y_full[tr_i])
        Mtr_t = torch.isfinite(Ytr_t).float(); Ytr_t[~torch.isfinite(Ytr_t)] = 0.0
        Xva_t = torch.from_numpy(X_full[va_i]); Yva_t = torch.from_numpy(Y_full[va_i])
        Mva_t = torch.isfinite(Yva_t).float(); Yva_t[~torch.isfinite(Yva_t)] = 0.0

        Xtr_z, Xva_z, mu_x, sd_x = standardize(Xtr_t, Xva_t)
        y_mu_f, y_sd_f = mask_aware_target_stats(Ytr_t, Mtr_t)
        Ytr_z = (Ytr_t - y_mu_f) / y_sd_f

        _, val_rmse = train_model(Xtr_z, Ytr_z, Mtr_t, Xva_z, Yva_t, Mva_t, y_mu_f, y_sd_f,
                                   din=len(pred_cols), dout=len(prop_cols),
                                   epochs=SEARCH_EPOCHS, patience=SEARCH_PATIENCE, **cfg)
        fold_rmses.append(val_rmse)
    mean_rmse = float(np.mean(fold_rmses))
    search_results.append((mean_rmse, cfg))
    print(f"  config {cfg_i+1:02d}/{N_CONFIGS}: {cfg} -> mean CV RMSE (masked, all properties) = {mean_rmse:.4f}")

search_results.sort(key=lambda t: t[0])
best_rmse, best_cfg = search_results[0]
print(f"\n[CV] Best joint-model config: {best_cfg} | mean CV RMSE = {best_rmse:.4f}")

pd.DataFrame([{"mean_cv_rmse": r, **c} for r, c in search_results]).to_csv("joint_tuning_search_log.csv", index=False)

Xtr_t = torch.from_numpy(X_full); Ytr_t = torch.from_numpy(Y_full)
Mtr_t = torch.isfinite(Ytr_t).float(); Ytr_t[~torch.isfinite(Ytr_t)] = 0.0
Xte_t = torch.from_numpy(Xte_np); Yte_t = torch.from_numpy(Yte_np)
Mte_t = torch.isfinite(Yte_t).float(); Yte_t[~torch.isfinite(Yte_t)] = 0.0

Xtr_z, Xte_z, x_mu, x_sd = standardize(Xtr_t, Xte_t)
y_mu, y_sd = mask_aware_target_stats(Ytr_t, Mtr_t)
Ytr_z = (Ytr_t - y_mu) / y_sd

n_all = Xtr_z.shape[0]
perm_all = np.random.RandomState(SEED).permutation(n_all)
val_n = max(1, int(0.15 * n_all))
val_idx2 = perm_all[:val_n]; tr2_idx = perm_all[val_n:]
X_tr2, Y_tr2, M_tr2 = Xtr_z[tr2_idx], Ytr_z[tr2_idx], Mtr_t[tr2_idx]
X_val2, Y_val2, M_val2 = Xtr_z[val_idx2], Ytr_t[val_idx2], Mtr_t[val_idx2]

final_model, _ = train_model(X_tr2, Y_tr2, M_tr2, X_val2, Y_val2, M_val2, y_mu, y_sd,
                              din=len(pred_cols), dout=len(prop_cols),
                              epochs=FINAL_EPOCHS, patience=60, **best_cfg)

yhat_test = predict(final_model, Xte_z, y_mu, y_sd)

rows = []
for j, prop in enumerate(prop_cols):
    m = Mte_t[:, j].numpy().astype(bool)
    yt = Yte_np[m, j]; yp = yhat_test[m, j]
    r2 = r2_score(yt, yp); mae = mean_absolute_error(yt, yp); rmse = rmse_fn(yt, yp)
    pr, _ = pearsonr(yt, yp); sr, _ = spearmanr(yt, yp)
    rows.append({"Property": prop, "R2": r2, "MAE": mae, "RMSE": rmse, "pearson_r": pr, "spearman_rho": sr})

tuned_df = pd.DataFrame(rows)
tuned_df.to_csv("magnet_joint_tuned_metrics.csv", index=False)

print("\n=== TUNED joint model (held-out test) ===")
print(tuned_df.round(4).to_string(index=False))

# ---------------- honest comparison against original fixed-config model ----------------
orig = pd.read_csv("proposed_model_metrics.csv")[["Property", "R2", "MAE", "RMSE"]]
orig = orig.rename(columns={"R2": "R2_orig", "MAE": "MAE_orig", "RMSE": "RMSE_orig"})
comp = tuned_df.merge(orig, on="Property")
comp["dR2"] = comp["R2"] - comp["R2_orig"]
print("\n=== COMPARISON: tuned joint model vs original fixed-config model ===")
print(comp[["Property", "R2_orig", "R2", "dR2", "MAE_orig", "MAE", "RMSE_orig", "RMSE"]].round(4).to_string(index=False))
print(f"\nProperties improved: {(comp['dR2']>0.001).sum()} / 7")
print(f"Properties worsened: {(comp['dR2']<-0.001).sum()} / 7")
print(f"Mean R2 -- original: {comp['R2_orig'].mean():.4f} | tuned: {comp['R2'].mean():.4f}")

# save per-drug predictions too, for potential Table 7 regeneration
pred_rows = []
for j, prop in enumerate(prop_cols):
    m = Mte_t[:, j].numpy().astype(bool)
    for d, s, yt, yp in zip(df_test.loc[m, "Drug"].values, df_test.loc[m, "SMILES"].values,
                             Yte_np[m, j], yhat_test[m, j]):
        pred_rows.append({"Drug": d, "SMILES": s, "Property": prop, "y_true": float(yt), "y_pred": float(yp)})
pd.DataFrame(pred_rows).to_csv("magnet_joint_tuned_predictions_long.csv", index=False)
print("\nSaved -> magnet_joint_tuned_metrics.csv, magnet_joint_tuned_predictions_long.csv, joint_tuning_search_log.csv")
