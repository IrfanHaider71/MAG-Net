
import os, random, warnings, copy
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cpu")

INDEX_SHORT_NAMES = ["M1", "M2", "mM2", "ReM3", "H", "F", "AZ", "ISI", "SDD"]

K_FOLDS          = 5     
N_YSCRAMBLE      = 200    
YSCRAMBLE_EPOCHS = 200    
EPOCHS           = 500    
BATCH            = 32
LR               = 2e-3
PATIENCE         = 60
HIDDEN           = 256
P_DROP           = 0.10


def std_cols(df):
    df.columns = df.columns.astype(str).str.strip()
    return df


def clean_drug(df):
    df = std_cols(df)
    if "Drug" not in df.columns:
        cand = [c for c in df.columns if c.lower().strip() == "drug"]
        if cand:
            df = df.rename(columns={cand[0]: "Drug"})
        else:
            raise ValueError("Column 'Drug' not found.")
    df["Drug"] = df["Drug"].astype(str).str.strip()
    if "SMILES" in df.columns:
        df["SMILES"] = df["SMILES"].astype(str).str.strip()
    if "Class" in df.columns:
        df["Class"] = df["Class"].astype(str).str.strip()
    return df


def coerce_numeric_except(df, skip_cols):
    for c in df.columns:
        if c in skip_cols:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def rmse_fn(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))


def safe_corr(y_true, y_pred):
    x = np.asarray(y_true, float)
    y = np.asarray(y_pred, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return np.nan, np.nan, np.nan, np.nan
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return float(pr), float(pp), float(sr), float(sp)


def standardize(Xtr, Xte=None):
    """Z-score normalise; returns z-scored tensors + (mu, sd)."""
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True) + 1e-8
    Xtr_z = (Xtr - mu) / sd
    if Xte is None:
        return Xtr_z, mu, sd
    return Xtr_z, (Xte - mu) / sd, mu, sd


def mask_aware_target_stats(Ytr, Mtr):
    """Compute mask-aware mean and std for each target column."""
    y_mu = Ytr.sum(0) / (Mtr.sum(0) + 1e-8)
    y_sd = torch.sqrt(
        (((Ytr - y_mu) ** 2 * Mtr).sum(0) / (Mtr.sum(0) + 1e-8))
    ) + 1e-8
    return y_mu, y_sd


class MAGNet(nn.Module):
    def __init__(self, din, dout, hidden=256, p_drop=0.10):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(din, hidden), nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.out         = nn.Linear(hidden, dout)
        self.gate_logits = nn.Parameter(torch.zeros(dout, din))
        self.alpha       = nn.Parameter(torch.ones(dout))

    def forward(self, x):
        h    = self.trunk(x)
        base = self.out(h)
        W    = torch.softmax(self.gate_logits, dim=1)
        z    = x @ W.t()
        return base + self.alpha * z


def batcher(X, Y, M, bs, rng=None):
    n   = X.shape[0]
    idx = (np.random.permutation(n) if rng is None
           else rng.permutation(n))
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield X[j], Y[j], M[j]


def train_model(Xtr_z, Ytr_z, Mtr,
                Xval, Yval_raw, Mval,
                y_mu, y_sd,
                din, dout,
                epochs=EPOCHS, batch=BATCH, lr=LR,
                patience=PATIENCE, hidden=HIDDEN, p_drop=P_DROP,
                seed=SEED, verbose=False):
    """
    Train MAGNet and return (model, best_val_rmse).
    Xval / Yval_raw are evaluated in the *original* (un-z-scored) scale.
    """
    rng   = np.random.RandomState(seed)
    model = MAGNet(din, dout, hidden=hidden, p_drop=p_drop)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.7, patience=10, min_lr=5e-5)

    best, best_state, bad = 1e30, None, 0

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb, mb in batcher(Xtr_z, Ytr_z, Mtr, batch, rng):
            opt.zero_grad()
            yhat = model(xb)
            loss = ((yhat - yb) ** 2 * mb).sum() / (mb.sum() + 1e-8)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot += loss.item() * xb.size(0)

        model.eval()
        with torch.no_grad():
            yv_hat   = model(Xval) * y_sd + y_mu
            val_rmse = torch.sqrt(
                (((yv_hat - Yval_raw) ** 2 * Mval).sum() /
                 (Mval.sum() + 1e-8))
            ).item()

        sched.step(val_rmse)

        if verbose and ep % 50 == 0:
            print(f"    Ep {ep:03d} | train_MSE {tot/Xtr_z.shape[0]:.5f} "
                  f"| val_RMSE {val_rmse:.5f}")

        if val_rmse + 1e-8 < best:
            best_state = copy.deepcopy(model.state_dict())
            best       = val_rmse
            bad        = 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print("    Early stopping.")
                break

    model.load_state_dict(best_state)
    return model, best


def predict(model, Xte_z, y_mu, y_sd):
    model.eval()
    with torch.no_grad():
        return (model(Xte_z) * y_sd + y_mu).numpy()

import os as _os

for f in [F_SMILES, F_EXACT, F_PROPS]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing input file: {f}")

smiles   = clean_drug(pd.read_csv(F_SMILES))
exact_df = clean_drug(pd.read_csv(F_EXACT))
props    = clean_drug(pd.read_csv(F_PROPS))

exact_df = exact_df.rename(columns=dict(zip(EXACT_INDEX_COLS, INDEX_SHORT_NAMES)))
exact_df = exact_df[["Drug"] + INDEX_SHORT_NAMES]
print("Using index columns:", INDEX_SHORT_NAMES)

df = (smiles
      .merge(exact_df, on="Drug", how="inner")
      .merge(props,    on="Drug", how="inner"))
df = df.drop_duplicates(subset=["Drug", "SMILES"]).reset_index(drop=True)

id_cols   = ["Drug", "SMILES", "Class"]
pred_cols = INDEX_SHORT_NAMES

df = coerce_numeric_except(df, skip_cols=id_cols + pred_cols)
df = df.dropna(subset=pred_cols).reset_index(drop=True)

prop_cols = [c for c in df.columns
             if c not in set(id_cols + pred_cols)
             and pd.api.types.is_numeric_dtype(df[c])]
if not prop_cols:
    raise ValueError("No numeric property columns detected.")

print(f"Detected properties : {prop_cols}")
print(f"Final dataset size  : {len(df)} rows\n")


idx_all = np.arange(len(df))
train_idx, test_idx = train_test_split(
    idx_all, test_size=0.2, random_state=SEED, shuffle=True)

df_train = df.iloc[train_idx].reset_index(drop=True)
df_test  = df.iloc[test_idx ].reset_index(drop=True)

assert not (set(df_train["Drug"])   & set(df_test["Drug"])),   "Drug overlap!"
assert not (set(df_train["SMILES"]) & set(df_test["SMILES"])), "SMILES overlap!"
print("Train/Test entity overlap: NONE ")

cols_save = ["Drug", "SMILES"] + (["Class"] if "Class" in df_test.columns else [])
df_test[cols_save].to_csv("cv_test_split.csv", index=False)
print(f"Saved -> cv_test_split.csv  (N_test={len(df_test)})\n")

X_full = df_train[pred_cols].values.astype(np.float32)
Y_full = df_train[prop_cols].values.astype(np.float32)
Xte_np = df_test [pred_cols].values.astype(np.float32)
Yte_np = df_test [prop_cols].values.astype(np.float32)

print("=" * 62)
print(f"  SECTION 1 :  {K_FOLDS}-FOLD CROSS-VALIDATION")
print("=" * 62)

kf         = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
kfold_rows = []

for fold, (tr_i, va_i) in enumerate(kf.split(X_full), 1):
    print(f"\n--- Fold {fold}/{K_FOLDS}  "
          f"(train={len(tr_i)}, val={len(va_i)}) ---")

    Xtr_t = torch.from_numpy(X_full[tr_i])
    Ytr_t = torch.from_numpy(Y_full[tr_i])
    Mtr_t = torch.isfinite(Ytr_t).float()
    Ytr_t[~torch.isfinite(Ytr_t)] = 0.0

    Xva_t = torch.from_numpy(X_full[va_i])
    Yva_t = torch.from_numpy(Y_full[va_i])
    Mva_t = torch.isfinite(Yva_t).float()
    Yva_t[~torch.isfinite(Yva_t)] = 0.0

    Xtr_z, Xva_z, mu_x, sd_x = standardize(Xtr_t, Xva_t)
    y_mu_f, y_sd_f            = mask_aware_target_stats(Ytr_t, Mtr_t)
    Ytr_z                     = (Ytr_t - y_mu_f) / y_sd_f

    model_fold, _ = train_model(
        Xtr_z, Ytr_z, Mtr_t,
        Xva_z, Yva_t, Mva_t,
        y_mu_f, y_sd_f,
        din=len(pred_cols), dout=len(prop_cols),
        seed=SEED + fold, verbose=True
    )

    yhat_va = predict(model_fold, Xva_z, y_mu_f, y_sd_f)

    for j, prop in enumerate(prop_cols):
        m = Mva_t[:, j].numpy().astype(bool)
        if m.sum() < 3:
            continue
        yt = Y_full[va_i][m, j]
        yp = yhat_va[m, j]
        pr, pp, sr, sp = safe_corr(yt, yp)
        kfold_rows.append({
            "Fold":         fold,
            "Property":     prop,
            "R2":           float(r2_score(yt, yp)),
            "MAE":          float(mean_absolute_error(yt, yp)),
            "RMSE":         float(rmse_fn(yt, yp)),
            "pearson_r":    pr,
            "pearson_p":    pp,
            "spearman_rho": sr,
            "spearman_p":   sp,
            "N_val":        int(m.sum())
        })

kfold_df = pd.DataFrame(kfold_rows)
kfold_df.to_csv("kfold_cv_metrics.csv", index=False)
print("\nSaved -> kfold_cv_metrics.csv")

kfold_summary = (kfold_df
                 .groupby("Property")[["R2", "MAE", "RMSE", "pearson_r"]]
                 .agg(["mean", "std"])
                 .round(4))
print("\nK-Fold CV Summary (mean ± std across folds):")
print(kfold_summary.to_string())

Xtr_t = torch.from_numpy(X_full)
Ytr_t = torch.from_numpy(Y_full)
Mtr_t = torch.isfinite(Ytr_t).float()
Ytr_t[~torch.isfinite(Ytr_t)] = 0.0

Xte_t = torch.from_numpy(Xte_np)
Yte_t = torch.from_numpy(Yte_np)
Mte_t = torch.isfinite(Yte_t).float()
Yte_t[~torch.isfinite(Yte_t)] = 0.0

Xtr_z, Xte_z, x_mu, x_sd = standardize(Xtr_t, Xte_t)
y_mu, y_sd                = mask_aware_target_stats(Ytr_t, Mtr_t)
Ytr_z                     = (Ytr_t - y_mu) / y_sd

n_all    = Xtr_z.shape[0]
perm_all = np.random.RandomState(SEED).permutation(n_all)
val_n    = max(1, int(0.15 * n_all))
val_idx2 = perm_all[:val_n]
tr2_idx  = perm_all[val_n:]

X_tr2  = Xtr_z[tr2_idx];   Y_tr2  = Ytr_z[tr2_idx];  M_tr2  = Mtr_t[tr2_idx]
X_val2 = Xtr_z[val_idx2];  Y_val2 = Ytr_t[val_idx2]; M_val2 = Mtr_t[val_idx2]

print("\n" + "=" * 62)
print(f"  SECTION 2 :  Y-SCRAMBLING  ({N_YSCRAMBLE} permutations)")
print("=" * 62)

print("\nTraining REAL model for Y-scramble baseline ...")
model_real, _ = train_model(
    X_tr2, Y_tr2, M_tr2,
    X_val2, Y_val2, M_val2,
    y_mu, y_sd,
    din=len(pred_cols), dout=len(prop_cols),
    verbose=True
)
yhat_real = predict(model_real, Xte_z, y_mu, y_sd)

real_r2, real_rmse = {}, {}
for j, prop in enumerate(prop_cols):
    m = Mte_t[:, j].numpy().astype(bool)
    if m.sum() < 3:
        continue
    yt = Yte_np[m, j]
    yp = yhat_real[m, j]
    real_r2[prop]   = float(r2_score(yt, yp))
    real_rmse[prop] = float(rmse_fn(yt, yp))

scramble_rows = []

for perm_i in range(N_YSCRAMBLE):
    rng_s      = np.random.RandomState(SEED + 1000 + perm_i)
    perm_order = rng_s.permutation(Ytr_t.shape[0])

    Ytr_s  = Ytr_t[perm_order]
    Mtr_s  = Mtr_t[perm_order]

    y_mu_s, y_sd_s = mask_aware_target_stats(Ytr_s, Mtr_s)
    Ytr_sz         = (Ytr_s - y_mu_s) / y_sd_s

    Y_tr2_s  = Ytr_sz[tr2_idx]
    M_tr2_s  = Mtr_s[tr2_idx]
    Y_val2_s = Ytr_s[val_idx2]
    M_val2_s = Mtr_s[val_idx2]

    model_s, _ = train_model(
        X_tr2, Y_tr2_s, M_tr2_s,
        X_val2, Y_val2_s, M_val2_s,
        y_mu_s, y_sd_s,
        din=len(pred_cols), dout=len(prop_cols),
        seed=SEED + 1000 + perm_i,
        verbose=False,
        epochs=YSCRAMBLE_EPOCHS          # faster scramble runs
    )
    yhat_s = predict(model_s, Xte_z, y_mu_s, y_sd_s)

    for j, prop in enumerate(prop_cols):
        m = Mte_t[:, j].numpy().astype(bool)
        if m.sum() < 3:
            continue
        yt = Yte_np[m, j]
        yp = yhat_s[m, j]
        scramble_rows.append({
            "Permutation":   perm_i + 1,
            "Property":      prop,
            "R2_scramble":   float(r2_score(yt, yp)),
            "RMSE_scramble": float(rmse_fn(yt, yp))
        })

    if (perm_i + 1) % 20 == 0 or (perm_i + 1) == N_YSCRAMBLE:
        print(f"  Y-scramble {perm_i+1:03d}/{N_YSCRAMBLE} done")

scramble_df = pd.DataFrame(scramble_rows)

scr_agg = (scramble_df
           .groupby("Property")
           .agg(
               R2_scramble_mean  =("R2_scramble",   "mean"),
               R2_scramble_std   =("R2_scramble",   "std"),
               R2_scramble_max   =("R2_scramble",   "max"),
               RMSE_scramble_mean=("RMSE_scramble",  "mean"),
               RMSE_scramble_std =("RMSE_scramble",  "std"),
           )
           .reset_index())

scr_agg["R2_real"]          = scr_agg["Property"].map(real_r2)
scr_agg["RMSE_real"]        = scr_agg["Property"].map(real_rmse)
scr_agg["delta_R2"]         = scr_agg["R2_real"] - scr_agg["R2_scramble_mean"]
scr_agg["passed_yscramble"] = scr_agg["delta_R2"] > 0.05

scr_agg.to_csv("yscramble_summary.csv", index=False)
print("\nSaved -> yscramble_summary.csv")
print(scr_agg[["Property", "R2_real", "R2_scramble_mean",
               "R2_scramble_max", "delta_R2", "passed_yscramble"]]
      .to_string(index=False))
print("\n" + "=" * 62)
print("  SECTION 3 :  CHEMICAL SPACE DIVERSITY")
print("=" * 62)

scaler_cs = StandardScaler()
X_tr_sc   = scaler_cs.fit_transform(X_full)
X_te_sc   = scaler_cs.transform(Xte_np)

pca      = PCA(n_components=min(len(pred_cols), 2), random_state=SEED)
X_tr_pca = pca.fit_transform(X_tr_sc)
X_te_pca = pca.transform(X_te_sc)

ev = pca.explained_variance_ratio_
print(f"\nPCA explained variance : PC1={ev[0]:.3f}  PC2={ev[1]:.3f}")

nn_dists_te = cdist(X_te_sc, X_tr_sc, metric="euclidean").min(axis=1)

d_intra = cdist(X_tr_sc, X_tr_sc, metric="euclidean")
np.fill_diagonal(d_intra, np.inf)
nn_dists_tr = d_intra.min(axis=1)

threshold_95 = np.percentile(nn_dists_tr, 95)
outside_hull = nn_dists_te > threshold_95

div_rows = []
for i, drug in enumerate(df_test["Drug"].values):
    div_rows.append({
        "Drug":                     drug,
        "SMILES":                   df_test["SMILES"].iloc[i],
        "NN_dist_to_train":         float(nn_dists_te[i]),
        "PC1":                      float(X_te_pca[i, 0]),
        "PC2":                      float(X_te_pca[i, 1]),
        "outside_train_hull_95pct": bool(outside_hull[i])
    })

div_df = pd.DataFrame(div_rows)
div_df.to_csv("chemical_space_diversity.csv", index=False)
print("Saved -> chemical_space_diversity.csv")

print(f"\nTest-set applicability domain summary:")
print(f"  Mean NN dist  (test→train)   : {nn_dists_te.mean():.4f}")
print(f"  Median NN dist               : {np.median(nn_dists_te):.4f}")
print(f"  AD threshold (train p95)     : {threshold_95:.4f}")
print(f"  Outside AD (95th pct)        : "
      f"{outside_hull.mean():.2%}  ({outside_hull.sum()}/{len(outside_hull)})")
print(f"  Intra-train mean NN dist     : {nn_dists_tr.mean():.4f}  (reference)")

if X_tr_sc.shape[0] <= 500:
    d_cross   = cdist(X_te_sc, X_tr_sc).mean()
    d_intr_te = cdist(X_te_sc, X_te_sc).mean()
    print(f"\nPairwise distances (normalised feature space):")
    print(f"  Intra-train mean : {d_intra[d_intra < np.inf].mean():.4f}")
    print(f"  Intra-test  mean : {d_intr_te:.4f}")
    print(f"  Cross mean       : {d_cross:.4f}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X_tr_pca[:, 0], X_tr_pca[:, 1],
               alpha=0.5, s=30, label="Train",
               color="#0856A5")
    ax.scatter(X_te_pca[~outside_hull, 0], X_te_pca[~outside_hull, 1],
               alpha=0.85, s=55, marker="^", label="Test (inside AD)",
               color="#E8624C", edgecolors="black", linewidths=0.4)
    ax.scatter(X_te_pca[outside_hull, 0], X_te_pca[outside_hull, 1],
               alpha=0.85, s=55, marker="X", label="Test (outside AD)",
               color="#F5A623", edgecolors="black", linewidths=0.4)
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
    ax.set_title("Chemical Space – Topological Descriptor PCA\n"
                )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("chemical_space_pca.png", dpi=600)
    plt.close()
    print("Saved -> chemical_space_pca.png")
except ImportError:
    print("matplotlib not available – skipping PCA plot.")
print("\n" + "=" * 62)
print("  SECTION 4 :  FINAL MODEL TRAINING & TEST EVALUATION")
print("=" * 62)

yhat_te = yhat_real

rows, pred_long = [], []

for j, prop in enumerate(prop_cols):
    m    = Mte_t[:, j].numpy().astype(bool)
    te_n = int(m.sum())
    tr_n = int(Mtr_t[:, j].sum().item())

    if te_n < 3 or tr_n < 10:
        rows.append({
            "Property": prop, "Model": "MAG-Net",
            "R2": np.nan, "MAE": np.nan, "RMSE": np.nan,
            "pearson_r": np.nan, "pearson_p": np.nan,
            "spearman_rho": np.nan, "spearman_p": np.nan,
            "Train_N": tr_n, "Test_N": te_n,
            "Note": "Insufficient data"
        })
        continue

    yt = df_test.loc[m, prop].values
    yp = yhat_te[m, j]

    rmse           = rmse_fn(yt, yp)
    pr, pp, sr, sp = safe_corr(yt, yp)

    rows.append({
        "Property":     prop,
        "Model":        "MAG-Net",
        "R2":           float(r2_score(yt, yp)),
        "MAE":          float(mean_absolute_error(yt, yp)),
        "RMSE":         float(rmse),
        "pearson_r":    pr,   "pearson_p":    pp,
        "spearman_rho": sr,   "spearman_p":   sp,
        "Train_N":      tr_n, "Test_N":       te_n,
        "Note":         ""
    })

    for d, s, yt_i, yp_i in zip(df_test.loc[m, "Drug"].values,
                                  df_test.loc[m, "SMILES"].values,
                                  yt, yp):
        pred_long.append({
            "Drug": d, "SMILES": s,
            "Property": prop, "Model": "MAG-Net",
            "y_true": float(yt_i), "y_pred": float(yp_i)
        })

metrics = pd.DataFrame(rows).sort_values("Property")
metrics.to_csv("proposed_model_metrics.csv", index=False)
print("Saved -> proposed_model_metrics.csv")

pred_long_df = pd.DataFrame(pred_long)
pred_long_df.to_csv("proposed_model_test_predictions_long.csv", index=False)
print("Saved -> proposed_model_test_predictions_long.csv")

if not pred_long_df.empty:
    wide_pred = pred_long_df.pivot_table(
        index=["Drug", "SMILES"], columns="Property",
        values="y_pred", aggfunc="first").reset_index()
    wide_true = pred_long_df.pivot_table(
        index=["Drug", "SMILES"], columns="Property",
        values="y_true", aggfunc="first").reset_index()
    wide_true = wide_true.rename(
        columns={c: f"{c}__y_true"
                 for c in wide_true.columns
                 if c not in ["Drug", "SMILES"]})
    (wide_pred
     .merge(wide_true, on=["Drug", "SMILES"], how="left")
     .to_csv("proposed_model_test_predictions_wide.csv", index=False))
    print("Saved -> proposed_model_test_predictions_wide.csv")

    corr_rows = []
    for (prop, model_name), g in pred_long_df.groupby(["Property", "Model"]):
        x    = np.asarray(g["y_true"], float)
        y    = np.asarray(g["y_pred"], float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size >= 3:
            pr, pp = pearsonr(x, y)
            sr, sp = spearmanr(x, y)
            corr_rows.append({
                "Property":     prop,
                "Model":        model_name,
                "pearson_r":    float(pr),
                "pearson_p":    float(pp),
                "spearman_rho": float(sr),
                "spearman_p":   float(sp),
                "N_test":       int(x.size)
            })
    pd.DataFrame(corr_rows).to_csv(
        "proposed_model_pred_vs_true_correlations.csv", index=False)
    print("Saved -> proposed_model_pred_vs_true_correlations.csv")

