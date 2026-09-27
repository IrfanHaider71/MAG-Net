
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr

import baseline_harness as H
import repeated_split_harness as R

R.SEED = 42
H.set_seed(42)

N_CONFIGS = 25
SEARCH_EPOCHS, SEARCH_PATIENCE = 200, 25
FINAL_EPOCHS = 500

df = H.load_merged()
df_train, df_test = H.canonical_split(df)  # exact same 88/22 split as everything else in the paper

X_full = df_train[H.EXACT_INDEX_COLS].values.astype(np.float32)
Y_full = df_train[H.PROP_COLS].values.astype(np.float32)
Xte_np = df_test[H.EXACT_INDEX_COLS].values.astype(np.float32)
Yte_np = df_test[H.PROP_COLS].values.astype(np.float32)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"Searching {N_CONFIGS} DescriptorMLP configs via 5-fold CV "
      f"(same search space/protocol as magnet_joint_tuning.py)...")
rng = np.random.RandomState(42)
search_results = []
for cfg_i in range(N_CONFIGS):
    cfg = R.sample_config(rng)
    fold_rmses = []
    for tr_i, va_i in kf.split(X_full):
        Xtr_t = torch.from_numpy(X_full[tr_i]); Ytr_t = torch.from_numpy(Y_full[tr_i])
        Mtr_t = torch.isfinite(Ytr_t).float(); Ytr_t[~torch.isfinite(Ytr_t)] = 0.0
        Xva_t = torch.from_numpy(X_full[va_i]); Yva_t = torch.from_numpy(Y_full[va_i])
        Mva_t = torch.isfinite(Yva_t).float(); Yva_t[~torch.isfinite(Yva_t)] = 0.0

        Xtr_z, Xva_z, mu_x, sd_x = R.standardize(Xtr_t, Xva_t)
        y_mu_f, y_sd_f = R.mask_aware_target_stats(Ytr_t, Mtr_t)
        Ytr_z = (Ytr_t - y_mu_f) / y_sd_f

        _, val_rmse = R.train_model(
            R.DescriptorMLP, Xtr_z, Ytr_z, Mtr_t, Xva_z, Yva_t, Mva_t, y_mu_f, y_sd_f,
            din=len(H.EXACT_INDEX_COLS), dout=len(H.PROP_COLS),
            epochs=SEARCH_EPOCHS, patience=SEARCH_PATIENCE, **cfg)
        fold_rmses.append(val_rmse)
    mean_rmse = float(np.mean(fold_rmses))
    search_results.append((mean_rmse, cfg))
    print(f"  config {cfg_i+1:02d}/{N_CONFIGS}: {cfg} -> mean CV RMSE = {mean_rmse:.4f}")

search_results.sort(key=lambda t: t[0])
best_rmse, best_cfg = search_results[0]
print(f"\n[CV] Best DescriptorMLP config: {best_cfg} | mean CV RMSE = {best_rmse:.4f}")

pd.DataFrame([{"mean_cv_rmse": r, **c} for r, c in search_results]).to_csv(
    "descriptor_mlp_joint_tuning_search_log.csv", index=False)
pd.DataFrame([{"mean_cv_rmse": best_rmse, **best_cfg}]).to_csv(
    "descriptor_mlp_best_config.csv", index=False)

# ---- final refit: identical 85/15 internal split protocol to magnet_joint_tuning.py ----
Xtr_t = torch.from_numpy(X_full); Ytr_t = torch.from_numpy(Y_full)
Mtr_t = torch.isfinite(Ytr_t).float(); Ytr_t[~torch.isfinite(Ytr_t)] = 0.0
Xte_t = torch.from_numpy(Xte_np); Yte_t = torch.from_numpy(Yte_np)
Mte_t = torch.isfinite(Yte_t).float(); Yte_t[~torch.isfinite(Yte_t)] = 0.0

Xtr_z, Xte_z, x_mu, x_sd = R.standardize(Xtr_t, Xte_t)
y_mu, y_sd = R.mask_aware_target_stats(Ytr_t, Mtr_t)
Ytr_z = (Ytr_t - y_mu) / y_sd

n_all = Xtr_z.shape[0]
perm_all = np.random.RandomState(42).permutation(n_all)
val_n = max(1, int(0.15 * n_all))
val_idx2 = perm_all[:val_n]; tr2_idx = perm_all[val_n:]
X_tr2, Y_tr2, M_tr2 = Xtr_z[tr2_idx], Ytr_z[tr2_idx], Mtr_t[tr2_idx]
X_val2, Y_val2, M_val2 = Xtr_z[val_idx2], Ytr_t[val_idx2], Mtr_t[val_idx2]

final_model, _ = R.train_model(
    R.DescriptorMLP, X_tr2, Y_tr2, M_tr2, X_val2, Y_val2, M_val2, y_mu, y_sd,
    din=len(H.EXACT_INDEX_COLS), dout=len(H.PROP_COLS),
    epochs=FINAL_EPOCHS, patience=60, **best_cfg)

yhat_test = R.predict(final_model, Xte_z, y_mu, y_sd)

rows = []
for j, prop in enumerate(H.PROP_COLS):
    m = Mte_t[:, j].numpy().astype(bool)
    yt = Yte_np[m, j]; yp = yhat_test[m, j]
    r2 = r2_score(yt, yp); mae = mean_absolute_error(yt, yp); rmse = H.rmse_fn(yt, yp)
    pr, _ = pearsonr(yt, yp); sr, _ = spearmanr(yt, yp)
    rows.append({"Property": prop, "Model": "Descriptor-MLP (fairly-tuned)",
                 "R2": r2, "MAE": mae, "RMSE": rmse, "pearson_r": pr, "spearman_rho": sr})

tuned_df = pd.DataFrame(rows)
tuned_df.to_csv("descriptor_mlp_joint_tuned_metrics.csv", index=False)

print("\n=== Fairly-tuned DescriptorMLP (held-out canonical test) ===")
print(tuned_df.round(4).to_string(index=False))

# param count for the record
n_params = sum(p.numel() for p in final_model.parameters())
print(f"\nDescriptorMLP (fairly-tuned) trainable params: {n_params}")
tgn_cfg = dict(hidden=384, n_layers=2)  # manuscript-reported MAG-Net config
tgn = R.MAGNet(din=9, dout=7, hidden=384, p_drop=0.15, n_layers=2)
n_tgn = sum(p.numel() for p in tgn.parameters())
print(f"MAG-Net (manuscript config hidden=384) trainable params: {n_tgn}")
print(f"Ratio: {n_tgn/n_params:.3f}x" if n_params else "N/A")
