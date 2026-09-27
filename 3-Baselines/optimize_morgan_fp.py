
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr

import baseline_harness as H

SEED = 42
H.set_seed(SEED)

FP_RADIUS = 2
FP_NBITS = 1024

N_CONFIGS = 25
SEARCH_EPOCHS, SEARCH_PATIENCE = 200, 25
FINAL_EPOCHS, FINAL_PATIENCE = 500, 60


def smiles_to_fp(drug, smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(
            f"RDKit failed to parse SMILES for drug '{drug}': '{smiles}'. "
            f"This indicates a bug in optimize_morgan_fp.py's SMILES handling, "
            f"since this SMILES has parsed successfully in earlier pipeline stages."
        )
    bitvec = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS)
    arr = np.zeros((FP_NBITS,), dtype=np.float32)
    for bit in bitvec.GetOnBits():
        arr[bit] = 1.0
    return arr

class FPMLp(nn.Module):

    def __init__(self, input_dim=FP_NBITS, hidden_dims=(256, 128), p_drop=0.3, n_targets=7):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers.append(nn.Linear(d, n_targets))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def cfg_hidden_dims(cfg):
    return (cfg["hidden1"],) if cfg["hidden2"] == 0 else (cfg["hidden1"], cfg["hidden2"])

def sample_config(rng):

    two_layer = bool(rng.choice([True, False], p=[0.7, 0.3]))
    hidden1 = int(rng.choice([128, 256, 384, 512]))
    hidden2 = int(rng.choice([64, 128, 256])) if two_layer else 0
    p_drop = float(rng.choice([0.1, 0.2, 0.3, 0.4, 0.5]))
    lr = float(rng.choice([5e-4, 1e-3, 2e-3, 3e-3]))
    weight_decay = float(rng.choice([1e-5, 1e-4, 1e-3, 5e-3, 1e-2]))
    batch = int(rng.choice([8, 16, 32]))
    return {
        "hidden1": hidden1, "hidden2": hidden2, "p_drop": p_drop,
        "lr": lr, "weight_decay": weight_decay, "batch": batch,
    }


# training loop (tensor-batched; same loss/opt/
# schedule/clip/early-stop regime as baseline_harness.train_generic, but
# batches directly over precomputed FP tensors instead of a Python
# collate_fn, since the fingerprint is a fixed-size dense array -- much
# faster for a 25-config x 5-fold search over 1024-dim input)
def batcher(X, Y, M, bs, rng):
    n = X.shape[0]
    idx = rng.permutation(n)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield X[j], Y[j], M[j]


def train_model(Xtr, Ytr_z, Mtr, Xval, Yval_raw, Mval, y_mu, y_sd,
                 hidden_dims, p_drop, lr, weight_decay, batch, epochs, patience, seed=SEED):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    model = FPMLp(input_dim=FP_NBITS, hidden_dims=hidden_dims, p_drop=p_drop,
                   n_targets=len(H.PROP_COLS))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.7, patience=10, min_lr=5e-5)

    best, bad = 1e30, 0
    best_state = copy.deepcopy(model.state_dict())  

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb, mb in batcher(Xtr, Ytr_z, Mtr, batch, rng):
            opt.zero_grad()
            yhat = model(xb)
            loss = ((yhat - yb) ** 2 * mb).sum() / (mb.sum() + 1e-8)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            yv_hat = model(Xval) * y_sd + y_mu
            val_rmse = torch.sqrt(
                (((yv_hat - Yval_raw) ** 2 * Mval).sum()) / (Mval.sum() + 1e-8)).item()

        if not np.isfinite(val_rmse):
            bad += 1
            if bad >= patience:
                break
            continue

        sched.step(val_rmse)
        if val_rmse + 1e-8 < best:
            best_state = copy.deepcopy(model.state_dict())
            best = val_rmse
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return model, best


def predict(model, X, y_mu, y_sd):
    model.eval()
    with torch.no_grad():
        return (model(X) * y_sd + y_mu).numpy()


def main():
    t0 = time.time()
    df = H.load_merged()
    df_train, df_test = H.canonical_split(df)

    n_parsed = 0
    fps_by_drug = {}
    for _, row in df.iterrows():
        fp = smiles_to_fp(row["Drug"], row["SMILES"])
        fps_by_drug[row["Drug"]] = fp
        n_parsed += 1
    print(f"[optimize_morgan_fp] Successfully parsed SMILES for {n_parsed} / {len(df)} compounds.")
    assert n_parsed == 110, f"Expected 110 parsed SMILES, got {n_parsed}"

    X_full = np.stack([fps_by_drug[d] for d in df_train["Drug"].values]).astype(np.float32)
    Y_full = df_train[H.PROP_COLS].values.astype(np.float32)
    Xte_np = np.stack([fps_by_drug[d] for d in df_test["Drug"].values]).astype(np.float32)
    Yte_np = df_test[H.PROP_COLS].values.astype(np.float32)
    assert X_full.shape == (88, FP_NBITS) and Xte_np.shape == (22, FP_NBITS)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    print(f"Searching {N_CONFIGS} Morgan-FP-MLP configs via 5-fold CV "
          f"(SEARCH_EPOCHS={SEARCH_EPOCHS}, SEARCH_PATIENCE={SEARCH_PATIENCE})...")
    rng = np.random.RandomState(SEED)
    search_results = []
    for cfg_i in range(N_CONFIGS):
        cfg = sample_config(rng)
        hidden_dims = cfg_hidden_dims(cfg)
        fold_rmses = []
        for tr_i, va_i in kf.split(X_full):
            Xtr_t = torch.from_numpy(X_full[tr_i]); Ytr_t = torch.from_numpy(Y_full[tr_i])
            Mtr_t = torch.isfinite(Ytr_t).float(); Ytr_t[~torch.isfinite(Ytr_t)] = 0.0
            Xva_t = torch.from_numpy(X_full[va_i]); Yva_t = torch.from_numpy(Y_full[va_i])
            Mva_t = torch.isfinite(Yva_t).float(); Yva_t[~torch.isfinite(Yva_t)] = 0.0

            y_mu_f, y_sd_f = H.mask_aware_target_stats(Ytr_t, Mtr_t)
            Ytr_z = (Ytr_t - y_mu_f) / y_sd_f

            _, val_rmse = train_model(
                Xtr_t, Ytr_z, Mtr_t, Xva_t, Yva_t, Mva_t, y_mu_f, y_sd_f,
                hidden_dims=hidden_dims, p_drop=cfg["p_drop"], lr=cfg["lr"],
                weight_decay=cfg["weight_decay"], batch=cfg["batch"],
                epochs=SEARCH_EPOCHS, patience=SEARCH_PATIENCE, seed=SEED)
            fold_rmses.append(val_rmse)
        mean_rmse = float(np.mean(fold_rmses))
        search_results.append((mean_rmse, cfg))
        print(f"  config {cfg_i + 1:02d}/{N_CONFIGS}: hidden={hidden_dims} p_drop={cfg['p_drop']} "
              f"lr={cfg['lr']} wd={cfg['weight_decay']} batch={cfg['batch']} "
              f"-> mean CV RMSE = {mean_rmse:.4f}  [{time.time() - t0:.0f}s elapsed]")

    search_results.sort(key=lambda t: t[0])
    best_rmse, best_cfg = search_results[0]
    print(f"\n[CV] Best Morgan-FP-MLP config: {best_cfg} | mean CV RMSE = {best_rmse:.4f}")

    search_log_df = pd.DataFrame([{"mean_cv_rmse": r, **c} for r, c in search_results])
    search_log_df.to_csv("morgan_fp_optimized_search_log.csv", index=False)
    pd.DataFrame([{"mean_cv_rmse": best_rmse, **best_cfg}]).to_csv(
        "morgan_fp_optimized_best_config.csv", index=False)

    # ---- final refit: 85/15 internal train/val carve-out, same pattern as
    # optimize_magnet.py / fair_tune_descriptor_mlp.py ----
    Xtr_full_t = torch.from_numpy(X_full); Ytr_full_t = torch.from_numpy(Y_full)
    Mtr_full_t = torch.isfinite(Ytr_full_t).float(); Ytr_full_t[~torch.isfinite(Ytr_full_t)] = 0.0
    Xte_t = torch.from_numpy(Xte_np); Yte_t = torch.from_numpy(Yte_np)
    Mte_t = torch.isfinite(Yte_t).float(); Yte_t[~torch.isfinite(Yte_t)] = 0.0

    y_mu, y_sd = H.mask_aware_target_stats(Ytr_full_t, Mtr_full_t)
    Ytr_full_z = (Ytr_full_t - y_mu) / y_sd

    n_all = Xtr_full_t.shape[0]
    perm_all = np.random.RandomState(SEED).permutation(n_all)
    val_n = max(1, int(0.15 * n_all))
    val_idx2 = perm_all[:val_n]; tr2_idx = perm_all[val_n:]
    X_tr2, Y_tr2, M_tr2 = Xtr_full_t[tr2_idx], Ytr_full_z[tr2_idx], Mtr_full_t[tr2_idx]
    X_val2, Y_val2, M_val2 = Xtr_full_t[val_idx2], Ytr_full_t[val_idx2], Mtr_full_t[val_idx2]

    best_hidden_dims = cfg_hidden_dims(best_cfg)
    final_model, final_val_rmse = train_model(
        X_tr2, Y_tr2, M_tr2, X_val2, Y_val2, M_val2, y_mu, y_sd,
        hidden_dims=best_hidden_dims, p_drop=best_cfg["p_drop"], lr=best_cfg["lr"],
        weight_decay=best_cfg["weight_decay"], batch=best_cfg["batch"],
        epochs=FINAL_EPOCHS, patience=FINAL_PATIENCE, seed=SEED)
    print(f"\n[final refit] best val RMSE (raw scale) = {final_val_rmse:.4f}")

    yhat_test = predict(final_model, Xte_t, y_mu, y_sd)

    rows = []
    for j, prop in enumerate(H.PROP_COLS):
        m = Mte_t[:, j].numpy().astype(bool)
        yt = Yte_np[m, j]; yp = yhat_test[m, j]
        r2 = float(r2_score(yt, yp)); mae = float(mean_absolute_error(yt, yp)); rmse = H.rmse_fn(yt, yp)
        pr, _ = pearsonr(yt, yp); sr, _ = spearmanr(yt, yp)
        rows.append({"Property": prop, "Model": "Morgan-FP-MLP (optimized)",
                     "R2": r2, "MAE": mae, "RMSE": rmse, "pearson_r": float(pr), "spearman_rho": float(sr)})

    tuned_df = pd.DataFrame(rows).sort_values("Property").reset_index(drop=True)
    tuned_df.to_csv("morgan_fp_optimized_metrics.csv", index=False)

    print("\n=== Morgan-FP-MLP (optimized) -- canonical held-out test ===")
    print(tuned_df.round(4).to_string(index=False))

    n_params = sum(p.numel() for p in final_model.parameters())
    print(f"\nOptimized Morgan-FP-MLP trainable params: {n_params} | hidden_dims={best_hidden_dims}")
    print(f"Wall-clock: {time.time() - t0:.0f}s")


    print("\n" + "=" * 70)
    print("SELF-VERIFICATION")
    print("=" * 70)

    ok = True

    # 1. search log has exactly 25 rows, all finite CV RMSE
    log_check = pd.read_csv("morgan_fp_optimized_search_log.csv")
    cond1 = len(log_check) == N_CONFIGS and np.isfinite(log_check["mean_cv_rmse"]).all()
    print(f"[1] search log has {len(log_check)} rows (expect {N_CONFIGS}), "
          f"all finite mean_cv_rmse: {np.isfinite(log_check['mean_cv_rmse']).all()} -> "
          f"{'PASS' if cond1 else 'FAIL'}")
    ok &= cond1

    # 2. final refit's test-set drug identity matches cv_test_split.csv exactly
    split_ref = pd.read_csv("cv_test_split.csv")
    split_ref["Drug"] = split_ref["Drug"].astype(str).str.strip()
    cond2 = set(df_test["Drug"]) == set(split_ref["Drug"]) and len(df_test) == 22
    print(f"[2] test-set drug identity matches cv_test_split.csv (n={len(df_test)}): "
          f"{'PASS' if cond2 else 'FAIL'}")
    ok &= cond2

    cond3 = (n_parsed == 110) and (len(fps_by_drug) == 110)
    print(f"[3] all 110 SMILES parsed for fingerprinting: {'PASS' if cond3 else 'FAIL'}")
    ok &= cond3

    # 4. independently recompute R2/MAE/RMSE from raw predictions
    max_abs_diff = 0.0
    for j, prop in enumerate(H.PROP_COLS):
        m = Mte_t[:, j].numpy().astype(bool)
        yt = Yte_np[m, j]; yp = yhat_test[m, j]
        r2_manual = 1.0 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)
        mae_manual = np.mean(np.abs(yt - yp))
        rmse_manual = np.sqrt(np.mean((yt - yp) ** 2))
        saved_row = tuned_df[tuned_df["Property"] == prop].iloc[0]
        d = max(abs(r2_manual - saved_row["R2"]), abs(mae_manual - saved_row["MAE"]),
                abs(rmse_manual - saved_row["RMSE"]))
        max_abs_diff = max(max_abs_diff, d)
    cond4 = max_abs_diff < 1e-4  # float32 accumulation tolerance
    print(f"[4] independent recompute of R2/MAE/RMSE from raw predictions matches saved CSV "
          f"(max abs diff = {max_abs_diff:.2e}): {'PASS' if cond4 else 'FAIL'}")
    ok &= cond4

    print(f"\nOVERALL SELF-VERIFICATION: {'ALL PASS' if ok else 'FAILURES DETECTED -- SEE ABOVE'}")
    print("\n" + "=" * 70)
    print("COMPARISON: optimized vs original (zero-search) Morgan-FP-MLP")
    print("=" * 70)
    orig = pd.read_csv("baseline_morgan_fp_metrics.csv")
    orig_small = orig[["Property", "R2", "MAE", "RMSE"]].rename(
        columns={"R2": "R2_orig", "MAE": "MAE_orig", "RMSE": "RMSE_orig"})
    comp = tuned_df.merge(orig_small, on="Property")
    comp["dR2"] = comp["R2"] - comp["R2_orig"]
    comp["dRMSE"] = comp["RMSE"] - comp["RMSE_orig"]
    print(comp[["Property", "R2_orig", "R2", "dR2", "RMSE_orig", "RMSE", "dRMSE"]]
          .round(4).to_string(index=False))
    n_improved = int((comp["dR2"] > 0.001).sum())
    n_worse = int((comp["dR2"] < -0.001).sum())
    n_same = 7 - n_improved - n_worse
    print(f"\nR2 improved: {n_improved}/7   worsened: {n_worse}/7   ~same: {n_same}/7")
    print(f"Mean R2: original={comp['R2_orig'].mean():.4f}  optimized={comp['R2'].mean():.4f}")
    print(f"Mean RMSE: original={comp['RMSE_orig'].mean():.4f}  optimized={comp['RMSE'].mean():.4f}")

    print("\n" + "=" * 70)
    print("WEIGHT_DECAY FINDING (original unoptimized baseline used weight_decay=5e-3)")
    print("=" * 70)
    print(f"Winning config weight_decay = {best_cfg['weight_decay']:g}")
    wd_group = log_check.groupby("weight_decay")["mean_cv_rmse"].agg(["mean", "min", "count"])
    print("\nMean/min/n CV RMSE by weight_decay value across all 25 configs tried:")
    print(wd_group.round(4).to_string())
    if best_cfg["weight_decay"] == 5e-3:
        print("\n-> The search independently RE-SELECTED weight_decay=5e-3: the original "
              "value appears reasonable/near-optimal for this architecture/input.")
    else:
        print(f"\n-> The search MOVED AWAY from weight_decay=5e-3 to "
              f"{best_cfg['weight_decay']:g}. This suggests the original hand-picked value "
              f"was NOT independently justified by a search -- consistent with the prior "
              f"audit's flag that it was an unjustified/unexamined outlier.")


if __name__ == "__main__":
    main()
