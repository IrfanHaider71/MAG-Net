
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

import baseline_harness as H
from baseline_smiles_cnn import build_vocab, tokenize, PAD_ID

SEED = 42
H.set_seed(SEED)

N_CONFIGS = 25
SEARCH_EPOCHS, SEARCH_PATIENCE = 200, 25
FINAL_EPOCHS, FINAL_PATIENCE = 500, 60

CHANNEL_KERNEL_PRESETS = {
    2: ([64, 128], [5, 3]),
    3: ([64, 128, 128], [7, 5, 3]),
}


class SMILESConvNet(nn.Module):

    def __init__(self, vocab_size, embed_dim=32, channels=(64, 128, 128),
                 kernels=(7, 5, 3), hidden=128, p_drop=0.2, n_out=7):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)

        def conv_block(c_in, c_out, k):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2),
                nn.ReLU(),
                nn.Dropout(p_drop),
            )

        convs = []
        c_in = embed_dim
        for c_out, k in zip(channels, kernels):
            convs.append(conv_block(c_in, c_out, k))
            c_in = c_out
        self.convs = nn.Sequential(*convs)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(c_in, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        h = self.embed(x)                 # [B, L, E]
        h = h.transpose(1, 2)             # [B, E, L]
        h = self.convs(h)                 # [B, C_last, L]
        h = self.pool(h).squeeze(-1)      # [B, C_last]
        return self.head(h)               # [B, n_out]

def sample_config(rng):
    n_conv_layers = int(rng.choice([2, 3]))
    return {
        "embed_dim":      int(rng.choice([16, 32, 48, 64])),
        "n_conv_layers":  n_conv_layers,
        "hidden":         int(rng.choice([64, 128, 192, 256])),
        "p_drop":         float(rng.choice([0.1, 0.2, 0.3, 0.4])),
        "lr":             float(rng.choice([5e-4, 1e-3, 2e-3, 3e-3])),
        "weight_decay":   float(rng.choice([1e-5, 1e-4, 1e-3])),
        "batch":          int(rng.choice([8, 16, 32])),
    }


def train_model(Xtr, Ytr_z, Mtr, Xval, Yval_raw, Mval, y_mu, y_sd, vocab_size, dout,
                 epochs, patience, embed_dim, n_conv_layers, hidden, p_drop, lr,
                 weight_decay, batch, seed=SEED):

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    channels, kernels = CHANNEL_KERNEL_PRESETS[n_conv_layers]
    model = SMILESConvNet(vocab_size, embed_dim=embed_dim, channels=channels,
                           kernels=kernels, hidden=hidden, p_drop=p_drop, n_out=dout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.7, patience=10, min_lr=5e-5)

    best, best_state, bad = 1e30, None, 0
    n = Xtr.shape[0]
    for ep in range(1, epochs + 1):
        model.train()
        idx = rng.permutation(n)
        for i in range(0, n, batch):
            j = idx[i:i + batch]
            xb, yb, mb = Xtr[j], Ytr_z[j], Mtr[j]
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
                (((yv_hat - Yval_raw) ** 2 * Mval).sum()) / (Mval.sum() + 1e-8)
            ).item()
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
    t_start = time.time()

    df = H.load_merged()
    df_train, df_test = H.canonical_split(df)

    train_smiles = df_train["SMILES"].tolist()
    test_smiles = df_test["SMILES"].tolist()

    # Vocab + max_len built ONCE from the full 88-compound training set,
    # reused across all 5 CV folds and the final refit (see module docstring).
    vocab = build_vocab(train_smiles)
    vocab_size = len(vocab)
    max_len = max(len(s) for s in train_smiles)
    print(f"Vocab size (incl. PAD/UNK): {vocab_size} | Train max SMILES length: {max_len}")

    X_full = torch.tensor([tokenize(s, vocab, max_len) for s in train_smiles], dtype=torch.long)
    Y_full_np = df_train[H.PROP_COLS].values.astype(np.float32)
    Xte_all = torch.tensor([tokenize(s, vocab, max_len) for s in test_smiles], dtype=torch.long)
    Yte_np = df_test[H.PROP_COLS].values.astype(np.float32)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    print(f"\nSearching {N_CONFIGS} SMILES-CNN configs via 5-fold CV "
          f"(SEARCH_EPOCHS={SEARCH_EPOCHS}, SEARCH_PATIENCE={SEARCH_PATIENCE})...")
    rng = np.random.RandomState(SEED)
    search_results = []
    for cfg_i in range(N_CONFIGS):
        cfg = sample_config(rng)
        fold_rmses = []
        for tr_i, va_i in kf.split(X_full.numpy()):
            Xtr_t = X_full[tr_i]
            Ytr_t = torch.from_numpy(Y_full_np[tr_i])
            Mtr_t = torch.isfinite(Ytr_t).float()
            Ytr_t[~torch.isfinite(Ytr_t)] = 0.0

            Xva_t = X_full[va_i]
            Yva_t = torch.from_numpy(Y_full_np[va_i])
            Mva_t = torch.isfinite(Yva_t).float()
            Yva_t[~torch.isfinite(Yva_t)] = 0.0

            y_mu_f, y_sd_f = H.mask_aware_target_stats(Ytr_t, Mtr_t)
            Ytr_z = (Ytr_t - y_mu_f) / y_sd_f

            _, val_rmse = train_model(
                Xtr_t, Ytr_z, Mtr_t, Xva_t, Yva_t, Mva_t, y_mu_f, y_sd_f,
                vocab_size=vocab_size, dout=len(H.PROP_COLS),
                epochs=SEARCH_EPOCHS, patience=SEARCH_PATIENCE, **cfg)
            fold_rmses.append(val_rmse)

        mean_rmse = float(np.mean(fold_rmses))
        search_results.append((mean_rmse, cfg))
        elapsed = (time.time() - t_start) / 60.0
        print(f"  config {cfg_i + 1:02d}/{N_CONFIGS}: {cfg} -> mean CV RMSE = {mean_rmse:.4f}"
              f"  [elapsed {elapsed:.1f} min]", flush=True)

    search_results.sort(key=lambda t: t[0])
    best_rmse, best_cfg = search_results[0]
    worst_rmse, worst_cfg = search_results[-1]
    print(f"\n[CV] Best SMILES-CNN config: {best_cfg} | mean CV RMSE = {best_rmse:.4f}")
    print(f"[CV] Worst SMILES-CNN config: {worst_cfg} | mean CV RMSE = {worst_rmse:.4f}")

    pd.DataFrame([{"mean_cv_rmse": r, **c} for r, c in search_results]).to_csv(
        H._p("smiles_cnn_optimized_search_log.csv"), index=False)
    pd.DataFrame([{"mean_cv_rmse": best_rmse, **best_cfg}]).to_csv(
        H._p("smiles_cnn_optimized_best_config.csv"), index=False)

    # ---- final refit: full 88-compound train set, 85/15 internal split ----
    Ytr_all = torch.from_numpy(Y_full_np)
    Mtr_all = torch.isfinite(Ytr_all).float()
    Ytr_all[~torch.isfinite(Ytr_all)] = 0.0
    Yte_all = torch.from_numpy(Yte_np)
    Mte_all = torch.isfinite(Yte_all).float()
    Yte_all[~torch.isfinite(Yte_all)] = 0.0

    y_mu, y_sd = H.mask_aware_target_stats(Ytr_all, Mtr_all)
    Ytr_z = (Ytr_all - y_mu) / y_sd

    n_all = X_full.shape[0]
    perm_all = np.random.RandomState(SEED).permutation(n_all)
    val_n = max(1, int(0.15 * n_all))
    val_idx2 = perm_all[:val_n]
    tr2_idx = perm_all[val_n:]
    X_tr2, Y_tr2, M_tr2 = X_full[tr2_idx], Ytr_z[tr2_idx], Mtr_all[tr2_idx]
    X_val2, Y_val2, M_val2 = X_full[val_idx2], Ytr_all[val_idx2], Mtr_all[val_idx2]

    print(f"\nFinal refit on full 88-compound train set "
          f"(train={len(tr2_idx)}, val={len(val_idx2)}) "
          f"with winning config, FINAL_EPOCHS={FINAL_EPOCHS}, patience={FINAL_PATIENCE}...")
    final_model, final_val_rmse = train_model(
        X_tr2, Y_tr2, M_tr2, X_val2, Y_val2, M_val2, y_mu, y_sd,
        vocab_size=vocab_size, dout=len(H.PROP_COLS),
        epochs=FINAL_EPOCHS, patience=FINAL_PATIENCE, **best_cfg)
    print(f"Final refit best internal val RMSE: {final_val_rmse:.5f}")

    yhat_test = predict(final_model, Xte_all, y_mu, y_sd)
    if not np.isfinite(yhat_test).all():
        print("WARNING: non-finite predictions detected in test set!")

    metrics = H.evaluate_and_save(
        "SMILES-CNN (optimized)", df_test, Yte_np, yhat_test, Mte_all.numpy(),
        out_prefix="smiles_cnn_optimized",
    )

    print("\n=== SMILES-CNN (optimized) -- canonical held-out test ===")
    print(metrics.round(4).to_string(index=False))

    n_params = sum(p.numel() for p in final_model.parameters())
    print(f"\nSMILES-CNN (optimized) trainable params: {n_params}")

    # ---- comparison vs original zero-search baseline ----
    orig = pd.read_csv(H._p("baseline_smiles_cnn_metrics.csv"))[
        ["Property", "R2", "MAE", "RMSE"]
    ].rename(columns={"R2": "R2_orig", "MAE": "MAE_orig", "RMSE": "RMSE_orig"})
    comp = metrics[["Property", "R2", "MAE", "RMSE"]].rename(
        columns={"R2": "R2_opt", "MAE": "MAE_opt", "RMSE": "RMSE_opt"}
    ).merge(orig, on="Property")
    comp["dR2"] = comp["R2_opt"] - comp["R2_orig"]
    comp["dRMSE"] = comp["RMSE_opt"] - comp["RMSE_orig"]
    comp = comp[["Property", "R2_orig", "R2_opt", "dR2", "RMSE_orig", "RMSE_opt", "dRMSE"]]

    print("\n=== Optimized SMILES-CNN vs original zero-search SMILES-CNN ===")
    print(comp.round(4).to_string(index=False))
    n_improved = int((comp["dR2"] > 0.001).sum())
    n_worsened = int((comp["dR2"] < -0.001).sum())
    n_same = int((comp["dR2"].abs() <= 0.001).sum())
    print(f"\nImproved (R2): {n_improved}/7   Worsened: {n_worsened}/7   Same: {n_same}/7")

    total_min = (time.time() - t_start) / 60.0
    print(f"\nTotal wall-clock time: {total_min:.1f} min")


if __name__ == "__main__":
    main()
