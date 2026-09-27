
import copy
import math
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
# SEARCH_EPOCHS reduced 150 -> 100 vs. the task's default for time-budget
# reasons; see module docstring "DEVIATION" note for the timed-pilot numbers
# that justified this.
SEARCH_EPOCHS, SEARCH_PATIENCE = 100, 20
FINAL_EPOCHS, FINAL_PATIENCE = 500, 60

# Valid (d_model, num_heads) pairs -- d_model divisible by num_heads is
# enforced by only ever sampling from this fixed list (see docstring).
DMODEL_NHEAD_PAIRS = [
    (16, 1), (16, 2), (16, 4),
    (32, 1), (32, 2), (32, 4),
    (64, 1), (64, 2), (64, 4),
]

class SinusoidalPositionalEncoding(nn.Module):


    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:, : pe[:, 1::2].shape[1]]
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, L, D]
        return x + self.pe[:, : x.size(1), :]

class TransformerSMILES(nn.Module):
    """Character-level Transformer encoder over SMILES token ids.

    Embedding -> sinusoidal positional encoding -> TransformerEncoder
    (with padding mask) -> masked mean-pool over non-PAD positions -> MLP
    head -> n_out predictions.
    """

    def __init__(self, vocab_size, max_len, d_model=32, num_heads=2,
                 num_encoder_layers=2, dim_feedforward=64, p_drop=0.2, n_out=7):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.in_drop = nn.Dropout(p_drop)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=p_drop, activation="relu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(dim_feedforward, n_out),
        )

    def forward(self, x):
        # x: [B, L] long token ids
        pad_mask = (x == PAD_ID)                       # [B, L] True where PAD
        h = self.embed(x)                               # [B, L, D]
        h = self.pos_enc(h)
        h = self.in_drop(h)
        h = self.encoder(h, src_key_padding_mask=pad_mask)  # [B, L, D]

        valid = (~pad_mask).unsqueeze(-1).float()        # [B, L, 1]
        summed = (h * valid).sum(dim=1)                  # [B, D]
        counts = valid.sum(dim=1).clamp(min=1.0)          # [B, 1]
        pooled = summed / counts                          # masked mean pool
        return self.head(pooled)                          # [B, n_out]

def sample_config(rng):
    pair_idx = int(rng.randint(0, len(DMODEL_NHEAD_PAIRS)))
    d_model, num_heads = DMODEL_NHEAD_PAIRS[pair_idx]
    return {
        "d_model":            d_model,
        "num_heads":          num_heads,
        "num_encoder_layers": int(rng.choice([1, 2, 3])),
        "dim_feedforward":    int(rng.choice([32, 64, 128])),
        "p_drop":             float(rng.choice([0.1, 0.2, 0.3])),
        "lr":                 float(rng.choice([5e-4, 1e-3, 2e-3, 3e-3])),
        "weight_decay":       float(rng.choice([1e-5, 1e-4, 1e-3])),
        "batch":              int(rng.choice([8, 16, 32])),
    }


def train_model(Xtr, Ytr_z, Mtr, Xval, Yval_raw, Mval, y_mu, y_sd, vocab_size, max_len, dout,
                 epochs, patience, d_model, num_heads, num_encoder_layers, dim_feedforward,
                 p_drop, lr, weight_decay, batch, seed=SEED):
    Mirrors baseline_harness.train_generic / optimize_smiles_cnn.train_model's
    optimisation loop (Adam, ReduceLROnPlateau factor=0.7/patience=10/min_lr=5e-5,
    grad-clip 2.0, masked-MSE loss, best-val-RMSE early stopping).
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    model = TransformerSMILES(
        vocab_size, max_len, d_model=d_model, num_heads=num_heads,
        num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
        p_drop=p_drop, n_out=dout,
    )
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
    # reused across all 5 CV folds and the final refit (matches
    # baseline_smiles_cnn.py / optimize_smiles_cnn.py exactly).
    vocab = build_vocab(train_smiles)
    vocab_size = len(vocab)
    max_len = max(len(s) for s in train_smiles)
    print(f"Vocab size (incl. PAD/UNK): {vocab_size} | Train max SMILES length: {max_len}")

    X_full = torch.tensor([tokenize(s, vocab, max_len) for s in train_smiles], dtype=torch.long)
    Y_full_np = df_train[H.PROP_COLS].values.astype(np.float32)
    Xte_all = torch.tensor([tokenize(s, vocab, max_len) for s in test_smiles], dtype=torch.long)
    Yte_np = df_test[H.PROP_COLS].values.astype(np.float32)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    print(f"\nSearching {N_CONFIGS} Transformer-SMILES configs via 5-fold CV "
          f"(SEARCH_EPOCHS={SEARCH_EPOCHS}, SEARCH_PATIENCE={SEARCH_PATIENCE})...")
    rng = np.random.RandomState(SEED)
    search_results = []
    failed_configs = []
    for cfg_i in range(N_CONFIGS):
        cfg = sample_config(rng)
        fold_rmses = []
        fold_error = None
        try:
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
                    vocab_size=vocab_size, max_len=max_len, dout=len(H.PROP_COLS),
                    epochs=SEARCH_EPOCHS, patience=SEARCH_PATIENCE, **cfg)
                if not np.isfinite(val_rmse):
                    raise RuntimeError(f"non-finite val_rmse={val_rmse}")
                fold_rmses.append(val_rmse)
        except Exception as e:
            fold_error = str(e)

        elapsed = (time.time() - t_start) / 60.0
        if fold_error is not None or len(fold_rmses) < 5:
            failed_configs.append({"cfg_i": cfg_i + 1, "cfg": cfg, "error": fold_error})
            print(f"  config {cfg_i + 1:02d}/{N_CONFIGS}: {cfg} -> FAILED ({fold_error}) "
                  f"[elapsed {elapsed:.1f} min]", flush=True)
            continue

        mean_rmse = float(np.mean(fold_rmses))
        search_results.append((mean_rmse, cfg))
        print(f"  config {cfg_i + 1:02d}/{N_CONFIGS}: {cfg} -> mean CV RMSE = {mean_rmse:.4f}"
              f"  [elapsed {elapsed:.1f} min]", flush=True)

    if failed_configs:
        print(f"\nWARNING: {len(failed_configs)} config(s) FAILED and were excluded from "
              f"selection (not silently dropped -- logged here and in search log with NaN "
              f"mean_cv_rmse): {failed_configs}")

    if not search_results:
        raise RuntimeError("All configs failed -- cannot select a winner.")

    search_results.sort(key=lambda t: t[0])
    best_rmse, best_cfg = search_results[0]
    worst_rmse, worst_cfg = search_results[-1]
    print(f"\n[CV] Best Transformer-SMILES config: {best_cfg} | mean CV RMSE = {best_rmse:.4f}")
    print(f"[CV] Worst Transformer-SMILES config: {worst_cfg} | mean CV RMSE = {worst_rmse:.4f}")

    log_rows = [{"mean_cv_rmse": r, **c} for r, c in search_results]
    log_rows += [{"mean_cv_rmse": np.nan, **fc["cfg"], "error": fc["error"]} for fc in failed_configs]
    pd.DataFrame(log_rows).to_csv(H._p("transformer_smiles_optimized_search_log.csv"), index=False)
    pd.DataFrame([{"mean_cv_rmse": best_rmse, **best_cfg}]).to_csv(
        H._p("transformer_smiles_optimized_best_config.csv"), index=False)

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
        vocab_size=vocab_size, max_len=max_len, dout=len(H.PROP_COLS),
        epochs=FINAL_EPOCHS, patience=FINAL_PATIENCE, **best_cfg)
    print(f"Final refit best internal val RMSE: {final_val_rmse:.5f}")

    yhat_test = predict(final_model, Xte_all, y_mu, y_sd)
    if not np.isfinite(yhat_test).all():
        print("WARNING: non-finite predictions detected in test set!")

    # Sanity: confirm test-set drug identity matches cv_test_split.csv exactly.
    split_drugs = pd.read_csv(H._p("cv_test_split.csv"))["Drug"].astype(str).str.strip().tolist()
    assert list(df_test["Drug"]) == split_drugs or set(df_test["Drug"]) == set(split_drugs), \
        "Test-set drug identity mismatch vs cv_test_split.csv!"

    metrics = H.evaluate_and_save(
        "Transformer-SMILES (optimized)", df_test, Yte_np, yhat_test, Mte_all.numpy(),
        out_prefix="transformer_smiles_optimized",
    )

    print("\n=== Transformer-SMILES (optimized) -- canonical held-out test (22 drugs) ===")
    print(metrics.round(4).to_string(index=False))

    n_params = sum(p.numel() for p in final_model.parameters())
    print(f"\nTransformer-SMILES (optimized) trainable params: {n_params}")
    print(f"Winning config: {best_cfg}")

    n_pos_r2 = int((metrics["R2"] > 0).sum())
    n_neg_r2 = int((metrics["R2"] <= 0).sum())
    print(f"\nProperties with POSITIVE R2: {n_pos_r2}/7   Non-positive (<=0) R2: {n_neg_r2}/7")

    total_min = (time.time() - t_start) / 60.0
    print(f"\n[optimize_transformer_smiles] TOTAL wall-clock time: {total_min:.1f} minutes "
          f"(N_CONFIGS={N_CONFIGS}, SEARCH_EPOCHS={SEARCH_EPOCHS}, "
          f"SEARCH_PATIENCE={SEARCH_PATIENCE}, FINAL_EPOCHS={FINAL_EPOCHS}, "
          f"FINAL_PATIENCE={FINAL_PATIENCE})")
    print("[optimize_transformer_smiles] DONE.")


if __name__ == "__main__":
    main()
