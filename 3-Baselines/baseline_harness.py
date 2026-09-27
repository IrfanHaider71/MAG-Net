
import os, random, warnings, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

SEED = 42
DEVICE = torch.device("cpu")


def set_seed(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)

PROP_COLS = [
    "(MW)", "(HAC)", "(EM)",
    "(C)", " (P)", " (MV)", "(MR)"
]

INDEX_SHORT_NAMES = ["M1", "M2", "mM2", "ReM3", "H", "F", "AZ", "ISI", "SDD"]


_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "01_data")



def load_merged():
    """Load and merge SMILES + discripter + properties, exactly as
    classical_baselines.py / proposed_validated_pipeline.py do.""
    for f in [F_SMILES, F_EXACT, F_PROPS, F_SPLIT]:
        if not os.path.exists(_p(f)):
            raise FileNotFoundError(f"Missing required file: {f}")

    smiles = pd.read_csv(_p(F_SMILES))
    exact = pd.read_csv(_p(F_EXACT))
    props = pd.read_csv(_p(F_PROPS))

    for d in (smiles, exact, props):
        d.columns = d.columns.astype(str).str.strip()
        d["Drug"] = d["Drug"].astype(str).str.strip()
    smiles["SMILES"] = smiles["SMILES"].astype(str).str.strip()

    df = (smiles
          .merge(exact, on="Drug", how="inner")
          .merge(props, on="Drug", how="inner"))
    df = df.drop_duplicates(subset=["Drug", "SMILES"]).reset_index(drop=True)

    missing_prop = [c for c in PROP_COLS if c not in df.columns]
    if missing_prop:
        raise ValueError(f"Missing property columns: {missing_prop}")
    missing_idx = [c for c in EXACT_INDEX_COLS if c not in df.columns]
 
    assert len(df) == 110, f"Expected 110 merged compounds, got {len(df)}"
    assert df[PROP_COLS].isna().sum().sum() == 0, "Unexpected NaNs in property columns"
    assert df[EXACT_INDEX_COLS].isna().sum().sum() == 0, "Unexpected NaNs in index columns"
    return df


def canonical_split(df):
    """Split df into (df_train, df_test) using the EXACT held-out set already
    fixed by the existing pipeline in cv_test_split.csv (22 drugs,
    random_state=42, test_size=0.2). Raises if anything doesn't line up."""
    split = pd.read_csv(_p(F_SPLIT))
    split["Drug"] = split["Drug"].astype(str).str.strip()
    test_drugs = set(split["Drug"])

    missing = test_drugs - set(df["Drug"])
    if missing:
        raise RuntimeError(f"Test-split drugs missing from merged df: {missing}")

    df_test = df[df["Drug"].isin(test_drugs)].reset_index(drop=True)
    df_train = df[~df["Drug"].isin(test_drugs)].reset_index(drop=True)

    assert len(df_test) == len(test_drugs) == 22, \
        f"Expected 22 test rows matching cv_test_split.csv, got {len(df_test)}"
    assert len(df_train) == len(df) - len(df_test) == 88
    assert not (set(df_train["Drug"]) & set(df_test["Drug"])), "Drug overlap!"
    assert not (set(df_train["SMILES"]) & set(df_test["SMILES"])), "SMILES overlap!"
    return df_train, df_test

def mask_aware_target_stats(Ytr, Mtr):
    y_mu = Ytr.sum(0) / (Mtr.sum(0) + 1e-8)
    y_sd = torch.sqrt((((Ytr - y_mu) ** 2 * Mtr).sum(0) / (Mtr.sum(0) + 1e-8))) + 1e-8
    return y_mu, y_sd


def targets_tensor(df, prop_cols=PROP_COLS):
    """Return (Y, M): raw target values (NaN->0) and observation mask."""
    Y = torch.tensor(df[prop_cols].values.astype(np.float32))
    M = torch.isfinite(Y).float()
    Y[~torch.isfinite(Y)] = 0.0
    return Y, M

def rmse_fn(y_true, y_pred):
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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


def evaluate_and_save(model_name, df_test, Yte_np, yhat_te, Mte_np, out_prefix,
                       prop_cols=PROP_COLS):
    """Standard evaluation + CSV saving. Mirrors proposed_model_metrics.csv schema
    (Property, Model, R2, MAE, RMSE, pearson_r/p, spearman_rho/p, Test_N, Note)."""
    rows, pred_long = [], []
    for j, prop in enumerate(prop_cols):
        m = Mte_np[:, j].astype(bool)
        te_n = int(m.sum())
        if te_n < 3:
            rows.append({"Property": prop, "Model": model_name, "R2": np.nan, "MAE": np.nan,
                         "RMSE": np.nan, "pearson_r": np.nan, "pearson_p": np.nan,
                         "spearman_rho": np.nan, "spearman_p": np.nan, "Test_N": te_n,
                         "Note": "Insufficient data"})
            continue
        yt = Yte_np[m, j]
        yp = yhat_te[m, j]
        pr, pp, sr, sp = safe_corr(yt, yp)
        rows.append({
            "Property": prop, "Model": model_name,
            "R2": float(r2_score(yt, yp)), "MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": rmse_fn(yt, yp), "pearson_r": pr, "pearson_p": pp,
            "spearman_rho": sr, "spearman_p": sp, "Test_N": te_n, "Note": ""
        })
        for d, s, yt_i, yp_i in zip(df_test.loc[m, "Drug"].values,
                                     df_test.loc[m, "SMILES"].values, yt, yp):
            pred_long.append({"Drug": d, "SMILES": s, "Property": prop, "Model": model_name,
                               "y_true": float(yt_i), "y_pred": float(yp_i)})

    metrics = pd.DataFrame(rows).sort_values("Property")
    metrics.to_csv(_p(f"{out_prefix}_metrics.csv"), index=False)
    pd.DataFrame(pred_long).to_csv(_p(f"{out_prefix}_predictions_long.csv"), index=False)
    print(f"[{model_name}] Saved -> {out_prefix}_metrics.csv / {out_prefix}_predictions_long.csv")
    return metrics


def train_generic(model, train_items, val_items, collate_fn, y_mu, y_sd,
                   epochs=500, batch=16, lr=2e-3, weight_decay=1e-4, patience=60,
                   seed=SEED, verbose=False):


    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.7, patience=10, min_lr=5e-5)

    best, best_state, bad = 1e30, None, 0
    rng = np.random.RandomState(seed)
    n = len(train_items)

    for ep in range(1, epochs + 1):
        model.train()
        idx = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch):
            sel = idx[i:i + batch]
            items = [train_items[k] for k in sel]
            xb, yb_raw, mb = collate_fn(items)
            yb = (yb_raw - y_mu) / y_sd
            opt.zero_grad()
            yhat = model(xb)
            loss = ((yhat - yb) ** 2 * mb).sum() / (mb.sum() + 1e-8)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot += loss.item() * len(items)

        model.eval()
        with torch.no_grad():
            xv, yv_raw, mv = collate_fn(val_items)
            yv_hat = model(xv) * y_sd + y_mu
            val_rmse = torch.sqrt(
                (((yv_hat - yv_raw) ** 2 * mv).sum() / (mv.sum() + 1e-8))
            ).item()
        sched.step(val_rmse)

        if verbose and ep % 50 == 0:
            print(f"    Ep {ep:03d} | train_MSE {tot / n:.5f} | val_RMSE {val_rmse:.5f}")

        if val_rmse + 1e-8 < best:
            best_state = copy.deepcopy(model.state_dict())
            best = val_rmse
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print("    Early stopping.")
                break

    model.load_state_dict(best_state)
    return model, best


def make_val_split(train_items, val_frac=0.15, seed=SEED):
    """Same 15%-of-train validation carve-out used by proposed_validated_pipeline.py
    (Section 2 / Section 4 final-model training)."""
    n = len(train_items)
    perm = np.random.RandomState(seed).permutation(n)
    val_n = max(1, int(val_frac * n))
    val_idx = perm[:val_n]
    tr_idx = perm[val_n:]
    tr_items = [train_items[i] for i in tr_idx]
    val_items = [train_items[i] for i in val_idx]
    return tr_items, val_items


if __name__ == "__main__":
    # Smoke test
    df = load_merged()
    df_train, df_test = canonical_split(df)
    print("Merged:", df.shape, "| Train:", df_train.shape, "| Test:", df_test.shape)
    print("Test drugs match cv_test_split.csv: OK")
    Y, M = targets_tensor(df_train)
    print("Targets tensor:", Y.shape, M.shape)
    print("Harness smoke test PASSED.")
