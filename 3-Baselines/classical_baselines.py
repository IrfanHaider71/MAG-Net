
import pandas as pd
import numpy as np
import os, random, warnings

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.astype(str).str.strip()
    return df

def clean_drug_col(df: pd.DataFrame, col="Drug") -> pd.DataFrame:
    df = standardize_columns(df)
    if col not in df.columns:
        cand = [c for c in df.columns if c.lower().strip() == "drug"]
        if cand:
            df = df.rename(columns={cand[0]: "Drug"})
        else:
            raise ValueError("Column 'Drug' not found in input.")
    df["Drug"] = df["Drug"].astype(str).str.strip()
    return df

def coerce_numeric_except(df: pd.DataFrame, skip_cols) -> pd.DataFrame:
    for c in df.columns:
        if c in skip_cols: continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def ensure_cols(df: pd.DataFrame, cols, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")

def rmse_fn(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

DO_TUNE  = True
CV       = KFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING  = "neg_mean_squared_error"

def _print_cv_result(name, search):
    if SCORING == "neg_mean_squared_error":
        cv_rmse = (-search.best_score_) ** 0.5
        print(f"[CV] {name} best params={search.best_params_} | CV RMSE={cv_rmse:.4f}")
    else:
        print(f"[CV] {name} best params={search.best_params_} | best_score={search.best_score_:.4f}")

def tune_models_on_train(Xtr, ytr):
    models = {}

    # RandomForest
    rf = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    rf_space = {
        "n_estimators": [300, 500, 800, 1000, 1200],
        "max_depth": [None, 10, 15, 20, 30],
        "min_samples_leaf": [1, 2, 3, 5, 8],
        "max_features": ["sqrt", "log2", 0.5, 0.8, 1.0],
        "bootstrap": [True]
    }
    rf_search = RandomizedSearchCV(rf, rf_space, n_iter=40, cv=CV, scoring=SCORING,
                                   random_state=SEED, n_jobs=-1, refit=True)
    rf_search.fit(Xtr, ytr)
    _print_cv_result("RandomForest", rf_search)
    models["RandomForest"] = rf_search.best_estimator_

    # GradientBoosting
    gbr = GradientBoostingRegressor(random_state=SEED)
    gbr_space = {
        "n_estimators": [200, 300, 500, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4],
        "subsample": [0.6, 0.8, 1.0],
        "min_samples_leaf": [1, 2, 3, 5]
    }
    gbr_search = RandomizedSearchCV(gbr, gbr_space, n_iter=40, cv=CV, scoring=SCORING,
                                    random_state=SEED, n_jobs=-1, refit=True)
    gbr_search.fit(Xtr, ytr)
    _print_cv_result("GradientBoosting", gbr_search)
    models["GradientBoosting"] = gbr_search.best_estimator_

    return models

import os as _os

for f in [F_SMILES, F_EXACT, F_PROPS]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing input file: {f}")

smiles_df = clean_drug_col(pd.read_csv(F_SMILES))
exact_df  = clean_drug_col(pd.read_csv(F_EXACT))
props_df  = clean_drug_col(pd.read_csv(F_PROPS))

ensure_cols(smiles_df, ["Drug", "SMILES"], "SMILES file")

exact_index_cols = [
    "M1 (First Zagreb)", "M2 (Second Zagreb)", "mM2 (Modified Second Zagreb)",
    "ReM3 (Redefined M3)", "Harmonic Index", "Forgotten Index",
    "AZ (Augmented Zagreb)", "ISI (Inverse Sum Indeg)", "SDD (Symmetric Division Degree)"
]

df = (smiles_df
      .merge(exact_df, on="Drug", how="inner")
      .merge(props_df, on="Drug", how="inner"))

if "SMILES" in df.columns:
    df["SMILES"] = df["SMILES"].astype(str).str.strip()
if "Class" in df.columns:
    df["Class"] = df["Class"].astype(str).str.strip()

before = len(df)
df = df.drop_duplicates(subset=["Drug", "SMILES"]).copy()
if len(df) < before:
    print(f"Dropped {before - len(df)} duplicate rows (Drug+SMILES).")

id_cols = ["Drug", "SMILES", "Class"]
non_target = set(id_cols + exact_index_cols)
df = coerce_numeric_except(df, skip_cols=list(non_target))

desired_order = [" (MW)", " (HAC)", "(EM)",
                 " (C)", "(P)", "(MV)", " (MR)"]
prop_cols = [c for c in desired_order if c in df.columns]

if not prop_cols:
    raise ValueError("No numeric property columns detected.")
print("Detected properties:", prop_cols)

df_feat_ok = df.dropna(subset=exact_index_cols).copy()
if len(df_feat_ok) < 10:
    raise ValueError("Too few rows with exact indices after merge.")

idx = np.arange(len(df_feat_ok))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, shuffle=True)
df_train = df_feat_ok.iloc[train_idx].reset_index(drop=True)
df_test  = df_feat_ok.iloc[test_idx].reset_index(drop=True)

if set(df_train["Drug"]) & set(df_test["Drug"]):
    raise RuntimeError("Train/Test Drug overlap detected.")
if set(df_train["SMILES"]) & set(df_test["SMILES"]):
    raise RuntimeError("Train/Test SMILES overlap detected.")
print("Train/Test entity overlap: NONE ")

cols_to_save = ["Drug", "SMILES"] + (["Class"] if "Class" in df_test.columns else [])
df_test[cols_to_save].to_csv("cv_test_split.csv", index=False)
print(f"Saved test split -> cv_test_split.csv  (N_test={len(df_test)})")

X_train_all = df_train[exact_index_cols].values
X_test_all  = df_test[exact_index_cols].values

metrics_records = []
pred_long_rows  = []

for prop in prop_cols:
    tr_mask = df_train[prop].notna().values
    te_mask = df_test[prop].notna().values
    tr_n, te_n = int(tr_mask.sum()), int(te_mask.sum())

    if tr_n < 15 or te_n < 5:
        metrics_records.append({
            "Property": prop, "Model": "NA", "R2": np.nan, "MAE": np.nan, "RMSE": np.nan,
            "Train_N": tr_n, "Test_N": te_n, "Note": "Insufficient data"
        })
        continue

    Xtr, ytr = X_train_all[tr_mask], df_train.loc[tr_mask, prop].values
    Xte, yte = X_test_all[te_mask],  df_test.loc[te_mask,  prop].values
    drugs_te = df_test.loc[te_mask, "Drug"].values
    smiles_te = df_test.loc[te_mask, "SMILES"].values

    if DO_TUNE:
        models_for_prop = tune_models_on_train(Xtr, ytr)
    else:

        models_for_prop = {
            "RandomForest": RandomForestRegressor(n_estimators=800, max_depth=None, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=300, random_state=SEED),
        }
        for mdl in models_for_prop.values():
            mdl.fit(Xtr, ytr)

    for name, mdl in models_for_prop.items():
        yhat = mdl.predict(Xte)
        r2 = r2_score(yte, yhat)
        mae = mean_absolute_error(yte, yhat)
        rmse = rmse_fn(yte, yhat)
        metrics_records.append({
            "Property": prop, "Model": name, "R2": r2, "MAE": mae, "RMSE": rmse,
            "Train_N": tr_n, "Test_N": te_n, "Note": ""
        })
        for d, s, yt, yp in zip(drugs_te, smiles_te, yte, yhat):
            pred_long_rows.append({
                "Drug": d, "SMILES": s, "Property": prop, "Model": name,
                "y_true": float(yt), "y_pred": float(yp)
            })

metrics_df = pd.DataFrame(metrics_records)
model_order = ["RandomForest", "GradientBoosting"]
metrics_df["Model"] = pd.Categorical(metrics_df["Model"], categories=model_order, ordered=True)
metrics_df["Property"] = pd.Categorical(metrics_df["Property"], categories=prop_cols, ordered=True)
metrics_df = metrics_df.sort_values(["Property", "Model"])
metrics_df.to_csv("cv_model_metrics.csv", index=False)
print("Saved metrics (TEST) -> cv_model_metrics.csv")

pred_long = pd.DataFrame(pred_long_rows)
pred_long["Property"] = pd.Categorical(pred_long["Property"], categories=prop_cols, ordered=True)
pred_long.to_csv("cv_test_predictions_long.csv", index=False)
print("Saved predictions (long) -> cv_test_predictions_long.csv")

if not pred_long.empty:
    wide = pred_long.pivot_table(index=["Drug", "SMILES"],
                                 columns=["Property", "Model"],
                                 values="y_pred", aggfunc="first")
    existing_cols = [c for c in [(p, m) for p in prop_cols for m in model_order] if c in wide.columns]
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(existing_cols))
    wide.columns = [f"{p}__{m}" for (p, m) in wide.columns]
    wide = wide.reset_index()

    ytrue_example = pred_long.pivot_table(index=["Drug", "SMILES"],
                                          columns="Property",
                                          values="y_true", aggfunc="first").reset_index()
    wide = wide.merge(ytrue_example, on=["Drug", "SMILES"], how="left")
    wide.to_csv("cv_test_predictions_wide.csv", index=False)
    print("Saved predictions (wide) -> cv_test_predictions_wide.csv")

if not pred_long.empty:
    rows = []
    for (prop, model), g in pred_long.groupby(["Property", "Model"]):
        x = np.asarray(g["y_true"], float)
        y = np.asarray(g["y_pred"], float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size >= 3:
            try:
                from scipy.stats import pearsonr, spearmanr
                pr, pp = pearsonr(x, y)
                sr, sp = spearmanr(x, y)
            except Exception:
                pr = np.corrcoef(x, y)[0,1]
                rx = x.argsort().argsort().astype(float)
                ry = y.argsort().argsort().astype(float)
                sr = np.corrcoef(rx, ry)[0,1]
                pp = np.nan
                sp = np.nan
            rows.append({
                "Property": prop, "Model": model,
                "pearson_r": float(pr), "pearson_p": float(pp),
                "spearman_rho": float(sr), "spearman_p": float(sp),
                "N_test": int(len(x))
            })
    corr_df = pd.DataFrame(rows)
    corr_df["Property"] = pd.Categorical(corr_df["Property"], categories=prop_cols, ordered=True)
    corr_df["Model"] = pd.Categorical(corr_df["Model"], categories=model_order, ordered=True)
    corr_df = corr_df.sort_values(["Property", "Model"])
    corr_df.to_csv("pred_vs_true_property_correlations.csv", index=False)
    print("Saved -> pred_vs_true_property_correlations.csv")
