
import warnings, random, copy
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch, torch.nn as nn
import os as _os
_DATA_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "01_data")

SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

INDEX_SHORT_NAMES = ["M1", "M2", "mM2", "ReM3", "H", "F", "AZ", "ISI", "SDD"]
BEST_CFG = dict(hidden=384, n_layers=2, p_drop=0.15, lr=5e-3, weight_decay=1e-5, batch=48)

def clean_drug(df):
    df.columns = df.columns.astype(str).str.strip()
    if "Drug" not in df.columns:
        df = df.rename(columns={[c for c in df.columns if c.lower().strip()=="drug"][0]: "Drug"})
    df["Drug"] = df["Drug"].astype(str).str.strip()
    if "SMILES" in df.columns: df["SMILES"] = df["SMILES"].astype(str).str.strip()
    return df
def coerce_numeric_except(df, skip):
    for c in df.columns:
        if c not in skip: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
def rmse_fn(y, p):
    try: return mean_squared_error(y, p, squared=False)
    except TypeError: return np.sqrt(mean_squared_error(y, p))
def standardize(Xtr, Xte=None):
    mu=Xtr.mean(0,keepdim=True); sd=Xtr.std(0,keepdim=True)+1e-8
    Xz=(Xtr-mu)/sd
    return (Xz,mu,sd) if Xte is None else (Xz,(Xte-mu)/sd,mu,sd)
def mask_stats(Ytr,Mtr):
    mu=Ytr.sum(0)/(Mtr.sum(0)+1e-8)
    sd=torch.sqrt((((Ytr-mu)**2*Mtr).sum(0)/(Mtr.sum(0)+1e-8)))+1e-8
    return mu,sd

class MAGNet(nn.Module):
    def __init__(self, din, dout, hidden, p_drop, n_layers):
        super().__init__()
        layers,d=[],din
        for _ in range(n_layers):
            layers+=[nn.Linear(d,hidden), nn.ReLU(), nn.Dropout(p_drop)]; d=hidden
        self.trunk=nn.Sequential(*layers); self.out=nn.Linear(hidden,dout)
        self.gate_logits=nn.Parameter(torch.zeros(dout,din)); self.alpha=nn.Parameter(torch.ones(dout))
    def forward(self,x):
        h=self.trunk(x); base=self.out(h); W=torch.softmax(self.gate_logits,dim=1)
        return base+self.alpha*(x@W.t())

def batcher(X,Y,M,bs,rng):
    n=X.shape[0]; idx=rng.permutation(n)
    for i in range(0,n,bs): j=idx[i:i+bs]; yield X[j],Y[j],M[j]

def train_model(Xtr,Ytr,Mtr,Xv,Yv,Mv,ymu,ysd,din,dout,hidden,p_drop,n_layers,lr,weight_decay,batch,
                epochs=500,patience=60,seed=SEED):
    rng=np.random.RandomState(seed); torch.manual_seed(seed)
    model=MAGNet(din,dout,hidden,p_drop,n_layers)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=0.7,patience=10,min_lr=5e-5)
    best,bstate,bad=1e30,None,0
    for ep in range(epochs):
        model.train()
        for xb,yb,mb in batcher(Xtr,Ytr,Mtr,batch,rng):
            opt.zero_grad(); yhat=model(xb); loss=((yhat-yb)**2*mb).sum()/(mb.sum()+1e-8)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step()
        model.eval()
        with torch.no_grad():
            vh=model(Xv)*ysd+ymu; vr=torch.sqrt((((vh-Yv)**2*Mv).sum())/(Mv.sum()+1e-8)).item()
        sched.step(vr)
        if vr+1e-8<best: bstate=copy.deepcopy(model.state_dict()); best=vr; bad=0
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(bstate); return model

exact=exact.rename(columns=dict(zip(EXACT_INDEX_COLS,INDEX_SHORT_NAMES)))[["Drug"]+INDEX_SHORT_NAMES]
df=smiles.merge(exact,on="Drug",how="inner").merge(props,on="Drug",how="inner")
df=df.drop_duplicates(subset=["Drug","SMILES"]).reset_index(drop=True)
pred_cols=INDEX_SHORT_NAMES
df=coerce_numeric_except(df, set(["Drug","SMILES","Class"]+pred_cols)).dropna(subset=pred_cols).reset_index(drop=True)
desired=["(MW)","(HAC)","(EM)"," (C)",
         "(P)","(MV)"," (MR)"]
prop_cols=[c for c in desired if c in df.columns]

idx=np.arange(len(df))
train_idx,test_idx=train_test_split(idx,test_size=0.2,random_state=SEED,shuffle=True)
df_train=df.iloc[train_idx].reset_index(drop=True)
df_test=df.iloc[test_idx].reset_index(drop=True)
saved_test = pd.read_csv(_os.path.join(_DATA_DIR, "cv_test_split.csv"))
assert set(df_test["Drug"]) == set(saved_test["Drug"]), "Test split mismatch!"

X_full=df_train[pred_cols].values.astype(np.float32)
Y_full=df_train[prop_cols].values.astype(np.float32)
Xte_np=df_test[pred_cols].values.astype(np.float32)
Yte_np=df_test[prop_cols].values.astype(np.float32)

Xtr_t=torch.from_numpy(X_full); Ytr_t=torch.from_numpy(Y_full)
Mtr_t=torch.isfinite(Ytr_t).float(); Ytr_t[~torch.isfinite(Ytr_t)]=0.0
Xte_t=torch.from_numpy(Xte_np); Yte_t=torch.from_numpy(Yte_np)
Mte_t=torch.isfinite(Yte_t).float(); Yte_t[~torch.isfinite(Yte_t)]=0.0

Xtr_z,Xte_z,x_mu,x_sd=standardize(Xtr_t,Xte_t)
y_mu,y_sd=mask_stats(Ytr_t,Mtr_t)
Ytr_z=(Ytr_t-y_mu)/y_sd

n_all=Xtr_z.shape[0]
perm_all=np.random.RandomState(SEED).permutation(n_all)
val_n=max(1,int(0.15*n_all))
val_idx2=perm_all[:val_n]; tr2_idx=perm_all[val_n:]
X_tr2,Y_tr2,M_tr2=Xtr_z[tr2_idx],Ytr_z[tr2_idx],Mtr_t[tr2_idx]
X_val2,Y_val2,M_val2=Xtr_z[val_idx2],Ytr_t[val_idx2],Mtr_t[val_idx2]

model=train_model(X_tr2,Y_tr2,M_tr2,X_val2,Y_val2,M_val2,y_mu,y_sd,
                   din=len(pred_cols),dout=len(prop_cols),**BEST_CFG)
model.eval()

# ---- sanity check against previously saved metrics ----
with torch.no_grad():
    yhat_test = (model(Xte_z)*y_sd+y_mu).numpy()
for j, prop in enumerate(prop_cols):
    m = Mte_t[:, j].numpy().astype(bool)
    r2 = r2_score(Yte_np[m, j], yhat_test[m, j])
    print(f"sanity check {prop}: R2={r2:.4f}")

# ---- extract gating weight matrix ----
with torch.no_grad():
    Wg = torch.softmax(model.gate_logits, dim=1).numpy()  # (P, 9)
gate_df = pd.DataFrame(Wg, index=prop_cols, columns=INDEX_SHORT_NAMES)
gate_df.to_csv("magnet_gating_weights.csv")
print("\nGating weight matrix (rows=properties, cols=indices):")
print(gate_df.round(4).to_string())

# ---- per-drug percentage error ----
rows = []
for j, prop in enumerate(prop_cols):
    m = Mte_t[:, j].numpy().astype(bool)
    yt = Yte_np[m, j]; yp = yhat_test[m, j]
    drugs = df_test.loc[m, "Drug"].values
    pct_err = np.abs(yp - yt) / np.abs(yt) * 100.0
    for d, a, p, e in zip(drugs, yt, yp, pct_err):
        rows.append({"Drug": d, "Property": prop, "Actual": float(a), "Predicted": float(p), "PctError": float(e)})
err_df = pd.DataFrame(rows)
err_df.to_csv("magnet_pct_error.csv", index=False)

print("\nPer-property error summary:")
summary = err_df.groupby("Property")["PctError"].agg(["mean", "max", lambda x: (x > 10).sum()])
summary.columns = ["MeanPctErr", "MaxPctErr", "N_over_10pct"]
print(summary.round(2).to_string())
print("\nSaved -> magnet_gating_weights.csv, magnet_pct_error.csv")
