
import warnings, random
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch, torch.nn as nn, copy
import os as _os
_DATA_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "01_data")

SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
E
INDEX_SHORT_NAMES = ["M1", "M2", "mM2", "ReM3", "H", "F", "AZ", "ISI", "SDD"]
BEST_CFG = {"hidden":384, "n_layers":2, "p_drop":0.15, "lr":5e-3, "weight_decay":1e-5, "batch":48}

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
    return (Xz, mu, sd) if Xte is None else (Xz,(Xte-mu)/sd,mu,sd)
def mask_stats(Ytr,Mtr):
    mu=Ytr.sum(0)/(Mtr.sum(0)+1e-8)
    sd=torch.sqrt((((Ytr-mu)**2*Mtr).sum(0)/(Mtr.sum(0)+1e-8)))+1e-8
    return mu,sd
class Net(nn.Module):
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
    model=Net(din,dout,hidden,p_drop,n_layers)
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
desired=["(MW)"," (HAC)"," (EM)"," (C)",
         " (P)","(MV)"," (MR)"]
prop_cols=[c for c in desired if c in df.columns]
idx=np.arange(len(df))
train_idx,test_idx=train_test_split(idx,test_size=0.2,random_state=SEED,shuffle=True)
df_train=df.iloc[train_idx].reset_index(drop=True)
X_full=df_train[pred_cols].values.astype(np.float32)
Y_full=df_train[prop_cols].values.astype(np.float32)
kf=KFold(n_splits=5,shuffle=True,random_state=SEED)
rows=[]
for fold,(tr_i,va_i) in enumerate(kf.split(X_full),1):
    Xtr=torch.from_numpy(X_full[tr_i]); Ytr=torch.from_numpy(Y_full[tr_i])
    Mtr=torch.isfinite(Ytr).float(); Ytr[~torch.isfinite(Ytr)]=0.0
    Xv=torch.from_numpy(X_full[va_i]); Yv=torch.from_numpy(Y_full[va_i])
    Mv=torch.isfinite(Yv).float(); Yv[~torch.isfinite(Yv)]=0.0
    Xtrz,Xvz,mux,sdx=standardize(Xtr,Xv)
    ymu,ysd=mask_stats(Ytr,Mtr); Ytrz=(Ytr-ymu)/ysd
    model=train_model(Xtrz,Ytrz,Mtr,Xvz,Yv,Mv,ymu,ysd,len(pred_cols),len(prop_cols),**BEST_CFG)
    model.eval()
    with torch.no_grad(): yhat=(model(Xvz)*ysd+ymu).numpy()
    for j,prop in enumerate(prop_cols):
        m=Mv[:,j].numpy().astype(bool)
        if m.sum()<3: continue
        yt=Yv[m,j].numpy(); yp=yhat[m,j]
        rows.append({"Fold":fold,"Property":prop,"R2":r2_score(yt,yp),"MAE":mean_absolute_error(yt,yp),"RMSE":rmse_fn(yt,yp)})
    print(f"fold {fold} done")
kdf=pd.DataFrame(rows)
summary=kdf.groupby("Property")[["R2","MAE","RMSE"]].agg(["mean","std"]).round(4)
print(summary.to_string())
summary.to_csv("magnet_tuned_kfold_summary.csv")
r2_means = kdf.groupby("Property")["R2"].mean()
print(f"\nR2 range across properties (mean over folds): min={r2_means.min():.3f} ({r2_means.idxmin()}), max={r2_means.max():.3f} ({r2_means.idxmax()})")
