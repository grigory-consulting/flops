import numpy as np, pandas as pd, math, json, yaml
from pathlib import Path
from scipy.signal import welch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

def load_params(): 
    with open(ROOT/"params.yaml","r") as f: return yaml.safe_load(f)

def load_uci_split():
    base = ROOT/"data/raw/UCI_HAR_Dataset"
    sigs = [
        "body_acc_x","body_acc_y","body_acc_z",
        "body_gyro_x","body_gyro_y","body_gyro_z"
    ]
    def read_block(folder):
        X = {}
        for s in sigs:
            p = base/f"{folder}/Inertial Signals/{s}_{folder}.txt"
            X[s] = np.loadtxt(p)
        y = np.loadtxt(base/f"{folder}/y_{folder}.txt").astype(int)  # 1..6
        return X, y
    Xtr, ytr = read_block("train")
    Xte, yte = read_block("test")
    return sigs, Xtr, ytr, Xte, yte

def stat_feat(x, kind):
    if kind=="mean": return float(np.mean(x))
    if kind=="std":  return float(np.std(x))
    if kind=="rms":  return float(np.sqrt(np.mean(x**2)))
    if kind=="mad":  return float(np.mean(np.abs(x-np.mean(x))))
    if kind=="iqr":  return float(np.percentile(x,75)-np.percentile(x,25))
    if kind=="min":  return float(np.min(x))
    if kind=="max":  return float(np.max(x))
    if kind=="skew": return float(pd.Series(x).skew())
    if kind=="kurt": return float(pd.Series(x).kurt())
    raise ValueError(kind)

def bandpower(x, fs, lo, hi):
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), 256))
    m = (f>=lo)&(f<=hi)
    return float(np.trapezoid(Pxx[m], f[m])) if m.any() else 0.0

def main():
    p = load_params()
    fs = p["data"]["sample_rate_hz"]
    stats = p["features"]["stats"]
    bands = p["features"]["bands"]

    sigs, Xtr, ytr, Xte, yte = load_uci_split()

    def build_df(X, y):
        rows = []
        for i in range(len(y)):
            r = {}
            for s in sigs:
                x = X[s][i]   # length 128 window already
                for k in stats:
                    r[f"{s}_{k}"] = stat_feat(x, k)
                for bi,(lo,hi) in enumerate(bands):
                    r[f"{s}_bp{bi}"] = bandpower(x, fs, lo, hi)
            r["y"] = int(y[i]-1)  # make labels 0..5
            rows.append(r)
        return pd.DataFrame(rows)

    df_tr = build_df(Xtr, ytr)
    df_te = build_df(Xte, yte)

    X_train = df_tr.drop(columns=["y"])
    y_train = df_tr["y"].values
    X_test  = df_te.drop(columns=["y"])
    y_test  = df_te["y"].values

    (ROOT/"data/processed").mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(ROOT/"data/processed/X_train_feat.parquet", index=False)
    X_test.to_parquet(ROOT/"data/processed/X_test_feat.parquet", index=False)
    pd.DataFrame({"y":y_train}).to_parquet(ROOT/"data/processed/y_train.parquet", index=False)
    pd.DataFrame({"y":y_test}).to_parquet(ROOT/"data/processed/y_test.parquet", index=False)

    (ROOT/"reports").mkdir(parents=True, exist_ok=True)
    pd.Series({
        "n_train": len(y_train), "n_test": len(y_test),
        "classes": 6
    }).to_json(ROOT/"reports/preprocess_features.json")
    print("Feature preprocessing done:", X_train.shape, X_test.shape)

if __name__=="__main__":
    main()
