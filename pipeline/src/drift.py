import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]

def psi(expected, actual, bins=10):
    lo, hi = np.nanmin(expected), np.nanmax(expected)
    if lo == hi: hi = lo + 1e-6
    e,_ = np.histogram(expected, bins=bins, range=(lo,hi), density=True)
    a,_ = np.histogram(actual,  bins=bins, range=(lo,hi), density=True)
    e += 1e-8; a += 1e-8
    return float(np.sum((a-e) * np.log(a/e)))

def main():
    Xtr = pd.read_parquet(ROOT/"data/processed/X_train_feat.parquet")
    Xte = pd.read_parquet(ROOT/"data/processed/X_test_feat.parquet")
    ks_res, psi_res = {}, {}
    for c in Xtr.columns:
        s, p = ks_2samp(Xtr[c].values, Xte[c].values)
        ks_res[c] = {"stat": float(s), "p": float(p)}
        psi_res[c] = psi(Xtr[c].values, Xte[c].values)
    summary = {
        "ks_drifted":[c for c,v in ks_res.items() if v["p"] < 0.01],
        "psi_warn":[c for c,v in psi_res.items() if v >= 0.1],
        "psi_alert":[c for c,v in psi_res.items() if v >= 0.25],
    }
    (ROOT/"reports").mkdir(parents=True, exist_ok=True)
    with open(ROOT/"reports"/"drift.json","w") as f:
        json.dump({"ks": ks_res, "psi": psi_res, "summary": summary}, f, indent=2)
    print("Drift summary:", summary)

if __name__ == "__main__":
    main()
