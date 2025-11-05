import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"data/raw/UCI_HAR_Dataset"

def load_block(split):
    sigs = ["body_acc_x","body_acc_y","body_acc_z","body_gyro_x","body_gyro_y","body_gyro_z"]
    Xs = [np.loadtxt(BASE/f"{split}/Inertial Signals/{s}_{split}.txt") for s in sigs]
    X = np.stack(Xs, axis=-1).astype(np.float32)  # (N,128,6)
    y = (np.loadtxt(BASE/f"{split}/y_{split}.txt").astype(int) - 1).astype(np.int64)
    return X, y

def main():
    Xtr, ytr = load_block("train")
    Xte, yte = load_block("test")
    mu = Xtr.mean(axis=(0,1), keepdims=True); sigma = Xtr.std(axis=(0,1), keepdims=True)+1e-8
    Xtr = (Xtr-mu)/sigma; Xte = (Xte-mu)/sigma
    (ROOT/"data/processed").mkdir(parents=True, exist_ok=True)
    np.savez(ROOT/"data/processed/seq.npz", X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte, mu=mu, sigma=sigma)
    print("Seq ready:", Xtr.shape, Xte.shape)

if __name__ == "__main__":
    main()
