import json, random
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import mlflow, mlflow.pytorch
from mlflow.models import infer_signature
from matplotlib import pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri("http://localhost:5005")
# ---------- device ----------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)

def load_params():
    with open(ROOT/"params.yaml","r") as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class RNNClassifier(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, n_classes=6, n_layers=3, dropout=0.3):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x: (B, T, C)
        _, h_n = self.rnn(x)     # (num_layers, B, H)
        h_last = h_n[-1]         # (B, H)
        return self.fc(h_last)   # (B, C)

def maybe_adjust_seq_len(Xtr, Xte, target_len: int | None):
    if not target_len:
        return Xtr, Xte
    T = Xtr.shape[1]
    if target_len == T:
        return Xtr, Xte
    if target_len < T:
        s = (T - target_len) // 2
        return Xtr[:, s:s+target_len, :], Xte[:, s:s+target_len, :]
    # pad at end
    pad = target_len - T
    Xtr_p = np.pad(Xtr, ((0,0),(0,pad),(0,0)))
    Xte_p = np.pad(Xte, ((0,0),(0,pad),(0,0)))
    return Xtr_p, Xte_p

def main():
    p = load_params()
    seed = int(p.get("seed", 2025))
    set_seed(seed)

    r = p.get("rnn", {})
    seq_len   = r.get("seq_len", None)
    hidden    = int(r.get("hidden_size", 64))
    n_layers  = int(r.get("n_layers", 3))
    dropout   = float(r.get("dropout", 0.2))
    batch_tr  = int(r.get("batch_size", 128))
    batch_te  = max(1024, batch_tr * 4)
    epochs    = int(r.get("epochs", 20))
    lr        = float(r.get("lr", 1e-3))
    clip_norm = float(r.get("clip_norm", 1.0))

    # ---- load sequences (N, T, F) from preprocess_seq.py ----
    d = np.load(ROOT/"data/processed/seq.npz", allow_pickle=True)
    Xtr_raw, ytr = d["X_train"], d["y_train"].astype(np.int64)
    Xte_raw, yte = d["X_test"],  d["y_test"].astype(np.int64)

    # optional center crop/pad to rnn.seq_len
    Xtr_raw, Xte_raw = maybe_adjust_seq_len(Xtr_raw, Xte_raw, seq_len)

    Ntr, T, F = Xtr_raw.shape
    n_classes = int(ytr.max() + 1)

    # per-channel standardization (fit on train)
    scaler = StandardScaler().fit(Xtr_raw.reshape(-1, F))
    Xtr = scaler.transform(Xtr_raw.reshape(-1, F)).reshape(Ntr, T, F).astype(np.float32)
    Xte = scaler.transform(Xte_raw.reshape(-1, F)).reshape(Xte_raw.shape).astype(np.float32)

    # loaders
    train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                              batch_size=batch_tr, shuffle=True, drop_last=False)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
                              batch_size=batch_te, shuffle=False)

    # model/optim/loss
    model = RNNClassifier(input_dim=F, hidden_dim=hidden, n_classes=n_classes,
                          n_layers=n_layers, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    # train
    losses = []
    mlflow.set_experiment("har-uci")
    with mlflow.start_run(run_name="rnn-seq"):
        for ep in range(1, epochs + 1):
            model.train()
            tot, nobs = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = crit(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                opt.step()
                bs = xb.size(0)
                tot += loss.item() * bs
                nobs += bs
            losses.append(tot / max(nobs, 1))
            if ep % 5 == 0 or ep == 1:
                print(f"Epoch {ep:02d}  loss {losses[-1]:.4f}")

        # eval
        model.eval()
        logits_all, y_all = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                logits_all.append(model(xb.to(device)).cpu())
                y_all.append(yb)
        logits = torch.cat(logits_all, dim=0)
        y_true = torch.cat(y_all, dim=0).numpy()
        y_pred = logits.argmax(1).numpy()
        acc = float(accuracy_score(y_true, y_pred))
        f1m = float(f1_score(y_true, y_pred, average="macro"))
        print(f"\nTest accuracy (RNN on raw windows): {acc*100:.2f}%")

        # confusion matrix artifact
        cm = confusion_matrix(y_true, y_pred)
        (ROOT/"reports").mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title("Confusion Matrix (RNN)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        classes = ["WALK","UP","DOWN","SIT","STAND","LAY"]
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        fig.tight_layout()
        cm_path = ROOT/"reports"/"rnn_confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)

        # persist model & metrics
        (ROOT/"models").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ROOT/"models"/"rnn.pth")
        with open(ROOT/"reports"/"metrics_rnn.json","w") as f:
            json.dump({"acc_test": acc, "f1_macro_test": f1m}, f, indent=2)

        # MLflow metrics + model with signature (positional args) + input example
        mlflow.log_metrics({"rnn_acc_test": acc, "rnn_f1_macro_test": f1m})
        x_example = Xte[:1]  # numpy (1, T, F)
        with torch.no_grad():
            y_example = model(torch.from_numpy(x_example).to(device)).cpu().numpy()
        signature = infer_signature(x_example, y_example)
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="rnn_model",
            signature=signature,
            input_example=x_example
        )
        mlflow.log_artifact(str(cm_path))

if __name__ == "__main__":
    main()
