import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]

# ---- RNN class identical to train_rnn.py ----
class RNNClassifier(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, n_classes=6, n_layers=3, dropout=0.0):
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
        _, h_n = self.rnn(x)   # (num_layers, B, H)
        h_last = h_n[-1]       # (B, H)
        return self.fc(h_last) # (B, C)

def infer_rnn_config_from_state(state):
    # hidden_dim & input_dim from first layer weights
    w_ih0 = state["rnn.weight_ih_l0"]              # (hidden, input)
    hidden_dim = w_ih0.shape[0]
    input_dim  = w_ih0.shape[1]
    # number of layers = count of weight_ih_l{idx}
    n_layers = len({k for k in state.keys() if k.startswith("rnn.weight_ih_l")})
    # n_classes from fc weight/out_features
    if "fc.weight" in state:
        n_classes = state["fc.weight"].shape[0]
    else:
        # fallback: try bias
        n_classes = state.get("fc.bias", torch.empty(0)).numel()
    return int(input_dim), int(hidden_dim), int(n_layers), int(n_classes)

def main():
    # ---- RF eval on features ----
    Xf = pd.read_parquet(ROOT/"data/processed/X_test_feat.parquet").values
    yf = pd.read_parquet(ROOT/"data/processed/y_test.parquet")["y"].values
    rf = joblib.load(ROOT/"models"/"rf.pkl")
    yhat_rf = rf.predict(Xf)
    rf_acc = float(accuracy_score(yf, yhat_rf))
    rf_f1  = float(f1_score(yf, yhat_rf, average="macro"))

    # ---- RNN eval on sequences ----
    d = np.load(ROOT/"data/processed/seq.npz", allow_pickle=True)
    Xte = d["X_test"].astype(np.float32)      # (N, T, F)
    yte = d["y_test"].astype(np.int64)

    state = torch.load(ROOT/"models"/"rnn.pth", map_location="cpu")
    input_dim, hidden_dim, n_layers, n_classes = infer_rnn_config_from_state(state)

    model = RNNClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_classes=n_classes,
        n_layers=n_layers,
        dropout=0.0
    )
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(Xte))
        yhat_rnn = logits.argmax(dim=1).numpy()

    rnn_acc = float(accuracy_score(yte, yhat_rnn))
    rnn_f1  = float(f1_score(yte, yhat_rnn, average="macro"))

    # ---- Write comparison ----
    (ROOT/"reports").mkdir(parents=True, exist_ok=True)
    with open(ROOT/"reports"/"compare.json","w") as f:
        json.dump({"rf":{"acc":rf_acc,"f1_macro":rf_f1},
                   "rnn":{"acc":rnn_acc,"f1_macro":rnn_f1}}, f, indent=2)

    print("Compare:", {"rf_acc": rf_acc, "rnn_acc": rnn_acc})

if __name__ == "__main__":
    main()

