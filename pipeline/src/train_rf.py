import json, yaml
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow, mlflow.sklearn, joblib
from mlflow.models import infer_signature 

ROOT = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri("http://localhost:5005")
def load_params():
    with open(ROOT/"params.yaml","r") as f:
        return yaml.safe_load(f)

def main():
    p = load_params()

    X_train_df = pd.read_parquet(ROOT/"data/processed/X_train_feat.parquet")
    X_test_df  = pd.read_parquet(ROOT/"data/processed/X_test_feat.parquet")
    y_train = pd.read_parquet(ROOT/"data/processed/y_train.parquet")["y"].values
    y_test  = pd.read_parquet(ROOT/"data/processed/y_test.parquet")["y"].values

    X_train, X_test = X_train_df.values, X_test_df.values

    mlflow.set_experiment("har-uci")
    with mlflow.start_run(run_name="rf-features"):
        clf = RandomForestClassifier(
            n_estimators=p["rf"]["n_estimators"],
            max_depth=p["rf"]["max_depth"],
            min_samples_leaf=p["rf"]["min_samples_leaf"],
            class_weight=p["rf"]["class_weight"],
            n_jobs=p["rf"]["n_jobs"],
            random_state=p["seed"]
        ).fit(X_train, y_train)

        # --- Metrics ---
        yhat_tr = clf.predict(X_train)
        yhat_te = clf.predict(X_test)
        acc_tr = float(accuracy_score(y_train, yhat_tr))
        acc_te = float(accuracy_score(y_test,  yhat_te))
        f1_te  = float(f1_score(y_test, yhat_te, average="macro"))
        mlflow.log_metrics({
            "rf_acc_train": acc_tr,
            "rf_acc_test": acc_te,
            "rf_f1_macro_test": f1_te
        })

        # --- Model logging ---
        (ROOT/"models").mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, ROOT/"models"/"rf.pkl")

        # ✅ Signature & example
        signature = infer_signature(X_train_df, clf.predict(X_train))
        input_example = X_train_df.iloc[:5]

        mlflow.sklearn.log_model(
            sk_model=clf,
            name="rf_model",                   # ✅ correct keyword
            signature=signature,
            input_example=input_example
        )

        # --- Metrics file for DVC ---
        (ROOT/"reports").mkdir(parents=True, exist_ok=True)
        with open(ROOT/"reports"/"metrics_rf.json","w") as f:
            json.dump({
                "acc_train": acc_tr,
                "acc_test": acc_te,
                "f1_macro_test": f1_te
            }, f, indent=2)

    print("RF done. acc_test=", acc_te)

if __name__=="__main__":
    main()
