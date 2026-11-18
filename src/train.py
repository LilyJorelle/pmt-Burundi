"""train.py: train the chosen technique (XGBoost) on the clean table and save the model for the API.

Reads data/processed/clean.csv (made by preprocess.py), repeats the split of notebook 02,
Step 1, trains XGBoost, checks it once on the test set, and saves it in models/.
"""

import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from ingest import ROOT

# Input and output files. The saved model holds tree rules, not household rows.
DATA_FILE = ROOT / "data" / "processed" / "clean.csv"
MODEL_FILE = ROOT / "models" / "xgb_model.joblib"
INFO_FILE = ROOT / "models" / "model_info.json"

# Same settings as notebook 02 (Step 1 split, Step 3 technique), so the results can be compared.
SEED = 42
TEST_SHARE = 0.20
CLUSTER_COLUMN, LABEL = "hh1", "poor"
XGB_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=SEED)



def load_and_split():
    """Read the clean table and split it by cluster: every cluster stays on one side."""
    data = pd.read_csv(DATA_FILE)
    predictors = [c for c in data.columns if c not in (CLUSTER_COLUMN, LABEL)]
    X, y, groups = data[predictors], data[LABEL], data[CLUSTER_COLUMN]
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=TEST_SHARE, random_state=SEED).split(X, y, groups))
    return predictors, X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def run():
    """Train XGBoost on the training set, score it once on the test set, and save the model and its info."""
    predictors, X_train, X_test, y_train, y_test = load_and_split()
    model = XGBClassifier(**XGB_PARAMS).fit(X_train, y_train)

    # One honest check on households the model has never seen (poor if the predicted risk is 0.5 or more).
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {"accuracy": accuracy_score(y_test, pred), "recall": recall_score(y_test, pred),
               "precision": precision_score(y_test, pred), "f1": f1_score(y_test, pred),
               "roc_auc": roc_auc_score(y_test, proba)}
    metrics = {k: round(float(v), 3) for k, v in metrics.items()}

    # Save the model, and an info file with the predictors in order (the API must send them in this order).
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    info = {"technique": "XGBoost", "label": "poor = bottom 40% of the item count m11cons, among filled interviews",
            "seed": SEED, "test_share": TEST_SHARE, "train_households": len(X_train), "test_households": len(X_test),
            "params": XGB_PARAMS, "test_metrics": metrics, "predictors": predictors}
    INFO_FILE.write_text(json.dumps(info, indent=2), encoding="utf-8")

    # Summary only: counts and scores, no household rows.
    print(f"Train: {len(X_train)} households | Test: {len(X_test)} households | Predictors: {len(predictors)}")
    print("Test scores:", metrics)
    print(f"Saved: {MODEL_FILE.name} and {INFO_FILE.name} in models/")


if __name__ == "__main__":
    # Runs only when the file is executed directly: python src/train.py
    run()