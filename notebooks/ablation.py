import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# -----------------------------
# Common XGBoost parameters
# -----------------------------
params = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss"
}


# -----------------------------
# XGBoost + scale_pos_weight
# -----------------------------
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

xgb = XGBClassifier(
    **params,
    scale_pos_weight=neg / pos
)

xgb.fit(X_train, y_train)

xgb_prob = xgb.predict_proba(X_test)[:, 1]
xgb_pred = (xgb_prob >= 0.80).astype(int)


# -----------------------------
# Isolation Forest
# -----------------------------
iso = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42
)

iso.fit(X_train[y_train == 0])

raw = -iso.decision_function(X_test)

iso_min = raw.min()
iso_max = raw.max()

iso_score = (raw - iso_min) / (iso_max - iso_min)
iso_pred = (iso_score >= 0.80).astype(int)


# -----------------------------
# Ensemble
# -----------------------------
risk = 0.7 * xgb_prob + 0.3 * iso_score
ensemble_pred = (risk >= 0.80).astype(int)


# -----------------------------
# Metrics
# -----------------------------
def metrics(name, pred, score):
    print(f"\n{name}")
    print(f"Precision : {precision_score(y_test, pred):.4f}")
    print(f"Recall    : {recall_score(y_test, pred):.4f}")
    print(f"F1        : {f1_score(y_test, pred):.4f}")
    print(f"AUC-PR    : {average_precision_score(y_test, score):.4f}")


print("\n======================================")
print("        FRAUDGUARD ABLATION")
print("======================================")

metrics("XGBoost + scale_pos_weight", xgb_pred, xgb_prob)
metrics("Isolation Forest", iso_pred, iso_score)
metrics("Ensemble", ensemble_pred, risk)

print("\n======================================")
print("        COMPARISON")
print("======================================")
print("XGBoost handles known fraud patterns.")
print("Isolation Forest detects anomalous behavior.")
print("The ensemble combines both signals.")