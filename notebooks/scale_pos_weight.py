import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# Load data
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


# Common parameters
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
# SMOTE
# -----------------------------
smote = SMOTE(random_state=42)

X_smote, y_smote = smote.fit_resample(X_train, y_train)

smote_model = XGBClassifier(**params)
smote_model.fit(X_smote, y_smote)

smote_prob = smote_model.predict_proba(X_test)[:, 1]
smote_pred = (smote_prob >= 0.80).astype(int)


# -----------------------------
# scale_pos_weight
# -----------------------------
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

weight = neg / pos

weighted_model = XGBClassifier(
    **params,
    scale_pos_weight=weight
)

weighted_model.fit(X_train, y_train)

weighted_prob = weighted_model.predict_proba(X_test)[:, 1]
weighted_pred = (weighted_prob >= 0.80).astype(int)


# -----------------------------
# Results
# -----------------------------
results = []

for name, pred, prob in [
    ("XGBoost + SMOTE", smote_pred, smote_prob),
    ("XGBoost + scale_pos_weight", weighted_pred, weighted_prob)
]:
    results.append({
        "Model": name,
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "AUC-PR": average_precision_score(y_test, prob)
    })


results_df = pd.DataFrame(results)

print("\n======================================")
print("   SMOTE vs SCALE_POS_WEIGHT")
print("======================================")

print(f"\nscale_pos_weight = {weight:.2f}\n")
print(results_df.round(4).to_string(index=False))


# -----------------------------
# Save winning model
# -----------------------------
best = results_df.loc[results_df["AUC-PR"].idxmax(), "Model"]

if best == "XGBoost + scale_pos_weight":
    joblib.dump(weighted_model, "models/fraud_model.pkl")
    print("\nSaved winning model:")
    print("models/fraud_model.pkl")
else:
    joblib.dump(smote_model, "models/fraud_model.pkl")
    print("\nSaved winning model:")
    print("models/fraud_model.pkl")


# Save comparison
results_df.to_csv(
    "models/imbalance_comparison.csv",
    index=False
)

print("\nSaved: models/imbalance_comparison.csv")