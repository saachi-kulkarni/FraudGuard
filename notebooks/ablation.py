import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)


# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]


# Same test split used for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# -----------------------------
# Load saved models
# -----------------------------

xgb_model = joblib.load("models/fraud_model.pkl")
if_model = joblib.load("models/isolation_forest.pkl")
if_meta = joblib.load("models/isolation_forest_meta.pkl")
config = joblib.load("models/ensemble_config.pkl")


# -----------------------------
# Get test-set scores
# -----------------------------

xgb_scores = xgb_model.predict_proba(X_test)[:, 1]

raw_scores = -if_model.decision_function(X_test)

if_scores = (
    (raw_scores - if_meta["if_min"])
    / (if_meta["if_max"] - if_meta["if_min"])
)

if_scores = if_scores.clip(0, 1)

ensemble_scores = (
    config["xgb_weight"] * xgb_scores
    + config["if_weight"] * if_scores
)


# -----------------------------
# Evaluation
# -----------------------------

def evaluate(name, scores, threshold):

    predictions = (
        scores >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc_pr = average_precision_score(
        y_test,
        scores
    )

    print(f"\n{name}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"AUC-PR    : {auc_pr:.4f}")

    return {
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC-PR": auc_pr
    }


# -----------------------------
# Compare
# -----------------------------

results = []

results.append(
    evaluate("XGBoost", xgb_scores, 0.80)
)

results.append(
    evaluate("Isolation Forest", if_scores, 0.80)
)

results.append(
    evaluate(
        "Ensemble",
        ensemble_scores,
        config["threshold"]
    )
)


# -----------------------------
# Save results
# -----------------------------

comparison = pd.DataFrame(results)

print("\n======================================")
print("       TEST-SET ABLATION")
print("======================================")

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

comparison.to_csv(
    "models/ablation_results.csv",
    index=False
)

print("\nSaved: models/ablation_results.csv")