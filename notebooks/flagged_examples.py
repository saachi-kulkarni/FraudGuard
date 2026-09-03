import pandas as pd
import joblib

from sklearn.model_selection import train_test_split


# Load data
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# Same unseen test split
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Load models
xgb = joblib.load("models/fraud_model.pkl")
iso = joblib.load("models/isolation_forest.pkl")
meta = joblib.load("models/isolation_forest_meta.pkl")
config = joblib.load("models/ensemble_config.pkl")

# XGBoost score
xgb_score = xgb.predict_proba(X_test)[:, 1]

# Isolation Forest score
raw = -iso.decision_function(X_test)
if_score = (raw - meta["if_min"]) / (
    meta["if_max"] - meta["if_min"]
)
if_score = if_score.clip(0, 1)

# Final ensemble risk
risk = (
    config["xgb_weight"] * xgb_score +
    config["if_weight"] * if_score
)

results = X_test.copy()
results["risk_score"] = risk
results["actual"] = y_test.values
results["prediction"] = (
    risk >= config["threshold"]
).astype(int)

# True positives
tp = results[
    (results["prediction"] == 1) &
    (results["actual"] == 1)
].sort_values("risk_score", ascending=False).head(3)

# False positives
fp = results[
    (results["prediction"] == 1) &
    (results["actual"] == 0)
].sort_values("risk_score", ascending=False).head(2)

examples = pd.concat([tp, fp])

print("\n======================================")
print("    FINAL ENSEMBLE FLAGGED EXAMPLES")
print("======================================")

print(
    examples[
        ["risk_score", "actual", "prediction",
         "Amount", "V14", "V10", "V4"]
    ].to_string()
)

examples.to_csv(
    "models/flagged_examples.csv",
    index=True
)

print("\nSaved: models/flagged_examples.csv")