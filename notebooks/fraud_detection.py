import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    average_precision_score
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier


# =========================
# 1. Load dataset
# =========================

df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

print("Dataset shape:", X.shape)
print("Fraud transactions:", sum(y == 1))
print("Normal transactions:", sum(y == 0))


# =========================
# 2. Train-test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================
# 3. Logistic Regression
# + StandardScaler + SMOTE
# =========================

print("\n==============================")
print("Logistic Regression + Scaling + SMOTE")
print("==============================")

lr_smote = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

lr_smote.fit(X_train, y_train)

lr_smote_prob = lr_smote.predict_proba(X_test)[:, 1]

lr_smote_pred = (
    lr_smote_prob >= 0.80
).astype(int)

print(
    classification_report(
        y_test,
        lr_smote_pred,
        digits=4
    )
)

lr_smote_auc = average_precision_score(
    y_test,
    lr_smote_prob
)

print(
    f"AUC-PR: {lr_smote_auc:.4f}"
)


# =========================
# 4. Logistic Regression
# + StandardScaler + Class Weight
# =========================

print("\n==============================")
print("Logistic Regression + Scaling + Class Weight")
print("==============================")

lr_weighted = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

lr_weighted.fit(X_train, y_train)

lr_weighted_prob = lr_weighted.predict_proba(X_test)[:, 1]

lr_weighted_pred = (
    lr_weighted_prob >= 0.80
).astype(int)

print(
    classification_report(
        y_test,
        lr_weighted_pred,
        digits=4
    )
)

lr_weighted_auc = average_precision_score(
    y_test,
    lr_weighted_prob
)

print(
    f"AUC-PR: {lr_weighted_auc:.4f}"
)


# =========================
# 5. XGBoost + SMOTE
# =========================

print("\n==============================")
print("XGBoost + SMOTE")
print("==============================")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

xgb_smote = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

xgb_smote.fit(
    X_train_smote,
    y_train_smote
)

xgb_prob = xgb_smote.predict_proba(X_test)[:, 1]

xgb_pred = (
    xgb_prob >= 0.80
).astype(int)

print(
    classification_report(
        y_test,
        xgb_pred,
        digits=4
    )
)

xgb_auc = average_precision_score(
    y_test,
    xgb_prob
)

print(
    f"AUC-PR: {xgb_auc:.4f}"
)


# =========================
# 6. Final comparison
# =========================

print("\n==============================")
print("FINAL MODEL COMPARISON")
print("==============================")

print(
    f"Logistic + SMOTE:       {lr_smote_auc:.4f}"
)

print(
    f"Logistic + Weighting:   {lr_weighted_auc:.4f}"
)

print(
    f"XGBoost + SMOTE:        {xgb_auc:.4f}"
)


# =========================
# 7. Select best model
# =========================

models = {
    "Logistic Regression + SMOTE": (
        lr_smote_auc,
        lr_smote
    ),
    "Logistic Regression + Weighting": (
        lr_weighted_auc,
        lr_weighted
    ),
    "XGBoost + SMOTE": (
        xgb_auc,
        xgb_smote
    )
}

best_name = max(
    models,
    key=lambda name: models[name][0]
)

best_auc, best_model = models[best_name]

print("\nBest model:", best_name)
print(f"Best AUC-PR: {best_auc:.4f}")


# =========================
# 8. Save best model
# =========================

joblib.dump(
    best_model,
    "models/fraud_model.pkl"
)

print(
    "\nBest model saved to models/fraud_model.pkl"
)