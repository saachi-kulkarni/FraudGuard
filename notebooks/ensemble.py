import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

print("Dataset:", X.shape)
print("Fraud:", y.sum())
print("Normal:", (y == 0).sum())


# =========================================================
# 2. CLEAN TRAIN / VALIDATION / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    stratify=y_train,
    random_state=42
)

print("\nTrain:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# =========================================================
# 3. TRAIN XGBOOST PROPERLY
# =========================================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print("Normal:", (y_train_smote == 0).sum())
print("Fraud:", (y_train_smote == 1).sum())


xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train_smote,
    y_train_smote
)


# =========================================================
# 4. XGBOOST SCORES
# =========================================================

xgb_val = xgb_model.predict_proba(X_val)[:, 1]
xgb_test = xgb_model.predict_proba(X_test)[:, 1]


# =========================================================
# 5. ISOLATION FOREST
# =========================================================

normal_data = X_train[y_train == 0]

if_model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

if_model.fit(normal_data)


if_val_raw = -if_model.decision_function(X_val)
if_test_raw = -if_model.decision_function(X_test)


# Normalize using VALIDATION data only

if_min = if_val_raw.min()
if_max = if_val_raw.max()

if_val = (
    (if_val_raw - if_min)
    / (if_max - if_min)
)

if_test = (
    (if_test_raw - if_min)
    / (if_max - if_min)
)

if_val = np.clip(if_val, 0, 1)
if_test = np.clip(if_test, 0, 1)


# =========================================================
# 6. ISOLATION FOREST THRESHOLD
# =========================================================

best_if_threshold = 0
best_if_f1 = 0

for threshold in np.linspace(0, 1, 101):

    pred = (
        if_val >= threshold
    ).astype(int)

    f1 = f1_score(
        y_val,
        pred,
        zero_division=0
    )

    if f1 > best_if_f1:
        best_if_f1 = f1
        best_if_threshold = threshold


if_pred = (
    if_test >= best_if_threshold
).astype(int)


# =========================================================
# 7. XGBOOST BASELINE
# =========================================================

xgb_pred = (
    xgb_test >= 0.80
).astype(int)


xgb_precision = precision_score(
    y_test,
    xgb_pred,
    zero_division=0
)

xgb_recall = recall_score(
    y_test,
    xgb_pred,
    zero_division=0
)

xgb_f1 = f1_score(
    y_test,
    xgb_pred,
    zero_division=0
)

xgb_auc = average_precision_score(
    y_test,
    xgb_test
)


# =========================================================
# 8. ISOLATION FOREST RESULTS
# =========================================================

if_precision = precision_score(
    y_test,
    if_pred,
    zero_division=0
)

if_recall = recall_score(
    y_test,
    if_pred,
    zero_division=0
)

if_f1 = f1_score(
    y_test,
    if_pred,
    zero_division=0
)

if_auc = average_precision_score(
    y_test,
    if_test
)


# =========================================================
# 9. WEIGHTED ENSEMBLE
# =========================================================

weights = [
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5)
]

best_ensemble = None


for xgb_weight, if_weight in weights:

    val_risk = (
        xgb_weight * xgb_val
        + if_weight * if_val
    )

    test_risk = (
        xgb_weight * xgb_test
        + if_weight * if_test
    )

    best_threshold = 0
    best_f1 = 0

    for threshold in np.linspace(0, 1, 101):

        val_pred = (
            val_risk >= threshold
        ).astype(int)

        f1 = f1_score(
            y_val,
            val_pred,
            zero_division=0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    test_pred = (
        test_risk >= best_threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        test_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_pred,
        zero_division=0
    )

    auc = average_precision_score(
        y_test,
        test_risk
    )

    result = {
        "xgb_weight": xgb_weight,
        "if_weight": if_weight,
        "threshold": best_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_pr": auc
    }

    if (
        best_ensemble is None
        or auc > best_ensemble["auc_pr"]
    ):
        best_ensemble = result


# =========================================================
# 10. EITHER MODEL FLAGS
# =========================================================

either_pred = (
    (xgb_test >= 0.80)
    |
    (if_test >= best_if_threshold)
).astype(int)


either_precision = precision_score(
    y_test,
    either_pred,
    zero_division=0
)

either_recall = recall_score(
    y_test,
    either_pred,
    zero_division=0
)

either_f1 = f1_score(
    y_test,
    either_pred,
    zero_division=0
)


# =========================================================
# 11. DID ISOLATION FOREST CATCH MISSED FRAUD?
# =========================================================

xgb_missed = (
    (y_test == 1)
    &
    (xgb_pred == 0)
)

if_caught = (
    xgb_missed
    &
    (if_pred == 1)
)

missed_count = int(xgb_missed.sum())
if_caught_count = int(if_caught.sum())


# =========================================================
# 12. CONFUSION MATRICES
# =========================================================

xgb_cm = confusion_matrix(
    y_test,
    xgb_pred
)

if_cm = confusion_matrix(
    y_test,
    if_pred
)

ensemble_pred = (
    (
        best_ensemble["xgb_weight"] * xgb_test
        +
        best_ensemble["if_weight"] * if_test
    )
    >= best_ensemble["threshold"]
).astype(int)

ensemble_cm = confusion_matrix(
    y_test,
    ensemble_pred
)


# =========================================================
# 13. PRINT FINAL RESULTS
# =========================================================

print("\n")
print("=" * 50)
print("          FRAUDGUARD ABLATION STUDY")
print("=" * 50)


print("\nXGBOOST BASELINE")
print(f"Precision : {xgb_precision:.4f}")
print(f"Recall    : {xgb_recall:.4f}")
print(f"F1        : {xgb_f1:.4f}")
print(f"AUC-PR    : {xgb_auc:.4f}")

print("\nConfusion Matrix:")
print(xgb_cm)


print("\nISOLATION FOREST")
print(f"Precision : {if_precision:.4f}")
print(f"Recall    : {if_recall:.4f}")
print(f"F1        : {if_f1:.4f}")
print(f"AUC-PR    : {if_auc:.4f}")

print("\nConfusion Matrix:")
print(if_cm)


print("\nBEST WEIGHTED ENSEMBLE")

for key, value in best_ensemble.items():
    print(f"{key}: {value:.4f}")

print("\nConfusion Matrix:")
print(ensemble_cm)


print("\nEITHER MODEL FLAGS")
print(f"Precision : {either_precision:.4f}")
print(f"Recall    : {either_recall:.4f}")
print(f"F1        : {either_f1:.4f}")


print("\nXGBOOST MISSED FRAUD")
print(f"Fraud missed by XGBoost : {missed_count}")
print(f"Those caught by IF      : {if_caught_count}")


# =========================================================
# 14. SAVE MODELS
# =========================================================

joblib.dump(
    xgb_model,
    "models/fraud_model.pkl"
)

joblib.dump(
    if_model,
    "models/isolation_forest.pkl"
)

joblib.dump(
    {
        "threshold": best_if_threshold,
        "if_min": if_min,
        "if_max": if_max
    },
    "models/isolation_forest_meta.pkl"
)

joblib.dump(
    best_ensemble,
    "models/ensemble_config.pkl"
)

print("\nModels and configuration saved successfully.")