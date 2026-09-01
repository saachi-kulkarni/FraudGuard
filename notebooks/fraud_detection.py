import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/creditcard.csv")

print("Dataset shape:", df.shape)

X = df.drop("Class", axis=1)
y = df["Class"]


# ==========================================
# 2. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nBefore SMOTE:")
print("Normal:", sum(y_train == 0))
print("Fraud:", sum(y_train == 1))


# ==========================================
# 3. HANDLE CLASS IMBALANCE
# ==========================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print("Normal:", sum(y_train_smote == 0))
print("Fraud:", sum(y_train_smote == 1))


# ==========================================
# 4. TRAIN XGBOOST
# ==========================================

print("\nTraining XGBoost...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train_smote, y_train_smote)

print("Training complete!")


# ==========================================
# 5. GET FRAUD PROBABILITIES
# ==========================================

y_probability = model.predict_proba(X_test)[:, 1]

# Default threshold
threshold = 0.50

y_pred = (y_probability >= threshold).astype(int)


# ==========================================
# 6. MODEL RESULTS
# ==========================================

print("\n==========================================")
print("           FRAUDGUARD RESULTS")
print("==========================================")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_pr = average_precision_score(y_test, y_probability)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC-PR:", auc_pr)


# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("Fraud Detection Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0, 1], ["Normal", "Fraud"])
plt.yticks([0, 1], ["Normal", "Fraud"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 va="center",
                 fontsize=16)

plt.colorbar()
plt.tight_layout()

plt.savefig("models/confusion_matrix.png")
plt.show()


# ==========================================
# 8. PRECISION-RECALL CURVE
# ==========================================

precision_curve, recall_curve, thresholds = precision_recall_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(recall_curve, precision_curve)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.grid(True)
plt.tight_layout()

plt.savefig("models/precision_recall_curve.png")
plt.show()


# ==========================================
# 9. THRESHOLD ANALYSIS
# ==========================================

print("\n==========================================")
print("         THRESHOLD ANALYSIS")
print("==========================================")

thresholds_to_test = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

print("\nThreshold | Precision | Recall | F1")
print("-------------------------------------")

for t in thresholds_to_test:

    predictions = (y_probability >= t).astype(int)

    p = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    r = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print(
        f"{t:9.2f} | "
        f"{p:9.3f} | "
        f"{r:6.3f} | "
        f"{f:5.3f}"
    )


# ==========================================
# 10. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/fraud_model.pkl"
)

print("\nModel saved to:")
print("models/fraud_model.pkl")

print("\n==========================================")
print("             PROJECT STEP DONE")
print("==========================================")