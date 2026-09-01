import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# =========================
# 1. Load dataset
# =========================

df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]


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
# 3. Apply SMOTE
# =========================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


# =========================
# 4. Train final XGBoost
# =========================

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(
    X_train_smote,
    y_train_smote
)


# =========================
# 5. Predictions
# =========================

probabilities = model.predict_proba(X_test)[:, 1]

THRESHOLD = 0.80

predictions = (
    probabilities >= THRESHOLD
).astype(int)


# =========================
# 6. Confusion Matrix
# =========================

cm = confusion_matrix(
    y_test,
    predictions
)

print("Confusion Matrix:")
print(cm)

print("\nThreshold:", THRESHOLD)

print(
    "\nAUC-PR:",
    round(
        average_precision_score(
            y_test,
            probabilities
        ),
        4
    )
)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Fraud"],
    yticklabels=["Normal", "Fraud"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("FraudGuard Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "models/confusion_matrix.png",
    dpi=300
)

plt.close()


# =========================
# 7. Precision-Recall Curve
# =========================

precision, recall, thresholds = precision_recall_curve(
    y_test,
    probabilities
)

auc_pr = average_precision_score(
    y_test,
    probabilities
)


plt.figure(figsize=(7, 5))

plt.plot(
    recall,
    precision,
    label=f"AUC-PR = {auc_pr:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "FraudGuard Precision-Recall Curve"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "models/precision_recall_curve.png",
    dpi=300
)

plt.close()


print(
    "\nEvaluation graphs saved successfully!"
)

print(
    "models/confusion_matrix.png"
)

print(
    "models/precision_recall_curve.png"
)