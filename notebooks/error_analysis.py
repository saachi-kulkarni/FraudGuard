import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# Load data
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]


# Same held-out test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# Load model
model = joblib.load("models/fraud_model.pkl")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= 0.80
).astype(int)


# Confusion matrix
tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions
).ravel()


precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)


# Simple business costs
# Missing fraud is more expensive than reviewing a normal transaction.
cost_fp = 1
cost_fn = 10

total_cost = (
    fp * cost_fp +
    fn * cost_fn
)


print("\n======================================")
print("          ERROR ANALYSIS")
print("======================================")

print(f"\nTrue Negatives  : {tn}")
print(f"False Positives : {fp}")
print(f"False Negatives : {fn}")
print(f"True Positives  : {tp}")

print("\n========== METRICS ==========")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1        : {f1:.4f}")

print("\n========== BUSINESS COST ==========")
print(f"False Positive Cost : {cost_fp}")
print(f"False Negative Cost : {cost_fn}")
print(f"Total Cost          : {total_cost}")


# Save confusion matrix
os.makedirs("models", exist_ok=True)

cm = confusion_matrix(
    y_test,
    predictions
)

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("FraudGuard Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["Normal", "Fraud"]
)

plt.yticks(
    [0, 1],
    ["Normal", "Fraud"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "models/confusion_matrix.png",
    dpi=150
)

plt.close()

print("\nSaved: models/confusion_matrix.png")