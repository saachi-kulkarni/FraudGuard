import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix


# 1. Load data
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

print("Dataset:", X.shape)
print("Fraud:", y.sum())
print("Normal:", (y == 0).sum())


# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Create validation set from training data
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.20,
    stratify=y_train,
    random_state=42
)


# 3. Train only on normal transactions
X_normal = X_train[y_train == 0]

model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

model.fit(X_normal)


# 4. Get anomaly scores
# Higher score = more anomalous
val_scores = -model.decision_function(X_val)
test_scores = -model.decision_function(X_test)


# 5. Find best threshold using validation data
best_threshold = 0
best_f1 = 0

for threshold in sorted(val_scores):
    predictions = (val_scores >= threshold).astype(int)
    f1 = f1_score(y_val, predictions, zero_division=0)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


# 6. Evaluate on unseen test data
predictions = (test_scores >= best_threshold).astype(int)

precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
auc_pr = average_precision_score(y_test, test_scores)

cm = confusion_matrix(y_test, predictions)


# 7. Print results
print("\n========== ISOLATION FOREST RESULTS ==========")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"AUC-PR    : {auc_pr:.4f}")

print("\nConfusion Matrix:")
print(cm)

print(f"\nSelected threshold: {best_threshold:.4f}")


# 8. Save model + threshold
joblib.dump(model, "models/isolation_forest.pkl")

joblib.dump(
    {"threshold": best_threshold},
    "models/isolation_forest_meta.pkl"
)

print("\nSaved:")
print("models/isolation_forest.pkl")
print("models/isolation_forest_meta.pkl")