import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

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

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

print(f"\nScale Pos Weight: {scale_pos_weight:.2f}")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= 0.80
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
    probabilities
)

print("\n========== SCALE_POS_WEIGHT RESULTS ==========")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1        : {f1:.4f}")
print(f"AUC-PR    : {auc_pr:.4f}")

print("\n========== COMPARISON ==========")
print("SMOTE XGBoost AUC-PR : 0.8794")
print(f"Weighted XGBoost     : {auc_pr:.4f}")