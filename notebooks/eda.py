import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/creditcard.csv")

print("\n========== DATASET ==========")
print(f"Rows    : {len(df)}")
print(f"Columns : {df.shape[1]}")
print(f"Fraud   : {df['Class'].sum()}")
print(f"Normal  : {(df['Class'] == 0).sum()}")
print(f"Fraud % : {df['Class'].mean() * 100:.3f}%")

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum().sum())

print("\n========== CLASS DISTRIBUTION ==========")
print(df["Class"].value_counts())

# Create models/eda if needed
import os
os.makedirs("models", exist_ok=True)

# 1. Class imbalance
plt.figure(figsize=(7, 5))
sns.countplot(x="Class", data=df)
plt.title("Fraud vs Normal Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig("models/class_distribution.png", dpi=150)
plt.close()

# 2. Amount distribution
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="Amount",
    hue="Class",
    bins=50,
    element="step"
)
plt.title("Transaction Amount Distribution")
plt.tight_layout()
plt.savefig("models/amount_distribution.png", dpi=150)
plt.close()

# 3. Time distribution
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="Time",
    hue="Class",
    bins=50,
    element="step"
)
plt.title("Transaction Time Distribution")
plt.tight_layout()
plt.savefig("models/time_distribution.png", dpi=150)
plt.close()

# 4. Correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(
    df.corr(),
    cmap="coolwarm",
    center=0
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("models/correlation_heatmap.png", dpi=150)
plt.close()

print("\n========== EDA COMPLETE ==========")
print("Saved:")
print("models/class_distribution.png")
print("models/amount_distribution.png")
print("models/time_distribution.png")
print("models/correlation_heatmap.png")