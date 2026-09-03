import matplotlib.pyplot as plt

models = [
    "XGBoost + scale_pos_weight",
    "Isolation Forest",
    "Ensemble"
]

precision = [91.01, 24.07, 92.77]
recall = [82.65, 13.27, 78.57]
f1 = [86.63, 17.11, 85.08]
auc_pr = [88.89, 12.83, 81.65]

data = [
    [
        models[i],
        f"{precision[i]:.2f}%",
        f"{recall[i]:.2f}%",
        f"{f1[i]:.2f}%",
        f"{auc_pr[i]:.2f}%"
    ]
    for i in range(len(models))
]

fig, ax = plt.subplots(figsize=(12, 4))

ax.axis("off")

table = ax.table(
    cellText=data,
    colLabels=[
        "Model",
        "Precision",
        "Recall",
        "F1",
        "AUC-PR"
    ],
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.2)

for cell in table.get_celld().values():
    cell.set_linewidth(0.5)

ax.set_title(
    "FraudGuard Ablation Study",
    fontsize=18,
    fontweight="bold",
    pad=20
)

plt.tight_layout()

plt.savefig(
    "models/ablation_results.png",
    dpi=200,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()

print("Ablation table created successfully.")
print("Saved: models/ablation_results.png")