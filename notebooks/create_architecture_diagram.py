import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 12))

ax.set_xlim(0, 14)
ax.set_ylim(-1, 12)
ax.axis("off")


def add_box(x, y, width, height, title, lines, fontsize=11):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.15",
        linewidth=1.5,
        edgecolor="black",
        facecolor="white"
    )

    ax.add_patch(patch)

    ax.text(
        x + width / 2,
        y + height - 0.35,
        title,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold"
    )

    ax.text(
        x + 0.25,
        y + height - 0.8,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=fontsize,
        linespacing=1.5
    )


def add_arrow(x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=1.5
        )
    )


ax.text(
    7,
    11.5,
    "FraudGuard – System Architecture",
    ha="center",
    fontsize=22,
    fontweight="bold"
)

ax.text(
    7,
    11.05,
    "Real-Time Transaction Risk Detection for Payment Platforms",
    ha="center",
    fontsize=13
)


add_box(
    4.5,
    9.35,
    5,
    1.25,
    "1. Transaction Stream",
    [
        "Incoming transactions",
        "Simulated real-time stream"
    ]
)

add_box(
    4.5,
    7.55,
    5,
    1.25,
    "2. Concurrent Scoring",
    [
        "ThreadPoolExecutor",
        "Concurrent transaction processing"
    ]
)

add_arrow(7, 9.35, 7, 8.8)


ax.text(
    7,
    7.05,
    "3. Ensemble Detection",
    ha="center",
    fontsize=17,
    fontweight="bold"
)

add_box(
    1.0,
    5.1,
    5.2,
    1.55,
    "XGBoost",
    [
        "Supervised learning",
        "Known fraud patterns",
        "scale_pos_weight"
    ],
    fontsize=10
)

add_box(
    7.8,
    5.1,
    5.2,
    1.55,
    "Isolation Forest",
    [
        "Unsupervised anomaly detection",
        "Unusual transaction patterns",
        "Complementary signal"
    ],
    fontsize=10
)

add_box(
    4,
    3.35,
    6,
    1.15,
    "Combined Risk Score",
    [
        "Combines XGBoost and anomaly signals"
    ],
    fontsize=11
)

add_arrow(3.6, 5.1, 5.3, 4.5)
add_arrow(10.4, 5.1, 8.7, 4.5)


add_box(
    4,
    1.75,
    6,
    1.15,
    "4. SHAP Explanation",
    [
        "Top contributing features",
        "Increased / decreased risk"
    ],
    fontsize=10
)

add_arrow(7, 3.35, 7, 2.9)


add_box(
    0.8,
    0.0,
    5.8,
    1.3,
    "5. Investigator Agent",
    [
        "Risk score + SHAP signals",
        "Deterministic investigation summary",
        "LLM-ready interface"
    ],
    fontsize=9.5
)

add_box(
    7.4,
    0.0,
    5.8,
    1.3,
    "6. Decision Agent",
    [
        "Explicit risk policies",
        "Deterministic decision logic",
        "ALLOW / REVIEW / AUTO-BLOCK"
    ],
    fontsize=9.5
)

add_arrow(7, 1.75, 3.7, 1.3)
add_arrow(7, 1.75, 10.3, 1.3)


ax.text(
    7,
    -0.7,
    "FastAPI   |   Streamlit   |   Docker",
    ha="center",
    fontsize=13,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "models/architecture.png",
    dpi=200,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()

print("Architecture diagram created successfully.")
print("Saved: models/architecture.png")
