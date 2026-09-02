import sys
import time

import joblib
import pandas as pd
import shap

from concurrent.futures import ThreadPoolExecutor

sys.path.append("src")

from transaction_stream import transaction_stream
from agents import run_agents


# =========================================================
# LOAD MODELS
# =========================================================

xgb_model = joblib.load(
    "models/fraud_model.pkl"
)

if_model = joblib.load(
    "models/isolation_forest.pkl"
)

if_meta = joblib.load(
    "models/isolation_forest_meta.pkl"
)

ensemble_config = joblib.load(
    "models/ensemble_config.pkl"
)


explainer = shap.TreeExplainer(
    xgb_model
)


# =========================================================
# CONFIGURATION
# =========================================================

IF_MIN = if_meta["if_min"]
IF_MAX = if_meta["if_max"]

XGB_WEIGHT = ensemble_config["xgb_weight"]
IF_WEIGHT = ensemble_config["if_weight"]

ENSEMBLE_THRESHOLD = ensemble_config["threshold"]


# =========================================================
# SCORE ONE TRANSACTION
# =========================================================

def score_transaction(item):

    transaction, actual_label = item

    data = pd.DataFrame(
        [transaction]
    )


    # -----------------------------------------------------
    # XGBOOST
    # -----------------------------------------------------

    xgb_score = xgb_model.predict_proba(
        data
    )[0][1]


    # -----------------------------------------------------
    # ISOLATION FOREST
    # -----------------------------------------------------

    raw_score = -if_model.decision_function(
        data
    )[0]

    if_score = (
        (raw_score - IF_MIN)
        / (IF_MAX - IF_MIN)
    )

    if_score = max(
        0,
        min(1, if_score)
    )


    # -----------------------------------------------------
    # ENSEMBLE RISK SCORE
    # -----------------------------------------------------

    risk_score = (
        XGB_WEIGHT * xgb_score
        + IF_WEIGHT * if_score
    )

    flagged = (
        risk_score >= ENSEMBLE_THRESHOLD
    )


    # -----------------------------------------------------
    # SHAP EXPLANATION
    # -----------------------------------------------------

    shap_reasons = []

    if flagged:

        shap_result = explainer(
            data
        )

        shap_values = shap_result.values[0]

        if len(shap_values.shape) > 1:

            shap_values = shap_values[:, 1]


        explanation = pd.DataFrame({

            "feature": data.columns,

            "impact": shap_values

        })


        explanation["absolute_impact"] = (
            explanation["impact"].abs()
        )


        explanation = explanation.sort_values(
            "absolute_impact",
            ascending=False
        ).head(5)


        for _, row in explanation.iterrows():

            direction = (
                "increased fraud risk"
                if row["impact"] > 0
                else "decreased fraud risk"
            )

            shap_reasons.append({

                "feature": row["feature"],

                "impact": round(
                    float(row["impact"]),
                    4
                ),

                "reason": direction

            })


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    result = {

        "transaction": transaction,

        "actual": actual_label,

        "xgb_score": float(xgb_score),

        "if_score": float(if_score),

        "risk_score": float(risk_score),

        "flagged": flagged,

        "shap_reasons": shap_reasons

    }


    # -----------------------------------------------------
    # MULTI-AGENT LAYER
    # -----------------------------------------------------

    if flagged:

        result["agents"] = run_agents({

            "xgb_score": xgb_score,

            "if_score": if_score,

            "risk_score": risk_score,

            "shap_reasons": shap_reasons

        })


    return result


# =========================================================
# SEQUENTIAL PROCESSING
# =========================================================

def run_sequential(transactions):

    start = time.perf_counter()

    results = [

        score_transaction(item)

        for item in transactions

    ]

    elapsed = (
        time.perf_counter()
        - start
    )

    return results, elapsed


# =========================================================
# CONCURRENT PROCESSING
# =========================================================

def run_concurrent(
    transactions,
    workers=4
):

    start = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        results = list(
            executor.map(
                score_transaction,
                transactions
            )
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    return results, elapsed


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "       FRAUDGUARD SCORING ENGINE"
    )

    print(
        "======================================"
    )


    transactions = list(
        transaction_stream(
            n=100,
            delay=(0, 0)
        )
    )


    # -----------------------------------------------------
    # SEQUENTIAL
    # -----------------------------------------------------

    sequential_results, sequential_time = (
        run_sequential(transactions)
    )


    # -----------------------------------------------------
    # CONCURRENT
    # -----------------------------------------------------

    concurrent_results, concurrent_time = (
        run_concurrent(transactions)
    )


    sequential_tps = (
        len(transactions)
        / sequential_time
    )

    concurrent_tps = (
        len(transactions)
        / concurrent_time
    )


    flagged = [

        result

        for result in concurrent_results

        if result["flagged"]

    ]


    print(
        f"\nTransactions: {len(transactions)}"
    )


    print("\nSEQUENTIAL")

    print(
        f"Time       : "
        f"{sequential_time:.4f} sec"
    )

    print(
        f"Throughput : "
        f"{sequential_tps:.2f} transactions/sec"
    )


    print("\nCONCURRENT")

    print(
        f"Time       : "
        f"{concurrent_time:.4f} sec"
    )

    print(
        f"Throughput : "
        f"{concurrent_tps:.2f} transactions/sec"
    )


    print(
        "\nFLAGGED TRANSACTIONS"
    )

    print(
        f"Flagged: {len(flagged)}"
    )


    for result in flagged[:5]:

        print(
            "\n------------------------------"
        )

        print(
            f"Risk Score : "
            f"{result['risk_score']:.3f}"
        )

        print(
            f"XGBoost    : "
            f"{result['xgb_score']:.3f}"
        )

        print(
            f"Isolation  : "
            f"{result['if_score']:.3f}"
        )

        print(
            f"Actual     : "
            f"{'FRAUD' if result['actual'] else 'NORMAL'}"
        )

        print(
            "Top SHAP   : "
            + ", ".join(
                reason["feature"]
                for reason in result["shap_reasons"][:5]
            )
        )

        print(
            f"Decision   : "
            f"{result['agents']['decision']}"
        )

        print(
            f"Explanation: "
            f"{result['agents']['explanation']}"
        )


    print(
        "\nFraudGuard scoring complete."
    )