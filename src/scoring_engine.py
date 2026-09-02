import time
import joblib
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from transaction_stream import transaction_stream
from agents import run_agents


xgb_model = joblib.load("models/fraud_model.pkl")
if_model = joblib.load("models/isolation_forest.pkl")
if_meta = joblib.load("models/isolation_forest_meta.pkl")
ensemble_config = joblib.load("models/ensemble_config.pkl")

XGB_THRESHOLD = 0.80
IF_MIN = if_meta["if_min"]
IF_MAX = if_meta["if_max"]

XGB_WEIGHT = ensemble_config["xgb_weight"]
IF_WEIGHT = ensemble_config["if_weight"]
ENSEMBLE_THRESHOLD = ensemble_config["threshold"]


def score_transaction(item):

    transaction, actual_label = item
    data = pd.DataFrame([transaction])

    xgb_score = xgb_model.predict_proba(data)[0][1]

    raw_score = -if_model.decision_function(data)[0]

    if_score = (raw_score - IF_MIN) / (IF_MAX - IF_MIN)
    if_score = max(0, min(1, if_score))

    risk_score = (
        XGB_WEIGHT * xgb_score +
        IF_WEIGHT * if_score
    )

    flagged = risk_score >= ENSEMBLE_THRESHOLD

    result = {
        "transaction": transaction,
        "actual": actual_label,
        "xgb_score": xgb_score,
        "if_score": if_score,
        "risk_score": risk_score,
        "flagged": flagged
    }

    if flagged:
        result["agents"] = run_agents(result)

    return result


def run_sequential(transactions):

    start = time.perf_counter()

    results = [
        score_transaction(item)
        for item in transactions
    ]

    return results, time.perf_counter() - start


def run_concurrent(transactions, workers=4):

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(score_transaction, transactions)
        )

    return results, time.perf_counter() - start


if __name__ == "__main__":

    print("\n======================================")
    print("       FRAUDGUARD SCORING ENGINE")
    print("======================================")

    transactions = list(
        transaction_stream(
            n=100,
            delay=(0, 0)
        )
    )

    sequential_results, sequential_time = run_sequential(
        transactions
    )

    concurrent_results, concurrent_time = run_concurrent(
        transactions
    )

    sequential_tps = 100 / sequential_time
    concurrent_tps = 100 / concurrent_time

    flagged = [
        r for r in concurrent_results
        if r["flagged"]
    ]

    print(f"\nTransactions: {len(transactions)}")

    print("\nSEQUENTIAL")
    print(f"Time       : {sequential_time:.4f} sec")
    print(f"Throughput : {sequential_tps:.2f} transactions/sec")

    print("\nCONCURRENT")
    print(f"Time       : {concurrent_time:.4f} sec")
    print(f"Throughput : {concurrent_tps:.2f} transactions/sec")

    print("\nFLAGGED TRANSACTIONS")
    print(f"Flagged: {len(flagged)}")

    for result in flagged[:5]:

        print("\n------------------------------")

        print(
            f"Risk Score : {result['risk_score']:.3f}"
        )

        print(
            f"XGBoost    : {result['xgb_score']:.3f}"
        )

        print(
            f"Isolation  : {result['if_score']:.3f}"
        )

        print(
            f"Actual     : "
            f"{'FRAUD' if result['actual'] else 'NORMAL'}"
        )

        print(
            f"Decision   : "
            f"{result['agents']['decision']}"
        )

        print(
            f"Explanation: "
            f"{result['agents']['explanation']}"
        )

    print("\nFraudGuard scoring complete.")