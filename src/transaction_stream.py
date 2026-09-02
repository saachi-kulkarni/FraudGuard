import pandas as pd
import random
import time


FEATURES = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]


def load_transactions():
    df = pd.read_csv("data/creditcard.csv")

    return df


def transaction_stream(n=20, delay=(0.05, 0.15)):
    df = load_transactions()

    samples = df.sample(
        n=n,
        random_state=42
    )

    for _, row in samples.iterrows():

        transaction = {
            feature: float(row[feature])
            for feature in FEATURES
        }

        label = int(row["Class"])

        yield transaction, label

        time.sleep(
            random.uniform(
                delay[0],
                delay[1]
            )
        )


if __name__ == "__main__":

    print("Starting FraudGuard transaction stream...\n")

    for i, (transaction, label) in enumerate(
        transaction_stream(10)
    ):

        print(
            f"Transaction {i + 1}: "
            f"Amount={transaction['Amount']:.2f} "
            f"Actual={'FRAUD' if label else 'NORMAL'}"
        )

    print("\nStream finished.")