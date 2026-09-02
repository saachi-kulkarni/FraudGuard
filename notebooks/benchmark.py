import sys
sys.path.append("src")

import matplotlib.pyplot as plt
from scoring_engine import transaction_stream, run_sequential, run_concurrent

transactions = list(transaction_stream(n=500, delay=(0, 0)))

_, sequential_time = run_sequential(transactions)
_, concurrent_time = run_concurrent(transactions, workers=4)

sequential_tps = len(transactions) / sequential_time
concurrent_tps = len(transactions) / concurrent_time
speedup = concurrent_tps / sequential_tps

print("\n========== THROUGHPUT BENCHMARK ==========")
print(f"Transactions   : {len(transactions)}")
print(f"Sequential     : {sequential_tps:.2f} transactions/sec")
print(f"Concurrent     : {concurrent_tps:.2f} transactions/sec")
print(f"Speedup        : {speedup:.2f}x")

plt.figure(figsize=(8, 5))

bars = plt.bar(
    ["Sequential", "Concurrent"],
    [sequential_tps, concurrent_tps]
)

plt.ylabel("Transactions / Second")
plt.title("FraudGuard Throughput Benchmark")

for bar, value in zip(bars, [sequential_tps, concurrent_tps]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("models/throughput_benchmark.png", dpi=150)
plt.close()

print("\nSaved: models/throughput_benchmark.png")