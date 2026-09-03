# FraudGuard

FraudGuard is a real-time fraud detection system that combines supervised machine learning, anomaly detection, explainability, concurrent transaction processing, and an investigation/decision layer.

The goal is to detect known fraud patterns with XGBoost while also using Isolation Forest to identify unusual transactions that may not match previously seen fraud patterns.

---

## Features

- XGBoost fraud classification
- Class imbalance handling with `scale_pos_weight`
- Isolation Forest anomaly detection
- Ensemble risk scoring
- SHAP-based explanations
- Concurrent transaction processing
- Throughput benchmarking
- Investigator and Decision agents
- Deterministic fallback when an LLM is unavailable
- FastAPI REST API
- Streamlit dashboard
- Docker support

---

## System Architecture

```text
Incoming Transactions
        |
        v
+-------------------------+
| Ensemble Detection     |
|                         |
| XGBoost + Isolation     |
| Forest                  |
+-----------+-------------+
            |
            v
       Risk Score
            |
       +----+----+
       |         |
    Normal     Flagged
       |         |
     Log/Skip    v
             SHAP Analysis
                 |
                 v
        Investigator Agent
                 |
                 v
          Decision Agent
                 |
       +---------+---------+
       |         |         |
     ALLOW     REVIEW   AUTO-BLOCK