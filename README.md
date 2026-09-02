# FraudGuard

FraudGuard is a fraud detection project that uses machine learning to identify suspicious credit card transactions.

The project combines XGBoost for supervised fraud detection with Isolation Forest for anomaly detection. Flagged transactions are then explained using SHAP and passed through a simple investigation and decision layer.

## What the project does

- Detects fraudulent transactions using XGBoost
- Detects unusual transactions using Isolation Forest
- Combines both models into a risk score
- Handles class imbalance using SMOTE
- Compares SMOTE with `scale_pos_weight`
- Uses SHAP to explain predictions
- Simulates a stream of incoming transactions
- Supports concurrent transaction scoring
- Provides an Investigator Agent and Decision Agent
- Has a deterministic fallback when an external LLM is unavailable
- Provides a FastAPI REST API
- Provides a Streamlit dashboard
- Includes Docker support

---

## Project Architecture

```text
Transaction
     |
     v
Transaction Stream
     |
     v
+-----------------------+
|   Detection Layer     |
|                       |
|  XGBoost              |
|  Isolation Forest     |
+----------+------------+
           |
           v
     Risk Score
           |
           v
     Is Fraudulent?
       /        \
     No          Yes
     |            |
     v            v
   Allow       SHAP
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