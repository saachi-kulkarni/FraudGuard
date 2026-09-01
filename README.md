# 🛡️ FraudGuard

A credit card fraud detection project built using machine learning.

I built this project to explore how machine learning can be used to detect fraudulent transactions in a highly imbalanced dataset. The project compares a Logistic Regression baseline with an XGBoost model and includes an API and web interface for making predictions.

## What it does

- Detects whether a credit card transaction is likely to be fraudulent
- Uses SMOTE to handle the imbalanced training data
- Compares Logistic Regression and XGBoost
- Uses Precision, Recall, F1-score and AUC-PR for evaluation
- Tests different classification thresholds
- Uses SHAP to show which features influenced a prediction
- Provides a REST API using FastAPI
- Provides a simple interface using Streamlit
- Runs the API inside Docker

## Model Results

### Logistic Regression + SMOTE

| Metric | Result |
|---|---:|
| Precision | 5.78% |
| Recall | 91.84% |
| F1 Score | 10.88% |
| AUC-PR | 72.44% |

### XGBoost + SMOTE

| Metric | Result |
|---|---:|
| Precision | 76.40% |
| Recall | 85.70% |
| F1 Score | 80.80% |
| AUC-PR | 87.90% |

XGBoost performed better than the Logistic Regression baseline, especially in balancing fraud detection with false positives.

## Dataset

The project uses the Credit Card Fraud Detection dataset.

The dataset contains:

- 284,807 transactions
- 492 fraudulent transactions
- 30 input features
- 28 PCA-transformed features (`V1`–`V28`)
- `Time` and `Amount`

Because fraud transactions are very rare, accuracy is not used as the main evaluation metric.

## How the project works

```text
Transaction
     ↓
Streamlit
     ↓
FastAPI
     ↓
XGBoost Model
     ↓
Fraud Probability
     ↓
Fraud / Normal
     ↓
SHAP Explanation# 🛡️ FraudGuard

A credit card fraud detection project built using machine learning.

I built this project to explore how machine learning can be used to detect fraudulent transactions in a highly imbalanced dataset. The project compares a Logistic Regression baseline with an XGBoost model and includes an API and web interface for making predictions.

## What it does

- Detects whether a credit card transaction is likely to be fraudulent
- Uses SMOTE to handle the imbalanced training data
- Compares Logistic Regression and XGBoost
- Uses Precision, Recall, F1-score and AUC-PR for evaluation
- Tests different classification thresholds
- Uses SHAP to show which features influenced a prediction
- Provides a REST API using FastAPI
- Provides a simple interface using Streamlit
- Runs the API inside Docker

## Model Results

### Logistic Regression + SMOTE

| Metric | Result |
|---|---:|
| Precision | 5.78% |
| Recall | 91.84% |
| F1 Score | 10.88% |
| AUC-PR | 72.44% |

### XGBoost + SMOTE

| Metric | Result |
|---|---:|
| Precision | 76.40% |
| Recall | 85.70% |
| F1 Score | 80.80% |
| AUC-PR | 87.90% |

XGBoost performed better than the Logistic Regression baseline, especially in balancing fraud detection with false positives.

## Dataset

The project uses the Credit Card Fraud Detection dataset.

The dataset contains:

- 284,807 transactions
- 492 fraudulent transactions
- 30 input features
- 28 PCA-transformed features (`V1`–`V28`)
- `Time` and `Amount`

Because fraud transactions are very rare, accuracy is not used as the main evaluation metric.

## How the project works

```text
Transaction
     ↓
Streamlit
     ↓
FastAPI
     ↓
XGBoost Model
     ↓
Fraud Probability
     ↓
Fraud / Normal
     ↓
SHAP Explanation