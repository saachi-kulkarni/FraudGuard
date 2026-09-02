from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import sys

sys.path.append("src")
from agents import run_agents


app = FastAPI(
    title="FraudGuard API",
    description="Real-Time Fraud Detection and Investigation API",
    version="2.0"
)


# Load models
xgb_model = joblib.load("models/fraud_model.pkl")
if_model = joblib.load("models/isolation_forest.pkl")
if_meta = joblib.load("models/isolation_forest_meta.pkl")
ensemble_config = joblib.load("models/ensemble_config.pkl")

explainer = shap.TreeExplainer(xgb_model)


# Ensemble configuration
XGB_WEIGHT = ensemble_config["xgb_weight"]
IF_WEIGHT = ensemble_config["if_weight"]
ENSEMBLE_THRESHOLD = ensemble_config["threshold"]

IF_MIN = if_meta["if_min"]
IF_MAX = if_meta["if_max"]


class Transaction(BaseModel):

    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


@app.get("/")
def home():

    return {
        "message": "FraudGuard API is running!"
    }


@app.post("/predict")
def predict(transaction: Transaction):

    data = pd.DataFrame(
        [transaction.model_dump()]
    )


    # -----------------------------
    # XGBoost
    # -----------------------------

    xgb_score = xgb_model.predict_proba(data)[0][1]


    # -----------------------------
    # Isolation Forest
    # -----------------------------

    raw_score = -if_model.decision_function(data)[0]

    if_score = (
        (raw_score - IF_MIN)
        / (IF_MAX - IF_MIN)
    )

    if_score = max(
        0,
        min(1, if_score)
    )


    # -----------------------------
    # Combined Risk Score
    # -----------------------------

    risk_score = (
        XGB_WEIGHT * xgb_score
        + IF_WEIGHT * if_score
    )

    flagged = (
        risk_score >= ENSEMBLE_THRESHOLD
    )


    # -----------------------------
    # SHAP Explanation
    # -----------------------------

    shap_result = explainer(data)
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


    reasons = []

    for _, row in explanation.iterrows():

        direction = (
            "increased fraud risk"
            if row["impact"] > 0
            else "decreased fraud risk"
        )

        reasons.append({
            "feature": row["feature"],
            "impact": round(
                float(row["impact"]),
                4
            ),
            "reason": direction
        })


    # -----------------------------
    # Multi-Agent Investigation
    # -----------------------------

    agents = None

    if flagged:

        agents = run_agents({
            "xgb_score": xgb_score,
            "if_score": if_score,
            "risk_score": risk_score,
            "shap_reasons": reasons
        })


    # -----------------------------
    # API Response
    # -----------------------------

    return {

        "prediction": int(flagged),

        "result": (
            "Fraud"
            if flagged
            else "Normal"
        ),

        "xgb_score": round(
            float(xgb_score),
            4
        ),

        "if_score": round(
            float(if_score),
            4
        ),

        "risk_score": round(
            float(risk_score),
            4
        ),

        "threshold": ENSEMBLE_THRESHOLD,

        "top_shap_reasons": reasons,

        "agents": agents
    }