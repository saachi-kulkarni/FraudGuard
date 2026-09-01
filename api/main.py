from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import shap


# --------------------------------
# FastAPI application
# --------------------------------

app = FastAPI(
    title="FraudGuard API",
    description="Credit Card Fraud Detection API",
    version="1.0"
)


# --------------------------------
# Load trained model
# --------------------------------

model = joblib.load("models/fraud_model.pkl")

explainer = shap.TreeExplainer(model)


# --------------------------------
# Operating threshold
# --------------------------------

THRESHOLD = 0.80


# --------------------------------
# Transaction input
# --------------------------------

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


# --------------------------------
# Home endpoint
# --------------------------------

@app.get("/")
def home():
    return {
        "message": "FraudGuard API is running!"
    }


# --------------------------------
# Prediction endpoint
# --------------------------------

@app.post("/predict")
def predict(transaction: Transaction):

    # Convert input into DataFrame
    data = pd.DataFrame(
        [transaction.model_dump()]
    )

    # Get fraud probability
    probability = model.predict_proba(data)[0][1]

    # Apply selected operating threshold
    prediction = 1 if probability >= THRESHOLD else 0

    result = "Fraud" if prediction == 1 else "Normal"


    # --------------------------------
    # SHAP explanation
    # --------------------------------

    shap_result = explainer(data)

    shap_values = shap_result.values[0]

    # Handle possible multi-output SHAP format
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


    # Create simple explanations
    reasons = []

    for _, row in explanation.iterrows():

        if row["impact"] > 0:
            direction = "increased fraud risk"
        else:
            direction = "decreased fraud risk"

        reasons.append({
            "feature": row["feature"],
            "impact": round(float(row["impact"]), 4),
            "reason": direction
        })


    # --------------------------------
    # API response
    # --------------------------------

    return {
        "prediction": prediction,
        "result": result,
        "fraud_probability": round(
            float(probability), 4
        ),
        "threshold": THRESHOLD,
        "top_shap_reasons": reasons
    }