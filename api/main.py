from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# Create FastAPI app
app = FastAPI(
    title="FraudGuard API",
    description="Credit Card Fraud Detection API",
    version="1.0"
)


# Load trained model
model = joblib.load("models/fraud_model.pkl")


# Input format
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


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "FraudGuard API is running!"
    }


# Fraud prediction endpoint
@app.post("/predict")
def predict(transaction: Transaction):

    data = pd.DataFrame([transaction.model_dump()])

    probability = model.predict_proba(data)[0][1]

    prediction = 1 if probability >= 0.50 else 0

    if prediction == 1:
        result = "Fraud"
    else:
        result = "Normal"

    return {
        "prediction": prediction,
        "result": result,
        "fraud_probability": round(float(probability), 4)
    }