import streamlit as st
import requests
import json
import pandas as pd
import joblib
import shap

st.set_page_config(
    page_title="FraudGuard",
    page_icon="🛡️",
    layout="centered"
)

# -----------------------------
# Load ML Model
# -----------------------------

model = joblib.load("models/fraud_model.pkl")

# SHAP explainer
explainer = shap.TreeExplainer(model)


# -----------------------------
# Page Header
# -----------------------------

st.title("🛡️ FraudGuard")
st.subheader("AI-Powered Credit Card Fraud Detection")

st.write(
    "Analyze a credit card transaction using machine learning "
    "and estimate the probability of fraud."
)

st.markdown("---")


# -----------------------------
# Session State
# -----------------------------

if "features" not in st.session_state:

    st.session_state.features = {
        "Time": 0.0,
        "Amount": 149.62
    }

    for i in range(1, 29):
        st.session_state.features[f"V{i}"] = 0.0


# -----------------------------
# Sample Transactions
# -----------------------------

st.markdown("### 🧪 Try a Sample Transaction")

col1, col2 = st.columns(2)

with col1:

    if st.button(
        "🚨 Load Fraud Sample",
        use_container_width=True
    ):

        try:

            with open("data/fraud_test.json", "r") as f:
                fraud_data = json.load(f)

            st.session_state.features = fraud_data

            st.success("Fraud sample loaded!")

        except Exception:

            st.error("Could not load fraud sample.")


with col2:

    if st.button(
        "✅ Load Normal Sample",
        use_container_width=True
    ):

        st.session_state.features = {
            "Time": 0.0,
            "Amount": 149.62
        }

        for i in range(1, 29):
            st.session_state.features[f"V{i}"] = 0.0

        st.success("Normal sample loaded!")


# -----------------------------
# Transaction Details
# -----------------------------

st.markdown("### 💳 Transaction Details")

st.session_state.features["Time"] = st.number_input(
    "Transaction Time",
    value=float(st.session_state.features["Time"])
)

st.session_state.features["Amount"] = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=float(st.session_state.features["Amount"])
)


# -----------------------------
# Advanced Features
# -----------------------------

with st.expander("🔬 Advanced Transaction Features (V1–V28)"):

    st.caption(
        "These are PCA-transformed features from the original "
        "credit card transaction dataset."
    )

    for i in range(1, 29):

        feature = f"V{i}"

        st.session_state.features[feature] = st.number_input(
            feature,
            value=float(st.session_state.features[feature]),
            format="%.6f"
        )


# -----------------------------
# Prediction
# -----------------------------

st.markdown("---")

if st.button(
    "🔍 Check Transaction",
    use_container_width=True,
    type="primary"
):

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=st.session_state.features,
            timeout=10
        )

        if response.status_code == 200:

            result = response.json()

            probability = result["fraud_probability"] * 100

            st.markdown("### 📊 Fraud Detection Result")

            if result["prediction"] == 1:

                st.error(
                    "🚨 FRAUDULENT TRANSACTION DETECTED"
                )

                st.metric(
                    "Fraud Probability",
                    f"{probability:.2f}%"
                )

            else:

                st.success(
                    "✅ TRANSACTION APPEARS NORMAL"
                )

                st.metric(
                    "Fraud Probability",
                    f"{probability:.2f}%"
                )


            # -----------------------------
            # SHAP Explanation
            # -----------------------------

            st.markdown("---")
            st.markdown("### 🔎 Why did the model make this decision?")

            input_data = pd.DataFrame(
                [st.session_state.features]
            )

            shap_result = explainer(input_data)

            shap_values = shap_result.values[0]

            # Handle binary-output SHAP format if present
            if len(shap_values.shape) > 1:
                shap_values = shap_values[:, 1]

            explanation = pd.DataFrame({
                "Feature": input_data.columns,
                "Impact": shap_values
            })

            explanation["Absolute Impact"] = (
                explanation["Impact"].abs()
            )

            explanation = explanation.sort_values(
                "Absolute Impact",
                ascending=False
            ).head(5)

            st.write(
                "Top 5 features influencing this prediction:"
            )

            for _, row in explanation.iterrows():

                feature = row["Feature"]
                impact = row["Impact"]

                if impact > 0:

                    st.write(
                        f"🔴 **{feature}** → increased fraud risk"
                    )

                else:

                    st.write(
                        f"🟢 **{feature}** → decreased fraud risk"
                    )


        else:

            st.error(
                "API returned an error. "
                "Please check the Docker container."
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to FraudGuard API. "
            "Make sure the Docker container is running."
        )

    except Exception as e:

        st.error(
            f"SHAP explanation error: {e}"
        )


# -----------------------------
# Footer
# -----------------------------

st.markdown("---")

st.caption(
    "FraudGuard • XGBoost + SMOTE • SHAP • FastAPI • Docker"
)