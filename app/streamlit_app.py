import streamlit as st
import requests
import json
import pandas as pd

st.set_page_config(
    page_title="FraudGuard",
    page_icon="🛡️",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000/predict"

st.title("🛡️ FraudGuard")
st.subheader("AI-Powered Credit Card Fraud Detection")

st.write(
    "Analyze a credit card transaction using machine learning "
    "and estimate its probability of being fraudulent."
)

st.markdown("---")

# -----------------------------
# Default transaction
# -----------------------------

if "features" not in st.session_state:
    st.session_state.features = {
        "Time": 0.0,
        "Amount": 149.62
    }

    for i in range(1, 29):
        st.session_state.features[f"V{i}"] = 0.0


# -----------------------------
# Sample transactions
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
                st.session_state.features = json.load(f)

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
# Upload JSON transaction
# -----------------------------

st.markdown("### 📁 Upload Transaction")

uploaded_file = st.file_uploader(
    "Upload a JSON transaction",
    type=["json"]
)

if uploaded_file is not None:

    try:
        uploaded_data = json.load(uploaded_file)

        required_features = ["Time", "Amount"] + [
            f"V{i}" for i in range(1, 29)
        ]

        if all(feature in uploaded_data for feature in required_features):

            st.session_state.features = uploaded_data

            st.success("Transaction uploaded successfully!")

        else:
            st.error(
                "Invalid JSON. It must contain Time, Amount and V1–V28."
            )

    except Exception:
        st.error("Could not read the JSON file.")


# -----------------------------
# Transaction details
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
# Advanced features
# -----------------------------

with st.expander("🔬 Advanced Features (V1–V28)"):

    st.caption(
        "V1–V28 are PCA-transformed features from the "
        "credit card fraud dataset."
    )

    for i in range(1, 29):

        feature = f"V{i}"

        st.session_state.features[feature] = st.number_input(
            feature,
            value=float(st.session_state.features[feature]),
            format="%.6f"
        )


st.markdown("---")


# -----------------------------
# Prediction
# -----------------------------

if st.button(
    "🔍 Check Transaction",
    use_container_width=True,
    type="primary"
):

    try:

        response = requests.post(
            API_URL,
            json=st.session_state.features,
            timeout=10
        )

        if response.status_code == 200:

            result = response.json()

            probability = result["fraud_probability"] * 100
            threshold = result["threshold"]

            st.markdown("### 📊 Fraud Detection Result")

            # -----------------------------
            # Result
            # -----------------------------

            if result["prediction"] == 1:

                st.error(
                    "🚨 FRAUDULENT TRANSACTION DETECTED"
                )

            else:

                st.success(
                    "✅ TRANSACTION APPEARS NORMAL"
                )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Fraud Probability",
                    f"{probability:.2f}%"
                )

            with col2:
                st.metric(
                    "Decision Threshold",
                    f"{threshold:.2f}"
                )


            st.progress(
                min(probability / 100, 1.0)
            )


            # -----------------------------
            # SHAP explanation
            # -----------------------------

            st.markdown("---")

            st.markdown(
                "### 🔎 Why did the model make this decision?"
            )

            reasons = result.get(
                "top_shap_reasons",
                []
            )

            if reasons:

                st.write(
                    "Top features influencing this prediction:"
                )

                explanation_data = []

                for reason in reasons:

                    feature = reason["feature"]
                    impact = reason["impact"]
                    direction = reason["reason"]

                    explanation_data.append({
                        "Feature": feature,
                        "SHAP Impact": impact,
                        "Effect": direction
                    })

                    if impact > 0:

                        st.write(
                            f"🔴 **{feature}** → "
                            f"increased fraud risk"
                        )

                    else:

                        st.write(
                            f"🟢 **{feature}** → "
                            f"decreased fraud risk"
                        )


                # -----------------------------
                # SHAP visual
                # -----------------------------

                st.markdown("#### SHAP Impact")

                chart_data = pd.DataFrame(
                    explanation_data
                )

                chart_data = chart_data.set_index(
                    "Feature"
                )[["SHAP Impact"]]

                st.bar_chart(chart_data)


            else:

                st.info(
                    "No SHAP explanation returned."
                )


        else:

            st.error(
                f"API error: {response.status_code}"
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to FraudGuard API. "
            "Make sure the Docker container is running."
        )


    except Exception as e:

        st.error(
            f"Something went wrong: {e}"
        )


# -----------------------------
# Footer
# -----------------------------

st.markdown("---")

st.caption(
    "FraudGuard • XGBoost + SMOTE • SHAP • FastAPI • Docker"
)