import json
import os
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="FraudGuard",
    page_icon="🛡️",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"

FEATURES = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]

st.title("🛡️ FraudGuard")
st.subheader("Real-Time Fraud Detection & Investigation")

st.write(
    "XGBoost + Isolation Forest → Risk Score → "
    "Investigator Agent → Decision Agent"
)

st.divider()

# ---------------------------------------------------------
# TRANSACTION INPUT
# ---------------------------------------------------------

st.header("💳 Transaction")

choice = st.radio(
    "Choose transaction",
    ["Upload JSON", "Sample Normal", "Sample Fraud"],
    horizontal=False
)

transaction = None

if choice == "Upload JSON":

    uploaded = st.file_uploader(
        "Upload transaction JSON",
        type=["json"]
    )

    if uploaded:
        try:
            transaction = json.load(uploaded)

            missing = [
                feature
                for feature in FEATURES
                if feature not in transaction
            ]

            if missing:
                st.error(
                    "Missing features: "
                    + ", ".join(missing)
                )
                transaction = None
            else:
                st.success("Transaction JSON loaded.")

        except Exception as e:
            st.error(f"Invalid JSON: {e}")

elif choice == "Sample Normal":

    try:
        with open("data/fraud_test.json") as f:
            samples = json.load(f)

        if isinstance(samples, list):
            normal = next(
                (x for x in samples if x.get("Class", 0) == 0),
                samples[0]
            )
        else:
            normal = samples

        transaction = {
            feature: float(normal[feature])
            for feature in FEATURES
        }

        st.success("Sample Normal transaction loaded.")

    except Exception as e:
        st.error(f"Could not load sample: {e}")

elif choice == "Sample Fraud":

    try:
        with open("data/fraud_test.json") as f:
            samples = json.load(f)

        if isinstance(samples, list):
            fraud = next(
                (x for x in samples if x.get("Class", 0) == 1),
                samples[0]
            )
        else:
            fraud = samples

        transaction = {
            feature: float(fraud[feature])
            for feature in FEATURES
        }

        st.success("Sample Fraud transaction loaded.")

    except Exception as e:
        st.error(f"Could not load sample: {e}")


# ---------------------------------------------------------
# MANUAL INPUT
# ---------------------------------------------------------

if transaction is None:

    st.info(
        "Upload a JSON transaction or enter values manually below."
    )

    transaction = {}

    col1, col2 = st.columns(2)

    with col1:

        transaction["Time"] = st.number_input(
            "Time",
            value=0.0
        )

        transaction["Amount"] = st.number_input(
            "Amount",
            value=100.0
        )

    with col2:

        st.write("V1–V28")

        for feature in FEATURES[1:-1]:

            transaction[feature] = st.number_input(
                feature,
                value=0.0
            )


else:

    st.write(
        f"**Amount:** {transaction['Amount']:.2f}"
    )


# ---------------------------------------------------------
# ANALYZE
# ---------------------------------------------------------

if st.button(
    "🔍 Analyze Transaction",
    type="primary"
):

    try:

        response = requests.post(
            API_URL,
            json=transaction,
            timeout=30
        )

        if response.status_code != 200:

            st.error(
                f"API Error: {response.status_code}"
            )

        else:

            result = response.json()

            st.divider()

            # -------------------------------------------------
            # DETECTION RESULT
            # -------------------------------------------------

            st.header("📊 Detection Result")

            col1, col2, col3, col4 = st.columns(4)

            xgb_score = result.get(
                "xgb_probability",
                result.get("fraud_probability", 0)
            )

            anomaly_score = result.get(
                "anomaly_score",
                result.get("if_score", 0)
            )

            risk_score = result.get(
                "risk_score",
                result.get("fraud_probability", 0)
            )

            threshold = result.get(
                "threshold",
                0.80
            )

            with col1:
                st.metric(
                    "XGBoost",
                    f"{xgb_score * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Anomaly Score",
                    f"{anomaly_score * 100:.2f}%"
                )

            with col3:
                st.metric(
                    "Risk Score",
                    f"{risk_score * 100:.2f}%"
                )

            with col4:
                st.metric(
                    "Threshold",
                    f"{threshold * 100:.0f}%"
                )

            prediction = result.get(
                "prediction",
                1 if risk_score >= threshold else 0
            )

            if prediction == 1:

                st.error("🚨 FRAUD DETECTED")

            else:

                st.success("✅ TRANSACTION ALLOWED")

            # -------------------------------------------------
            # SHAP EXPLANATION
            # -------------------------------------------------

            st.header("🔎 Model Explanation")

            reasons = result.get(
                "top_shap_reasons",
                []
            )

            if reasons:

                for reason in reasons:

                    feature = reason.get(
                        "feature",
                        "Unknown"
                    )

                    impact = reason.get(
                        "impact",
                        0
                    )

                    explanation = reason.get(
                        "reason",
                        ""
                    )

                    st.write(
                        f"**{feature}** — "
                        f"{explanation} "
                        f"(impact: {impact:.4f})"
                    )

            # -------------------------------------------------
            # MULTI-AGENT INVESTIGATION
            # -------------------------------------------------

            st.header("🤖 Multi-Agent Investigation")

            investigator = result.get(
                "investigator_explanation",
                result.get(
                    "investigator",
                    ""
                )
            )

            decision = result.get(
                "decision",
                result.get(
                    "decision_agent",
                    ""
                )
            )

            st.info(
                f"🕵️ **Investigator Agent**\n\n"
                f"{investigator if investigator else 'No investigation explanation returned.'}"
            )

            if decision:

                if "AUTO-BLOCK" in decision:

                    st.error(
                        f"🛑 **Decision Agent: {decision}**"
                    )

                elif "REVIEW" in decision:

                    st.warning(
                        f"⚠️ **Decision Agent: {decision}**"
                    )

                else:

                    st.success(
                        f"✅ **Decision Agent: {decision}**"
                    )

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Cannot connect to FraudGuard API. "
            "Make sure FastAPI is running on port 8000."
        )

    except Exception as e:

        st.error(
            f"Something went wrong: {e}"
        )


# ---------------------------------------------------------
# THROUGHPUT BENCHMARK
# ---------------------------------------------------------

st.divider()

st.header("⚡ Real-Time Throughput Benchmark")

st.write(
    "FraudGuard compares sequential scoring with "
    "concurrent transaction processing."
)

benchmark_path = "models/throughput_benchmark.png"

if os.path.exists(benchmark_path):

    st.image(
        benchmark_path,
        caption="Sequential vs Concurrent Transaction Throughput"
    )

else:

    st.warning(
        "Throughput benchmark image not found."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "FraudGuard • XGBoost • Isolation Forest • SHAP • "
    "FastAPI • Multi-Agent Decision Layer • Concurrent Processing"
)