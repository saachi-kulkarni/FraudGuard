import json
import requests
import streamlit as st

st.set_page_config(
    page_title="FraudGuard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Transaction Risk Scoring Dashboard")
st.caption(
    "Real-time transaction risk detection using ensemble ML, "
    "SHAP explainability, and decision agents"
)

st.divider()

mode = st.radio(
    "Choose transaction input",
    ["Upload JSON", "Sample Normal", "Sample High-Risk"],
    horizontal=True
)

transaction = None

if mode == "Upload JSON":
    uploaded_file = st.file_uploader(
        "Upload transaction JSON",
        type=["json"]
    )

    if uploaded_file:
        try:
            transaction = json.load(uploaded_file)
            st.success("Transaction loaded successfully.")
        except Exception:
            st.error("Invalid JSON file.")

elif mode == "Sample Normal":
    transaction = {
        "Time": 406.0,
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536347,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": 0.090794,
        "V11": -0.5516,
        "V12": -0.617801,
        "V13": -0.99139,
        "V14": -0.311169,
        "V15": 1.468177,
        "V16": -0.4704,
        "V17": 0.207971,
        "V18": 0.025791,
        "V19": 0.403993,
        "V20": 0.251412,
        "V21": -0.018307,
        "V22": 0.277838,
        "V23": -0.110474,
        "V24": 0.066928,
        "V25": 0.128539,
        "V26": -0.189115,
        "V27": 0.133558,
        "V28": -0.021053,
        "Amount": 2.69
    }

elif mode == "Sample High-Risk":
    try:
        with open("data/fraud_test.json") as f:
            transaction = json.load(f)
    except Exception:
        st.error("Could not load sample transaction.")

if transaction:

    if st.button("🔍 Analyze Transaction", type="primary"):

        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=transaction,
                timeout=30
            )

            if response.status_code != 200:
                st.error(f"API error: {response.status_code}")

            else:
                result = response.json()

                st.subheader("Risk Assessment")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "XGBoost Risk",
                    f"{result['xgb_score'] * 100:.2f}%"
                )

                col2.metric(
                    "Anomaly Score",
                    f"{result['if_score'] * 100:.2f}%"
                )

                col3.metric(
                    "Overall Risk",
                    f"{result['risk_score'] * 100:.2f}%"
                )

                col4.metric(
                    "Decision Threshold",
                    f"{result['threshold'] * 100:.0f}%"
                )

                if result["prediction"] == 1:
                    st.error("🚨 HIGH RISK TRANSACTION")
                else:
                    st.success("✅ LOW RISK TRANSACTION")

                st.divider()

                st.subheader("🔎 SHAP-Based Explanation")

                for reason in result["top_shap_reasons"]:

                    direction = (
                        "increased risk"
                        if reason["impact"] > 0
                        else "decreased risk"
                    )

                    st.write(
                        f"**{reason['feature']}** — "
                        f"{direction} "
                        f"(impact: {reason['impact']:.4f})"
                    )

                if result.get("agents"):

                    st.divider()
                    st.subheader("🤖 Decision Layer")

                    agents = result["agents"]

                    st.write(
                        f"**Investigator:** "
                        f"{agents['explanation']}"
                    )

                    if agents["decision"] == "AUTO-BLOCK":
                        st.error(
                            f"**Final Decision: {agents['decision']}**"
                        )

                    elif agents["decision"] == "FLAG FOR REVIEW":
                        st.warning(
                            f"**Final Decision: {agents['decision']}**"
                        )

                    else:
                        st.success(
                            f"**Final Decision: {agents['decision']}**"
                        )

                st.divider()

                st.subheader("⚡ Throughput Benchmark")

                try:
                    st.image(
                        "models/throughput_benchmark.png",
                        caption=(
                            "Sequential vs concurrent "
                            "transaction processing"
                        )
                    )
                except Exception:
                    st.info("Benchmark chart unavailable.")

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to FraudGuard API. "
                "Make sure the FastAPI server is running."
            )

        except Exception as e:
            st.error(f"Unexpected error: {e}")