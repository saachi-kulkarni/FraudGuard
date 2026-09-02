class InvestigatorAgent:

    def investigate(self, result):

        try:
            return self.llm_explanation(result)
        except Exception:
            return self.fallback_explanation(result)

    def llm_explanation(self, result):

        raise RuntimeError("LLM not enabled")

    def fallback_explanation(self, result):

        risk = result["risk_score"]
        xgb = result["xgb_score"]
        isolation = result["if_score"]
        reasons = result.get("shap_reasons", [])

        top_reasons = ", ".join(
            r["feature"] for r in reasons[:3]
        )

        if risk >= 0.90:
            level = "very high"
        elif risk >= 0.70:
            level = "high"
        else:
            level = "moderate"

        return (
            f"Fallback investigation: transaction has "
            f"{level} fraud risk (risk score={risk:.3f}). "
            f"XGBoost probability={xgb:.3f}, "
            f"Isolation Forest anomaly score={isolation:.3f}. "
            f"Top contributing features: {top_reasons}."
        )


class DecisionAgent:

    def decide(self, result):

        risk = result["risk_score"]

        if risk >= 0.90:
            return "AUTO-BLOCK"

        if risk >= 0.70:
            return "FLAG FOR REVIEW"

        return "ALLOW"


def run_agents(result):

    investigator = InvestigatorAgent()
    decision = DecisionAgent()

    explanation = investigator.investigate(result)
    action = decision.decide(result)

    return {
        "explanation": explanation,
        "decision": action
    }