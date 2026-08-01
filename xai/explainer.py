"""
BullwhipAI - Explainable AI (XAI) Engine
==========================================
Generates human-readable explanations for every ML prediction.

Approach: Perturbation-based local importance
  - For each input, we perturb each feature by ±1 std
  - Measure how much the predicted probability changes
  - Rank features by their local impact on THIS prediction
  - Build a natural language explanation

This is analogous to LIME / local sensitivity analysis
without requiring any paid libraries.

Outputs:
  data/xai_explanation.json   - latest explanation (for dashboard)
  data/xai_history.csv        - all explanations over time
"""

import numpy as np
import pandas as pd
import json
import joblib
import os
from datetime import datetime


# -------------------------------------------------------
# FEATURE DESCRIPTIONS  (human-readable names + context)
# -------------------------------------------------------

FEATURE_DESCRIPTIONS = {
    "Supply_Disruption":     "Supply chain disruption (strike/flood/port delay)",
    "Bullwhip_Ratio":        "Bullwhip amplification ratio (order variance / demand variance)",
    "Commodity_Price_INR":   "Commodity price (INR per quintal)",
    "Lead_Time_Days":        "Supplier lead time (days)",
    "Price_7d_Mean":         "7-day average commodity price",
    "Disruption_7d":         "Disruptions in last 7 days",
    "Wind_Speed_kmh":        "Wind speed (km/h)",
    "News_Sentiment":        "News sentiment score (-1 negative → +1 positive)",
    "Day_of_Year":           "Day of the year (seasonality indicator)",
    "Consumer_Demand":       "Consumer-level demand",
    "Retailer_Order":        "Retailer order quantity",
    "Wholesaler_Order":      "Wholesaler order quantity",
    "Manufacturer_Order":    "Manufacturer order quantity",
    "Inventory_Level":       "Current inventory level",
    "Demand_Amplification":  "Demand amplification factor (retailer/consumer)",
    "Temperature_C":         "Temperature (°C)",
    "Rainfall_mm":           "Rainfall (mm)",
    "Commodity_Price_INR":   "Commodity price (INR/quintal)",
    "Price_Change_Pct":      "7-day commodity price change (%)",
    "Month":                 "Month of the year",
    "Season":                "Season (0=Winter,1=Summer,2=Monsoon,3=Festive)",
    "Is_Festival":           "Festival season indicator",
    "Is_Monsoon":            "Monsoon season indicator",
    "Demand_7d_Mean":        "7-day average consumer demand",
    "Demand_7d_Std":         "7-day demand variability (std dev)",
    "Product_Code":          "Product category code",
    "LeadTime_7d_Max":       "Maximum lead time in last 7 days",
}

RISK_COLORS = {0: "🟢 LOW", 1: "🟡 MEDIUM", 2: "🔴 HIGH"}
RISK_NAMES  = {0: "LOW",    1: "MEDIUM",    2: "HIGH"}

RISK_THRESHOLDS = {
    "Supply_Disruption":    {"high": 1,    "medium": 0},
    "Bullwhip_Ratio":       {"high": 2.5,  "medium": 1.5},
    "Commodity_Price_INR":  {"high": 5500, "medium": 4500},
    "Lead_Time_Days":       {"high": 9,    "medium": 7},
    "Disruption_7d":        {"high": 3,    "medium": 1},
    "News_Sentiment":       {"high": -0.3, "medium": 0},
    "Rainfall_mm":          {"high": 20,   "medium": 8},
    "Temperature_C":        {"high": 36,   "medium": 32},
}


# -------------------------------------------------------
# LOAD MODEL ARTIFACTS
# -------------------------------------------------------

def load_model():
    """Load the trained model, scaler and feature list."""
    model_dir = "models"
    try:
        model    = joblib.load(f"{model_dir}/bullwhip_model.pkl")
        scaler   = joblib.load(f"{model_dir}/feature_scaler.pkl")
        features = joblib.load(f"{model_dir}/feature_names.pkl")
        feat_imp = pd.read_csv(f"{model_dir}/feature_importance.csv")
        return model, scaler, features, feat_imp
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run: python scripts/train_model.py first")
        return None, None, None, None


# -------------------------------------------------------
# LOCAL FEATURE IMPORTANCE (Perturbation Method)
# -------------------------------------------------------

def local_feature_importance(model, scaler, features, input_vector, n_std=1.0):
    """
    Compute local importance for each feature by perturbation.

    For each feature i:
      - perturb X[i] += n_std * std(feature)
      - observe change in predicted probability for original class
      - higher delta = more important for this specific prediction
    """
    X = np.array(input_vector, dtype=float).reshape(1, -1)
    X_sc = scaler.transform(X)

    # Original prediction
    orig_proba = model.predict_proba(X_sc)[0]
    orig_class = np.argmax(orig_proba)
    orig_conf  = orig_proba[orig_class]

    # Perturbation deltas
    local_imp = []
    feature_stds = np.std(scaler.inverse_transform(
        np.eye(len(features))
    ), axis=0)
    feature_stds = np.where(feature_stds < 1e-6, 1.0, feature_stds)

    for i, feat in enumerate(features):
        X_up = X.copy()
        X_dw = X.copy()
        X_up[0, i] += n_std * feature_stds[i]
        X_dw[0, i] -= n_std * feature_stds[i]

        p_up = model.predict_proba(scaler.transform(X_up))[0][orig_class]
        p_dw = model.predict_proba(scaler.transform(X_dw))[0][orig_class]

        # Delta = how much confidence changes when feature changes
        delta = abs(orig_conf - p_up) + abs(orig_conf - p_dw)
        local_imp.append(delta)

    # Normalize to 0-100
    total = sum(local_imp) + 1e-9
    local_imp_pct = [round(v / total * 100, 2) for v in local_imp]

    result = sorted(
        zip(features, local_imp_pct, input_vector),
        key=lambda x: x[1],
        reverse=True
    )
    return orig_class, orig_conf, orig_proba, result


# -------------------------------------------------------
# NATURAL LANGUAGE EXPLANATION BUILDER
# -------------------------------------------------------

def build_explanation(risk_class, confidence, proba, local_importances,
                       current_data: dict, feat_imp_global: pd.DataFrame):
    """
    Build a structured explanation dict with:
      - prediction + confidence
      - top driving factors (local XAI)
      - per-factor analysis in plain English
      - bullwhip-specific reasoning
      - recommended actions
    """

    risk_label = RISK_NAMES[risk_class]
    risk_emoji = RISK_COLORS[risk_class]

    # Top 5 local drivers
    top_drivers = local_importances[:6]

    # Build factor sentences
    factor_analysis = []
    for feat, local_pct, val in top_drivers:
        desc = FEATURE_DESCRIPTIONS.get(feat, feat)
        thresh = RISK_THRESHOLDS.get(feat, {})

        direction = ""
        concern   = ""

        if feat == "Supply_Disruption":
            if val >= 1:
                concern = "⚠️ Active supply disruption detected"
            else:
                concern = "✅ No disruption currently active"

        elif feat == "Bullwhip_Ratio":
            if val > 2.5:
                concern = f"🔴 Ratio {val:.2f} — severe amplification upstream"
            elif val > 1.5:
                concern = f"🟡 Ratio {val:.2f} — moderate order amplification"
            else:
                concern = f"🟢 Ratio {val:.2f} — demand signal relatively stable"

        elif feat == "Commodity_Price_INR":
            if val > 5500:
                concern = f"🔴 Price ₹{val:,.0f}/quintal — significantly elevated"
            elif val > 4500:
                concern = f"🟡 Price ₹{val:,.0f}/quintal — above normal range"
            else:
                concern = f"🟢 Price ₹{val:,.0f}/quintal — within normal range"

        elif feat == "Lead_Time_Days":
            if val > 9:
                concern = f"🔴 Lead time {val:.0f} days — critically delayed"
            elif val > 7:
                concern = f"🟡 Lead time {val:.0f} days — slightly delayed"
            else:
                concern = f"🟢 Lead time {val:.0f} days — on schedule"

        elif feat == "News_Sentiment":
            if val < -0.3:
                concern = f"🔴 Sentiment {val:.2f} — negative market news dominating"
            elif val < 0:
                concern = f"🟡 Sentiment {val:.2f} — mildly negative news"
            else:
                concern = f"🟢 Sentiment {val:.2f} — positive/neutral market news"

        elif feat == "Rainfall_mm":
            if val > 20:
                concern = f"🔴 Rainfall {val:.1f}mm — heavy rain, logistics disruption risk"
            elif val > 8:
                concern = f"🟡 Rainfall {val:.1f}mm — moderate rain impact"
            else:
                concern = f"🟢 Rainfall {val:.1f}mm — minimal weather impact"

        elif feat == "Disruption_7d":
            if val >= 3:
                concern = f"🔴 {val:.0f} disruptions in last 7 days — persistent instability"
            elif val >= 1:
                concern = f"🟡 {val:.0f} disruption(s) in last 7 days — watch closely"
            else:
                concern = f"🟢 No disruptions in last 7 days"

        elif feat == "Demand_Amplification":
            if val > 2.0:
                concern = f"🔴 Amplification {val:.2f}x — retailer over-ordering vs actual demand"
            elif val > 1.5:
                concern = f"🟡 Amplification {val:.2f}x — moderate order inflation"
            else:
                concern = f"🟢 Amplification {val:.2f}x — ordering matches demand"

        elif feat == "Temperature_C":
            if val > 36:
                concern = f"🔴 {val:.1f}°C — extreme heat, cold-chain pressure, higher perishable risk"
            elif val > 32:
                concern = f"🟡 {val:.1f}°C — above normal, monitor perishables"
            else:
                concern = f"🟢 {val:.1f}°C — comfortable temperature range"
        else:
            concern = f"Value: {val}"

        factor_analysis.append({
            "feature":       feat,
            "description":   desc,
            "value":         val,
            "local_impact":  local_pct,
            "assessment":    concern,
        })

    # Bullwhip-specific narrative
    bwr = current_data.get("Bullwhip_Ratio", 1.0)
    disr = current_data.get("Supply_Disruption", 0)
    lt   = current_data.get("Lead_Time_Days", 5)

    if risk_class == 2:  # HIGH
        bullwhip_narrative = (
            f"The Bullwhip Effect is ACTIVE. Order amplification ratio of {bwr:.2f} "
            f"means manufacturers are ordering {bwr:.1f}x more than actual consumer demand "
            f"requires. This leads to overproduction, excess inventory costs, and "
            f"sudden stockouts when the demand signal corrects. "
            + ("A supply disruption is compounding the panic-ordering behaviour. " if disr else "")
            + (f"Extended lead times ({lt} days) are causing safety-stock over-accumulation. " if lt > 7 else "")
        )
    elif risk_class == 1:  # MEDIUM
        bullwhip_narrative = (
            f"Mild Bullwhip Effect detected. Ratio of {bwr:.2f} indicates some "
            f"order inflation upstream. Early intervention can prevent escalation to "
            f"HIGH risk. Monitor demand signals closely and avoid reactive over-ordering."
        )
    else:  # LOW
        bullwhip_narrative = (
            f"Bullwhip Effect is minimal (ratio {bwr:.2f}). Supply chain demand "
            f"signals are relatively stable and well-matched across tiers. "
            f"Continue normal operations."
        )

    # Recommended actions based on risk
    actions = _get_actions(risk_class, current_data)

    # Confidence breakdown
    confidence_breakdown = {
        "LOW":    round(float(proba[0]) * 100, 1),
        "MEDIUM": round(float(proba[1]) * 100, 1),
        "HIGH":   round(float(proba[2]) * 100, 1),
    }

    explanation = {
        "generated_at":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": {
            "risk_label":        risk_label,
            "risk_emoji":        risk_emoji,
            "confidence_pct":    round(float(confidence) * 100, 1),
            "confidence_breakdown": confidence_breakdown,
        },
        "top_drivers":           factor_analysis,
        "bullwhip_narrative":    bullwhip_narrative,
        "actions":               actions,
        "model_info": {
            "algorithm":         "Random Forest (200 trees)",
            "features_used":     len(local_importances),
            "xai_method":        "Perturbation-based local sensitivity",
        }
    }

    return explanation


def _get_actions(risk_class, data):
    bwr  = data.get("Bullwhip_Ratio", 1.0)
    disr = data.get("Supply_Disruption", 0)
    lt   = data.get("Lead_Time_Days", 5)
    pri  = data.get("Commodity_Price_INR", 4000)
    rain = data.get("Rainfall_mm", 0)
    temp = data.get("Temperature_C", 28)

    if risk_class == 2:  # HIGH
        return {
            "🏭 Procurement Team": [
                "Activate alternate/backup suppliers immediately",
                "Expedite critical raw material orders",
                f"Negotiate emergency procurement — current price ₹{pri:,.0f}/quintal",
                "Avoid panic bulk-buying; it amplifies the bullwhip further",
            ],
            "📦 Inventory Team": [
                "Increase safety stock by 20-30% for high-risk SKUs",
                "Audit warehouse capacity — avoid overstock induced by false signals",
                "Prioritise FIFO for perishables" + (" in heat conditions" if temp > 35 else ""),
                "Implement daily stock visibility reporting",
            ],
            "🏗 Production Team": [
                "Reduce production batch sizes to stay demand-responsive",
                "Do NOT inflate production targets based on retailer panic orders",
                "Use 7-day rolling demand average as planning baseline",
                "Delay large production runs until demand signal stabilises",
            ],
            "🚚 Logistics Team": [
                f"Lead time {lt} days — pre-book transportation in advance" if lt > 7 else "Monitor transport availability",
                "Assess road/rail disruptions" + (" due to heavy rainfall" if rain > 15 else ""),
                "Activate contingency routes if primary logistics are blocked",
            ],
        }
    elif risk_class == 1:  # MEDIUM
        return {
            "🏭 Procurement Team": [
                "Review supplier lead times — flag delays early",
                "Maintain existing supplier contracts; avoid reactive spot buying",
                f"Commodity price trending at ₹{pri:,.0f}/quintal — monitor weekly",
            ],
            "📦 Inventory Team": [
                "Maintain current safety stock levels",
                "Run demand variance analysis for the week",
                "Check for signs of accumulation due to retailer over-ordering",
            ],
            "🏗 Production Team": [
                "Continue current production plan",
                "Build small buffer (5-10%) for festive/monsoon variability" if data.get("Is_Festival") or data.get("Is_Monsoon") else "No major production changes required",
            ],
            "🚚 Logistics Team": [
                "Standard monitoring — no major disruptions expected",
                "Pre-plan monsoon contingencies" if data.get("Is_Monsoon") else "Ensure regular delivery schedules",
            ],
        }
    else:  # LOW
        return {
            "🏭 Procurement Team": [
                "Continue standard procurement schedule",
                "Opportunity to renegotiate long-term contracts at stable prices",
            ],
            "📦 Inventory Team": [
                "Maintain lean inventory — avoid unnecessary stock accumulation",
                "Use this stable period to optimise warehouse layout",
            ],
            "🏗 Production Team": [
                "Normal production run",
                "Review and update demand forecasts for next quarter",
            ],
            "🚚 Logistics Team": [
                "Standard operations — all routes clear",
            ],
        }


# -------------------------------------------------------
# MAIN PREDICT + EXPLAIN FUNCTION
# -------------------------------------------------------

def predict_and_explain(current_data: dict):
    """
    Given a dictionary of current feature values,
    returns the ML prediction + full XAI explanation.

    current_data keys must match FEATURES list.
    Missing keys will be filled with safe defaults.
    """

    model, scaler, features, feat_imp = load_model()
    if model is None:
        return {"error": "Model not loaded. Train first."}

    # Fill defaults for any missing features
    DEFAULTS = {
        "Consumer_Demand":      1000,
        "Retailer_Order":       1800,
        "Wholesaler_Order":     3960,
        "Manufacturer_Order":   8712,
        "Inventory_Level":      500,
        "Lead_Time_Days":       5,
        "Demand_Amplification": 1.8,
        "Bullwhip_Ratio":       1.4,
        "Supply_Disruption":    0,
        "Disruption_7d":        0,
        "LeadTime_7d_Max":      5,
        "Commodity_Price_INR":  4200,
        "Price_7d_Mean":        4200,
        "Price_Change_Pct":     0,
        "News_Sentiment":       0.1,
        "Temperature_C":        28,
        "Wind_Speed_kmh":       10,
        "Rainfall_mm":          0,
        "Month":                datetime.now().month,
        "Day_of_Year":          datetime.now().timetuple().tm_yday,
        "Season":               2 if datetime.now().month in [6,7,8,9] else 3,
        "Is_Festival":          1 if datetime.now().month in [10,11] else 0,
        "Is_Monsoon":           1 if datetime.now().month in [6,7,8,9] else 0,
        "Demand_7d_Mean":       1000,
        "Demand_7d_Std":        50,
        "Product_Code":         0,
    }

    for key, val in DEFAULTS.items():
        if key not in current_data:
            current_data[key] = val

    # Build input vector in correct feature order
    input_vec = [current_data.get(f, DEFAULTS.get(f, 0)) for f in features]

    # Predict
    risk_class, confidence, proba, local_imp = local_feature_importance(
        model, scaler, features, input_vec
    )

    # Build explanation
    explanation = build_explanation(
        risk_class, confidence, proba, local_imp,
        current_data, feat_imp
    )

    # Save to file
    os.makedirs("data", exist_ok=True)
    with open("data/xai_explanation.json", "w") as f:
        json.dump(explanation, f, indent=2)

    # Append to history
    history_row = {
        "Timestamp": explanation["generated_at"],
        "Risk_Label": explanation["prediction"]["risk_label"],
        "Confidence_Pct": explanation["prediction"]["confidence_pct"],
        "Top_Driver": local_imp[0][0] if local_imp else "",
        "Bullwhip_Ratio": current_data.get("Bullwhip_Ratio", ""),
        "Supply_Disruption": current_data.get("Supply_Disruption", ""),
        "Commodity_Price": current_data.get("Commodity_Price_INR", ""),
    }
    history_path = "data/xai_history.csv"
    hist_df = pd.DataFrame([history_row])
    if os.path.exists(history_path):
        old = pd.read_csv(history_path)
        hist_df = pd.concat([old, hist_df]).tail(200)
    hist_df.to_csv(history_path, index=False)

    return explanation


# -------------------------------------------------------
# STANDALONE TEST
# -------------------------------------------------------

if __name__ == "__main__":
    print("Testing XAI Explainer...")

    # Simulate current conditions
    test_data = {
        "Supply_Disruption":    1,
        "Bullwhip_Ratio":       2.8,
        "Commodity_Price_INR":  5800,
        "Lead_Time_Days":       10,
        "Rainfall_mm":          25,
        "Temperature_C":        31,
        "News_Sentiment":       -0.4,
        "Disruption_7d":        3,
        "Demand_Amplification": 2.1,
    }

    result = predict_and_explain(test_data)

    print(f"\nPrediction: {result['prediction']['risk_emoji']}")
    print(f"Confidence: {result['prediction']['confidence_pct']}%")
    print(f"\nTop Drivers:")
    for d in result["top_drivers"][:5]:
        print(f"  [{d['local_impact']:.1f}%] {d['feature']}: {d['assessment']}")
    print(f"\nBullwhip Narrative:")
    print(result["bullwhip_narrative"])
    print(f"\nActions:")
    for team, acts in result["actions"].items():
        print(f"\n  {team}:")
        for a in acts:
            print(f"    • {a}")
