"""
BullwhipAI - ML Prediction Agent
==================================
Reads all live data CSVs, engineers features,
runs the trained ML model, and generates XAI explanation.

Output:
  data/ml_prediction.csv     - prediction + confidence
  data/xai_explanation.json  - full XAI explanation
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime

# Add project root to path so we can import xai module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("BULLWHIP AI - ML PREDICTION AGENT")
print("=" * 60)


# -------------------------------------------------------
# LOAD LIVE DATA
# -------------------------------------------------------

def load_live_data():
    """Load the latest readings from all live data files."""

    data = {}

    # --- Weather ---
    weather_path = "data/weather_data.csv"
    if os.path.exists(weather_path):
        w = pd.read_csv(weather_path)
        if not w.empty:
            row = w.iloc[-1]
            data["Temperature_C"]   = float(row.get("Temperature_C", 28))
            data["Wind_Speed_kmh"]  = float(row.get("Wind_Speed_kmh", 10))
            data["Rainfall_mm"]     = float(row.get("Rainfall_mm", 0) if "Rainfall_mm" in row else 0)
            print(f"  Weather: {data['Temperature_C']}°C, wind {data['Wind_Speed_kmh']} km/h")

    # --- Commodity ---
    commodity_path = "data/commodity_data.csv"
    if os.path.exists(commodity_path):
        c = pd.read_csv(commodity_path)
        if not c.empty:
            row = c.iloc[-1]
            price = float(str(row.get("Price", 4200)).replace(",", ""))
            data["Commodity_Price_INR"] = price
            print(f"  Commodity: ₹{price:,.0f}/quintal")

    # --- Market Events ---
    events_path = "data/market_events.csv"
    supply_disruption = 0
    disruption_count  = 0
    if os.path.exists(events_path):
        e = pd.read_csv(events_path).fillna("")
        high_risk = e[e["Risk"].astype(str).str.lower() == "high"]
        if len(high_risk) > 0:
            supply_disruption = 1
            disruption_count  = len(high_risk)
        print(f"  Market events: {len(e)} total, {disruption_count} high-risk")

    data["Supply_Disruption"] = supply_disruption
    data["Disruption_7d"]     = disruption_count

    # --- Risk Score (from rule-based agent) ---
    risk_path = "data/risk_score.csv"
    if os.path.exists(risk_path):
        r = pd.read_csv(risk_path)
        if not r.empty:
            row = r.iloc[0]
            data["Supply_Risk_Score"]    = float(row.get("Supply Risk Score", 15))
            data["Demand_Risk_Score"]    = float(row.get("Demand Risk Score", 50))
            data["Inventory_Risk_Score"] = float(row.get("Inventory Risk Score", 10))
            print(f"  Risk scores: supply={data['Supply_Risk_Score']}, "
                  f"demand={data['Demand_Risk_Score']}, inv={data['Inventory_Risk_Score']}")

    # --- News Sentiment ---
    news_path = "data/news_data.csv"
    sentiment = 0.1
    if os.path.exists(news_path):
        n = pd.read_csv(news_path)
        # Simple keyword-based sentiment
        neg_words = ["shortage", "disruption", "crisis", "delay", "flood",
                     "drought", "strike", "ban", "inflation", "surge", "rise"]
        pos_words = ["growth", "increase", "stable", "recover", "invest",
                     "expand", "surplus", "boost"]
        if not n.empty:
            titles = " ".join(n["Title"].astype(str).head(10)).lower()
            neg_count = sum(titles.count(w) for w in neg_words)
            pos_count = sum(titles.count(w) for w in pos_words)
            total = neg_count + pos_count + 1
            sentiment = round((pos_count - neg_count) / total, 3)
            sentiment = max(-1.0, min(1.0, sentiment))
        print(f"  News sentiment: {sentiment}")

    data["News_Sentiment"] = sentiment

    return data


# -------------------------------------------------------
# ENGINEER FEATURES FOR ML MODEL
# -------------------------------------------------------

def engineer_features(live_data: dict) -> dict:
    """
    Convert live data signals into the full feature vector
    required by the ML model.
    Also calculates the bullwhip ratio from historical data
    if available.
    """

    now = datetime.now()
    month = now.month
    doy   = now.timetuple().tm_yday

    season_map = {
        1: 0, 2: 0, 12: 0,        # Winter
        3: 1, 4: 1, 5: 1,          # Summer
        6: 2, 7: 2, 8: 2, 9: 2,   # Monsoon
        10: 3, 11: 3               # Festive
    }
    season = season_map.get(month, 0)

    # --- Demand-side simulation from commodity + risk signals ---
    # We approximate demand from price signals and risk scores
    base_demand = 1000
    price       = live_data.get("Commodity_Price_INR", 4200)
    temp        = live_data.get("Temperature_C", 28)
    is_monsoon  = 1 if month in [6, 7, 8, 9] else 0
    is_festival = 1 if month in [10, 11] else 0

    demand_mult = (
        1.0
        + (is_festival * 0.15)
        - (is_monsoon  * 0.10)
        + (-0.05 if temp > 36 else 0)
    )
    consumer_demand   = base_demand * demand_mult
    retailer_order    = consumer_demand * 1.8
    wholesaler_order  = retailer_order  * 2.0
    manufacturer_order= wholesaler_order* 2.5
    inventory_level   = 500 + retailer_order * 0.3

    # Bullwhip ratio: compute from historical data if available
    hist_path = "data/historical_supply_chain.csv"
    bullwhip_ratio    = 1.4   # default
    demand_7d_mean    = consumer_demand
    demand_7d_std     = 50.0
    price_7d_mean     = price
    price_change_pct  = 0.0
    lead_time_7d_max  = 5.0

    if os.path.exists(hist_path):
        try:
            hist = pd.read_csv(hist_path)
            # Get last 30 records for the same month
            month_hist = hist[hist["Month"] == month].tail(30)
            if len(month_hist) >= 5:
                d_cv = (month_hist["Consumer_Demand"].std() /
                        (month_hist["Consumer_Demand"].mean() + 1e-9))
                o_cv = (month_hist["Manufacturer_Order"].std() /
                        (month_hist["Manufacturer_Order"].mean() + 1e-9))
                bullwhip_ratio = round(float(o_cv / (d_cv + 1e-9)), 4)
                bullwhip_ratio = max(0.8, min(bullwhip_ratio, 10.0))

                demand_7d_mean   = float(month_hist["Consumer_Demand"].mean())
                demand_7d_std    = float(month_hist["Consumer_Demand"].std())
                price_hist_mean  = float(month_hist["Commodity_Price_INR"].mean())
                price_7d_mean    = price_hist_mean
                price_change_pct = round((price - price_hist_mean) / (price_hist_mean + 1e-9) * 100, 2)

                lt_hist = month_hist["Lead_Time_Days"]
                lead_time_7d_max = float(lt_hist.tail(7).max())
        except Exception as ex:
            print(f"  [warn] Could not read historical data: {ex}")

    # Disruption-adjusted lead time
    base_lead_time = 5
    if live_data.get("Supply_Disruption", 0):
        base_lead_time += 3
    if is_monsoon:
        base_lead_time += 2
    if is_festival:
        base_lead_time += 1

    demand_amp = round(retailer_order / (consumer_demand + 1e-9), 3)

    product_code = now.day % 6   # cycles through products by day

    features = {
        # Supply chain core
        "Consumer_Demand":      round(consumer_demand, 2),
        "Retailer_Order":       round(retailer_order, 2),
        "Wholesaler_Order":     round(wholesaler_order, 2),
        "Manufacturer_Order":   round(manufacturer_order, 2),
        "Inventory_Level":      round(inventory_level, 2),
        "Lead_Time_Days":       base_lead_time,
        "Demand_Amplification": demand_amp,
        "Bullwhip_Ratio":       bullwhip_ratio,

        # Disruption
        "Supply_Disruption":    live_data.get("Supply_Disruption", 0),
        "Disruption_7d":        live_data.get("Disruption_7d", 0),
        "LeadTime_7d_Max":      lead_time_7d_max,

        # Market
        "Commodity_Price_INR":  price,
        "Price_7d_Mean":        price_7d_mean,
        "Price_Change_Pct":     price_change_pct,
        "News_Sentiment":       live_data.get("News_Sentiment", 0.1),

        # Weather
        "Temperature_C":        temp,
        "Wind_Speed_kmh":       live_data.get("Wind_Speed_kmh", 10),
        "Rainfall_mm":          live_data.get("Rainfall_mm", 0),

        # Temporal
        "Month":                month,
        "Day_of_Year":          doy,
        "Season":               season,
        "Is_Festival":          is_festival,
        "Is_Monsoon":           is_monsoon,

        # Rolling
        "Demand_7d_Mean":       round(demand_7d_mean, 2),
        "Demand_7d_Std":        round(demand_7d_std, 2),

        # Product
        "Product_Code":         product_code,
    }

    return features


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

if __name__ == "__main__":

    print("\n📡 Loading live data...")
    live_data = load_live_data()

    print("\n🔧 Engineering features...")
    features = engineer_features(live_data)

    print("\n🤖 Running ML prediction + XAI...")

    from xai.explainer import predict_and_explain

    explanation = predict_and_explain(features)

    risk_label  = explanation["prediction"]["risk_label"]
    confidence  = explanation["prediction"]["confidence_pct"]
    risk_emoji  = explanation["prediction"]["risk_emoji"]

    print(f"\n{'='*50}")
    print(f"  ML PREDICTION: {risk_emoji}")
    print(f"  Confidence:    {confidence}%")
    print(f"{'='*50}")

    print("\n📊 Confidence Breakdown:")
    for level, pct in explanation["prediction"]["confidence_breakdown"].items():
        bar = "█" * int(pct / 5)
        print(f"  {level:6s}: {bar} {pct}%")

    print("\n🔍 Top Driving Factors (XAI):")
    for i, d in enumerate(explanation["top_drivers"][:5], 1):
        print(f"  {i}. [{d['local_impact']:5.1f}%] {d['feature']}")
        print(f"       → {d['assessment']}")

    print("\n📝 Bullwhip Analysis:")
    print(f"  {explanation['bullwhip_narrative']}")

    # Save prediction to CSV for dashboard
    pred_row = {
        "Timestamp":         explanation["generated_at"],
        "Risk_Label":        risk_label,
        "Confidence_Pct":    confidence,
        "Risk_LOW_Pct":      explanation["prediction"]["confidence_breakdown"]["LOW"],
        "Risk_MEDIUM_Pct":   explanation["prediction"]["confidence_breakdown"]["MEDIUM"],
        "Risk_HIGH_Pct":     explanation["prediction"]["confidence_breakdown"]["HIGH"],
        "Top_Driver":        explanation["top_drivers"][0]["feature"] if explanation["top_drivers"] else "",
        "Bullwhip_Ratio":    features.get("Bullwhip_Ratio", ""),
        "Supply_Disruption": features.get("Supply_Disruption", ""),
        "Lead_Time_Days":    features.get("Lead_Time_Days", ""),
        "Commodity_Price":   features.get("Commodity_Price_INR", ""),
        "Temperature":       features.get("Temperature_C", ""),
        "News_Sentiment":    features.get("News_Sentiment", ""),
    }

    pred_df = pd.DataFrame([pred_row])
    pred_path = "data/ml_prediction.csv"
    if os.path.exists(pred_path):
        old = pd.read_csv(pred_path)
        pred_df = pd.concat([old, pred_df]).tail(100)
    pred_df.to_csv(pred_path, index=False)

    print(f"\n✅ Prediction saved → data/ml_prediction.csv")
    print(f"✅ XAI saved       → data/xai_explanation.json")
    print("\n🎓 ML Prediction Agent complete!")
