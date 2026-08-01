"""
BullwhipAI - Historical Dataset Generator v2.1
================================================
Generates 6 years (2020-2025) of realistic food supply chain data.
Fixed: Bullwhip Ratio now uses Coefficient of Variation method (range 0.8-4.0)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

random.seed(42)
np.random.seed(42)

print("=" * 60)
print("BULLWHIP AI - HISTORICAL DATASET GENERATOR v2.1")
print("=" * 60)

# -------------------------------------------------------
# 1. DATE RANGE
# -------------------------------------------------------
start_date = datetime(2020, 1, 1)
end_date   = datetime(2025, 12, 31)
dates = []
d = start_date
while d <= end_date:
    dates.append(d)
    d += timedelta(days=1)

n = len(dates)
print(f"Total days in dataset: {n}")

# -------------------------------------------------------
# 2. HELPER FUNCTIONS
# -------------------------------------------------------

def seasonal_demand(date, base=1000):
    month = date.month
    doy   = date.timetuple().tm_yday
    annual   = 0.30 * np.sin(2 * np.pi * (doy - 60) / 365)
    festival = 0.15 if month in [10, 11] else 0.0
    monsoon  = -0.10 if month in [7, 8] else 0.0
    noise    = np.random.normal(0, 0.05)
    return max(100, base * (1 + annual + festival + monsoon + noise))


def commodity_price(date, base=4000):
    years_elapsed = (date - start_date).days / 365
    trend    = base + years_elapsed * 300
    month    = date.month
    harvest  = -200 if month in [10, 11, 12, 3, 4] else 0
    seasonal = 80 * np.sin(2 * np.pi * month / 12)
    shock    = np.random.choice([0, 0, 0, 300, -200, 500],
                                p=[0.55, 0.20, 0.10, 0.05, 0.05, 0.05])
    noise    = np.random.normal(0, 80)
    return max(1000, trend + harvest + seasonal + shock + noise)


def temperature(date):
    month = date.month
    seasonal = {1:24,2:26,3:30,4:34,5:37,6:31,7:27,8:27,9:29,10:32,11:28,12:25}
    return round(seasonal[month] + np.random.normal(0, 1.5), 1)


def wind_speed(date):
    month = date.month
    base = 18 if month in [5,6,7,8,9] else 10
    return round(max(0, base + np.random.normal(0, 3)), 1)


def rainfall(date):
    month = date.month
    if month in [6,7,8,9]: return round(max(0, np.random.exponential(15)), 1)
    elif month in [10,11]:  return round(max(0, np.random.exponential(4)), 1)
    else:                   return round(max(0, np.random.exponential(0.5)), 1)


def lead_time(date):
    month = date.month
    base = 5
    if month in [7,8]:   base += 3   # monsoon delays
    if month in [10,11]: base += 2   # festival rush
    return max(1, base + np.random.randint(-1, 3))


def supply_disruption(date):
    month = date.month
    prob = 0.05
    if month in [6,7,8,9]:           prob = 0.12   # monsoon
    if date.year in [2020, 2021]:    prob += 0.08  # COVID
    return int(np.random.random() < prob)


def news_sentiment(date):
    base = {2020: -0.4, 2021: -0.1, 2022: 0.1, 2023: 0.2, 2024: 0.2, 2025: 0.2}
    b = base.get(date.year, 0.1)
    return round(float(np.clip(b + np.random.normal(0, 0.3), -1, 1)), 3)


def risk_label(bwr, supply_disr, lead_t, price):
    """
    Multi-signal risk scoring.
    Returns: 'LOW', 'MEDIUM', or 'HIGH'
    """
    score = 0
    if bwr > 2.5:      score += 2
    elif bwr > 1.5:    score += 1

    if supply_disr == 1: score += 2
    if lead_t > 8:       score += 1
    if price > 5000:     score += 1
    # COVID penalty
    score_map = {0: "LOW", 1: "MEDIUM", 2: "MEDIUM", 3: "HIGH", 4: "HIGH", 5: "HIGH"}
    return score_map.get(score, "HIGH")


# -------------------------------------------------------
# 3. BUILD DATASET
# -------------------------------------------------------

print("\nGenerating daily supply chain records...")

PRODUCTS = ["Rice", "Wheat", "Tomato", "Onion", "Sugar", "Milk"]

records = []

# Rolling history for Bullwhip Ratio calculation
WINDOW = 30
consumer_history = []
manufacturer_history = []

for i, date in enumerate(dates):

    # ── Demand & Orders ──────────────────────────────────
    consumer_demand     = seasonal_demand(date, base=1000)

    # Each tier amplifies the order it receives
    # Amplification factors: retailer 1.8x, wholesaler 2.0x, manufacturer 2.5x
    # + independent noise at each stage (this is what causes bullwhip)
    retailer_order      = consumer_demand * 1.8  + np.random.normal(0, consumer_demand * 0.08)
    wholesaler_order    = retailer_order  * 2.0  + np.random.normal(0, retailer_order  * 0.10)
    manufacturer_order  = wholesaler_order* 2.5  + np.random.normal(0, wholesaler_order* 0.12)

    retailer_order     = max(0, retailer_order)
    wholesaler_order   = max(0, wholesaler_order)
    manufacturer_order = max(0, manufacturer_order)

    # ── Rolling Bullwhip Ratio (CV method) ────────────────
    # CV = std/mean   (Coefficient of Variation, dimensionless)
    # Bullwhip Ratio = CV(manufacturer_orders) / CV(consumer_demand)
    # This stays in a sensible range regardless of absolute scale

    consumer_history.append(consumer_demand)
    manufacturer_history.append(manufacturer_order)

    if len(consumer_history) > WINDOW:
        consumer_history.pop(0)
        manufacturer_history.pop(0)

    if len(consumer_history) >= 5:
        c_mean = np.mean(consumer_history);    c_std = np.std(consumer_history)
        m_mean = np.mean(manufacturer_history); m_std = np.std(manufacturer_history)
        c_cv   = c_std / (c_mean + 1e-9)
        m_cv   = m_std / (m_mean + 1e-9)
        bwr    = round(float(m_cv / (c_cv + 1e-9)), 4)
        bwr    = max(0.5, min(bwr, 10.0))   # cap at reasonable range
    else:
        bwr = 1.0

    # ── Environmental ─────────────────────────────────────
    temp  = temperature(date)
    wind  = wind_speed(date)
    rain  = rainfall(date)
    price = round(commodity_price(date), 0)
    lt    = lead_time(date)
    disr  = supply_disruption(date)
    senti = news_sentiment(date)

    # ── Inventory (simplified) ────────────────────────────
    inv = round(max(0, 500 + manufacturer_order * 0.4 + np.random.normal(0, 50)), 0)

    # ── Derived metrics ───────────────────────────────────
    demand_amp = round(retailer_order / max(consumer_demand, 1), 3)

    # ── Risk Score (0-100) ────────────────────────────────
    risk_score = round(min(100, max(0,
        bwr * 15
        + disr * 30
        + (lt - 5) * 5
        + max(0, price - 4000) / 100
        + (1 - senti) * 5
    )), 2)

    # ── Risk Label ────────────────────────────────────────
    rl = risk_label(bwr, disr, lt, price)

    product = PRODUCTS[i % len(PRODUCTS)]

    records.append({
        "Date":                 date.strftime("%Y-%m-%d"),
        "Year":                 date.year,
        "Month":                date.month,
        "Day_of_Year":          date.timetuple().tm_yday,
        "Product":              product,
        "Consumer_Demand":      round(consumer_demand, 2),
        "Retailer_Order":       round(retailer_order, 2),
        "Wholesaler_Order":     round(wholesaler_order, 2),
        "Manufacturer_Order":   round(manufacturer_order, 2),
        "Inventory_Level":      inv,
        "Lead_Time_Days":       lt,
        "Supply_Disruption":    disr,
        "Temperature_C":        temp,
        "Wind_Speed_kmh":       wind,
        "Rainfall_mm":          rain,
        "Commodity_Price_INR":  price,
        "News_Sentiment":       senti,
        "Demand_Amplification": demand_amp,
        "Bullwhip_Ratio":       bwr,
        "Risk_Score":           risk_score,
        "Risk_Label":           rl,
        "Risk_Level_Num":       {"LOW":0,"MEDIUM":1,"HIGH":2}[rl],
    })

    if (i + 1) % 365 == 0:
        year_done = date.year
        year_recs = records[-365:]
        highs = sum(1 for r in year_recs if r["Risk_Label"] == "HIGH")
        lows  = sum(1 for r in year_recs if r["Risk_Label"] == "LOW")
        avg_bwr = round(np.mean([r["Bullwhip_Ratio"] for r in year_recs]), 3)
        print(f"  Year {year_done}: HIGH={highs:3d} | LOW={lows:3d} | Avg BWR={avg_bwr}")

# -------------------------------------------------------
# 4. SAVE
# -------------------------------------------------------

df = pd.DataFrame(records)
os.makedirs("data", exist_ok=True)
out_path = "data/historical_supply_chain.csv"
df.to_csv(out_path, index=False)

print(f"\nDataset saved: {out_path}")
print(f"Shape: {df.shape}")
print(f"\nRisk Distribution:")
print(df["Risk_Label"].value_counts())
print(f"\nBullwhip Ratio stats:")
print(df["Bullwhip_Ratio"].describe().round(3))
print(f"\nSample (last 5 rows):")
print(df.tail(5)[["Date","Product","Consumer_Demand","Bullwhip_Ratio","Risk_Label"]].to_string())
print("\n✅ Historical dataset generation complete!")
