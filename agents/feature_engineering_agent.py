"""
BullwhipAI
Feature Engineering Agent

Purpose:
Reads historical supply chain data,
creates ML-friendly business features,
and saves the processed dataset.
"""

import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR /
    "datasets" /
    "historical" /
    "historical_supply_chain.csv"
)

OUTPUT_DIR = (
    BASE_DIR /
    "datasets" /
    "processed"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "featured_supply_chain.csv"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\nLoading historical dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} rows.")

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

print("\nCreating engineered features...")

# -----------------------------
# Inventory Gap
# Positive = demand exceeds inventory
# -----------------------------
df["InventoryGap"] = df["Demand"] - df["Inventory"]

# -----------------------------
# Inventory Ratio
# >1 means enough stock
# <1 means shortage
# -----------------------------
df["InventoryRatio"] = (
    df["Inventory"] /
    df["Demand"].replace(0, 1)
)

# -----------------------------
# Demand / Inventory Ratio
# Higher value means higher stress
# -----------------------------
df["DemandSupplyRatio"] = (
    df["Demand"] /
    df["Inventory"].replace(0, 1)
)

# -----------------------------
# Weather Severity Index
# -----------------------------
df["WeatherSeverity"] = (
    (df["Rainfall"] * 0.6) +
    (df["WindSpeed"] * 0.4)
)

# Normalize weather score (0-100)
max_weather = df["WeatherSeverity"].max()

if max_weather != 0:
    df["WeatherSeverity"] = (
        df["WeatherSeverity"] /
        max_weather
    ) * 100

# -----------------------------
# Supply Stress Score
# -----------------------------
df["SupplyStress"] = (
    df["CommodityPrice"] *
    df["LeadTime"]
)

# -----------------------------
# Transport Risk
# -----------------------------
df["TransportRisk"] = (
    df["TransportDelay"] /
    df["LeadTime"].replace(0, 1)
)

# -----------------------------
# Fuel Impact
# -----------------------------
df["FuelImpact"] = (
    df["FuelPrice"] *
    df["TransportDelay"]
)

# -----------------------------
# Market Risk Score
# -----------------------------
sentiment_score = {
    "Positive": 0,
    "Neutral": 1,
    "Negative": 2
}

df["MarketRisk"] = (
    df["MarketSentiment"]
    .map(sentiment_score)
)

# -----------------------------
# Delay Score
# -----------------------------
df["DelayScore"] = (
    df["TransportDelay"] +
    (df["PortDelay"] * 2) +
    (df["RailwayDelay"] * 2)
)

# -----------------------------
# External Disruption Score
# -----------------------------
df["ExternalDisruption"] = (
    df["GovernmentAlert"] +
    df["Festival"] +
    df["Holiday"]
)

# --------------------------------------------------
# ROUND DECIMAL FEATURES
# --------------------------------------------------

decimal_columns = [
    "InventoryRatio",
    "DemandSupplyRatio",
    "WeatherSeverity",
    "TransportRisk",
    "FuelImpact"
]

for col in decimal_columns:
    df[col] = df[col].round(2)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\nFeature Engineering Complete!")

print(f"Rows : {len(df)}")
print(f"Columns : {len(df.columns)}")

print("\nNew Features Added:")

new_features = [
    "InventoryGap",
    "InventoryRatio",
    "DemandSupplyRatio",
    "WeatherSeverity",
    "SupplyStress",
    "TransportRisk",
    "FuelImpact",
    "MarketRisk",
    "DelayScore",
    "ExternalDisruption"
]

for feature in new_features:
    print(f"✓ {feature}")

print("\nOutput File:")
print(OUTPUT_FILE)

print("\nPreview:\n")
print(df.head())