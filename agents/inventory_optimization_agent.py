"""
BullwhipAI
Inventory Optimization Agent

Purpose:
Calculate inventory KPIs and recommended order quantity.

Output:
datasets/optimization/inventory_optimization.json
"""

import json
from pathlib import Path

import pandas as pd

from config.inventory_rules import (
    TARGET_COVERAGE_DAYS,
    SAFETY_STOCK_PERCENT,
    HOLDING_COST_PER_UNIT,
    HEALTHY_RATIO_LOW,
    HEALTHY_RATIO_HIGH
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR /
    "datasets" /
    "processed" /
    "featured_supply_chain.csv"
)

OUTPUT_DIR = (
    BASE_DIR /
    "datasets" /
    "optimization"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "inventory_optimization.json"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("=" * 60)
print("BullwhipAI - Inventory Optimization Agent")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

latest = df.iloc[-1]

print("\n✓ Dataset Loaded :", len(df))
print("✓ Latest Record Selected")

# --------------------------------------------------
# EXTRACT VALUES
# --------------------------------------------------

inventory = float(latest["Inventory"])
demand = float(latest["Demand"])

inventory_gap = demand - inventory

daily_demand = max(demand / 30.0, 1)

# --------------------------------------------------
# SAFETY STOCK
# --------------------------------------------------

safety_stock = round(demand * SAFETY_STOCK_PERCENT)

# --------------------------------------------------
# REORDER POINT
# --------------------------------------------------

lead_time = float(latest["LeadTime"])

reorder_point = round(
    (daily_demand * lead_time) + safety_stock
)

# --------------------------------------------------
# RECOMMENDED ORDER
# --------------------------------------------------

recommended_inventory = demand + safety_stock

recommended_order = max(
    0,
    round(recommended_inventory - inventory)
)

# --------------------------------------------------
# DAYS OF INVENTORY
# --------------------------------------------------

days_remaining = round(
    inventory / daily_demand,
    1
)

# --------------------------------------------------
# HOLDING COST
# --------------------------------------------------

holding_cost = round(
    inventory * HOLDING_COST_PER_UNIT,
    2
)

# --------------------------------------------------
# INVENTORY HEALTH SCORE
# --------------------------------------------------

ratio = inventory / max(demand, 1)

if HEALTHY_RATIO_LOW <= ratio <= HEALTHY_RATIO_HIGH:

    health_score = 100

elif ratio < HEALTHY_RATIO_LOW:

    health_score = max(
        0,
        round(ratio * 100)
    )

else:

    excess = ratio - HEALTHY_RATIO_HIGH

    health_score = max(
        0,
        round(100 - excess * 100)
    )

# --------------------------------------------------
# HEALTH STATUS
# --------------------------------------------------

if health_score >= 90:

    status = "Excellent"

elif health_score >= 75:

    status = "Good"

elif health_score >= 50:

    status = "Warning"

else:

    status = "Critical"

# --------------------------------------------------
# OUTPUT JSON
# --------------------------------------------------

output = {

    "current_inventory": int(inventory),

    "predicted_demand": int(demand),

    "inventory_gap": int(inventory_gap),

    "recommended_inventory": int(recommended_inventory),

    "recommended_order_quantity": int(recommended_order),

    "safety_stock": int(safety_stock),

    "reorder_point": int(reorder_point),

    "days_remaining": days_remaining,

    "estimated_holding_cost": holding_cost,

    "inventory_health_score": health_score,

    "inventory_status": status

}

with open(OUTPUT_FILE, "w") as file:

    json.dump(output, file, indent=4)

# --------------------------------------------------
# TERMINAL OUTPUT
# --------------------------------------------------

print("\n==============================")
print("INVENTORY ANALYSIS")
print("==============================")

for key, value in output.items():

    print(f"{key:<30} {value}")

print("\nSaved To:")

print(OUTPUT_FILE)

print("\nInventory Optimization Complete.")