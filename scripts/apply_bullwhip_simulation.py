import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# BullwhipAI
# Controlled Daily Bullwhip Simulation
#
# Kaggle + SDV + Bullwhip Simulation
# =========================================================


np.random.seed(42)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "synthetic"
    / "synthetic_supply_chain.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "final"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "augmented_supply_chain.csv"
)


# =========================================================
# CONFIGURATION
# =========================================================

WINDOW = 30


# =========================================================
# LOAD SYNTHETIC DATA
# =========================================================

print("=" * 65)
print("BULLWHIPAI - CONTROLLED BULLWHIP SIMULATION")
print("=" * 65)

print("\nLoading SDV synthetic dataset...")

df = pd.read_csv(INPUT_FILE)

print(
    f"Input transaction shape: {df.shape}"
)


# =========================================================
# DATE PROCESSING
# =========================================================

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)

df = df.dropna(
    subset=["Order_Date"]
).copy()

df["Date"] = (
    df["Order_Date"]
    .dt.normalize()
)


# =========================================================
# DAILY AGGREGATION
# =========================================================

print(
    "\nAggregating transactions by date..."
)


daily = (
    df.groupby(
        "Date",
        as_index=False
    )
    .agg(

        Consumer_Demand=(
            "Order_Quantity",
            "sum"
        ),

        Product_Price=(
            "Product_Price",
            "mean"
        ),

        Sales=(
            "Sales",
            "sum"
        ),

        Order_Total=(
            "Order_Total",
            "sum"
        ),

        Discount_Rate=(
            "Discount_Rate",
            "mean"
        ),

        Actual_Shipping_Days=(
            "Actual_Shipping_Days",
            "mean"
        ),

        Scheduled_Shipping_Days=(
            "Scheduled_Shipping_Days",
            "mean"
        ),

        Late_Delivery_Risk=(
            "Late_Delivery_Risk",
            "mean"
        ),

        Shipping_Delay_Days=(
            "Shipping_Delay_Days",
            "mean"
        ),

        Product_Category=(
            "Product_Category",
            lambda x:
            x.mode().iloc[0]
            if not x.mode().empty
            else x.iloc[0]
        ),

        Shipping_Mode=(
            "Shipping_Mode",
            lambda x:
            x.mode().iloc[0]
            if not x.mode().empty
            else x.iloc[0]
        ),

        Market=(
            "Market",
            lambda x:
            x.mode().iloc[0]
            if not x.mode().empty
            else x.iloc[0]
        ),

        Order_Region=(
            "Order_Region",
            lambda x:
            x.mode().iloc[0]
            if not x.mode().empty
            else x.iloc[0]
        ),

        Customer_Segment=(
            "Customer_Segment",
            lambda x:
            x.mode().iloc[0]
            if not x.mode().empty
            else x.iloc[0]
        ),
    )
)


daily = (
    daily
    .sort_values("Date")
    .reset_index(drop=True)
)


print(
    f"Daily dataset shape: {daily.shape}"
)


# =========================================================
# CONSUMER DEMAND
# =========================================================

consumer_demand = (
    daily["Consumer_Demand"]
    .astype(float)
)


# =========================================================
# CONTROLLED SUPPLY CHAIN SIMULATION
# =========================================================
#
# The order levels are based on:
#
# Retailer:
#     Demand × 1.8
#     + small uncertainty
#
# Wholesaler:
#     Retailer × 2.0
#     + moderate uncertainty
#
# Manufacturer:
#     Wholesaler × 2.5
#     + larger uncertainty
#
# Multiplicative log-normal noise is used because it:
#
# 1. Keeps orders positive
# 2. Doesn't create negative orders
# 3. Increases variability progressively
# 4. Produces a more realistic Bullwhip Effect
#
# =========================================================


def generate_order(
    demand,
    amplification,
    uncertainty
):

    base_order = (
        demand
        * amplification
    )


    # Log-normal multiplicative uncertainty
    #
    # Median multiplier remains approximately 1,
    # while variability increases with uncertainty.

    noise = np.random.lognormal(
        mean=-(uncertainty ** 2) / 2,
        sigma=uncertainty,
        size=len(demand)
    )


    order = (
        base_order
        * noise
    )


    return order


# =========================================================
# RETAILER
# =========================================================

daily["Retailer_Order"] = generate_order(
    demand=consumer_demand,

    amplification=1.8,

    # Low uncertainty
    uncertainty=0.10
)


# =========================================================
# WHOLESALER
# =========================================================

daily["Wholesaler_Order"] = generate_order(
    demand=daily["Retailer_Order"],

    amplification=2.0,

    # Moderate uncertainty
    uncertainty=0.20
)


# =========================================================
# MANUFACTURER
# =========================================================

daily["Manufacturer_Order"] = generate_order(
    demand=daily["Wholesaler_Order"],

    amplification=2.5,

    # Higher uncertainty
    uncertainty=0.35
)


# =========================================================
# SAFETY CHECK
# =========================================================

daily["Retailer_Order"] = (
    daily["Retailer_Order"]
    .clip(lower=0)
)

daily["Wholesaler_Order"] = (
    daily["Wholesaler_Order"]
    .clip(lower=0)
)

daily["Manufacturer_Order"] = (
    daily["Manufacturer_Order"]
    .clip(lower=0)
)


# =========================================================
# DEMAND AMPLIFICATION
# =========================================================

daily["Demand_Amplification"] = (

    daily["Retailer_Order"]

    / daily["Consumer_Demand"]
)


# =========================================================
# ROLLING BULLWHIP RATIO
# =========================================================
#
# CV = Standard deviation / Mean
#
# Bullwhip Ratio =
#
# CV(Manufacturer)
# ----------------
# CV(Consumer)
#
# =========================================================


consumer_cv = (
    daily["Consumer_Demand"]
    .rolling(
        window=WINDOW,
        min_periods=5
    )
    .std()
    /
    daily["Consumer_Demand"]
    .rolling(
        window=WINDOW,
        min_periods=5
    )
    .mean()
)


manufacturer_cv = (
    daily["Manufacturer_Order"]
    .rolling(
        window=WINDOW,
        min_periods=5
    )
    .std()
    /
    daily["Manufacturer_Order"]
    .rolling(
        window=WINDOW,
        min_periods=5
    )
    .mean()
)


daily["Bullwhip_Ratio"] = (

    manufacturer_cv

    / (
        consumer_cv
        + 1e-9
    )
)


# ---------------------------------------------------------
# Initial rolling period
# ---------------------------------------------------------

daily["Bullwhip_Ratio"] = (
    daily["Bullwhip_Ratio"]
    .fillna(1.0)
)


# ---------------------------------------------------------
# Remove impossible numerical values
# ---------------------------------------------------------

daily["Bullwhip_Ratio"] = (
    daily["Bullwhip_Ratio"]
    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )
    .fillna(1.0)
)


# =========================================================
# LEAD TIME
# =========================================================

daily["Lead_Time_Days"] = (
    daily["Actual_Shipping_Days"]
    .round(2)
)


# =========================================================
# SUPPLY DISRUPTION
# =========================================================

daily["Supply_Disruption_Rate"] = (
    daily["Late_Delivery_Risk"]
    .round(3)
)


daily["Supply_Disruption"] = (
    daily["Supply_Disruption_Rate"]
    >= 0.5
).astype(int)


# =========================================================
# INVENTORY
# =========================================================

inventory_noise = np.random.normal(
    0,
    daily["Consumer_Demand"] * 0.10
)


daily["Inventory_Level"] = (

    500

    + daily["Manufacturer_Order"]
    * 0.20

    + inventory_noise
)


daily["Inventory_Level"] = (
    daily["Inventory_Level"]
    .clip(lower=0)
    .round(2)
)


# =========================================================
# PRICE STRESS
# =========================================================

price_median = (
    daily["Product_Price"]
    .median()
)


daily["Price_Stress"] = (

    daily["Product_Price"]

    / max(
        price_median,
        1
    )
)


# =========================================================
# RISK SCORE
# =========================================================

daily["Risk_Score"] = (

    daily["Bullwhip_Ratio"]
    * 20

    + daily["Supply_Disruption"]
    * 30

    + (
        daily["Lead_Time_Days"]
        - 5
    ).clip(lower=0)
    * 5

    + (
        daily["Price_Stress"]
        - 1
    ).clip(lower=0)
    * 10
)


daily["Risk_Score"] = (
    daily["Risk_Score"]
    .clip(
        0,
        100
    )
    .round(2)
)


# =========================================================
# RISK LABEL
# =========================================================

def calculate_risk_label(row):

    score = 0


    # Bullwhip Effect

    if row["Bullwhip_Ratio"] > 2.5:

        score += 2

    elif row["Bullwhip_Ratio"] > 1.5:

        score += 1


    # Supply disruption

    if row["Supply_Disruption"] == 1:

        score += 2


    # Lead time

    if row["Lead_Time_Days"] > 8:

        score += 1


    # Price stress

    if (
        row["Product_Price"]
        > price_median * 1.25
    ):

        score += 1


    if score >= 3:

        return "HIGH"

    elif score >= 1:

        return "MEDIUM"

    return "LOW"


daily["Risk_Label"] = (
    daily.apply(
        calculate_risk_label,
        axis=1
    )
)


# =========================================================
# NUMERIC RISK LEVEL
# =========================================================

daily["Risk_Level_Num"] = (
    daily["Risk_Label"]
    .map(
        {
            "LOW": 0,
            "MEDIUM": 1,
            "HIGH": 2
        }
    )
)


# =========================================================
# DATE FEATURES
# =========================================================

daily["Year"] = (
    daily["Date"]
    .dt.year
)

daily["Month"] = (
    daily["Date"]
    .dt.month
)

daily["Day_of_Year"] = (
    daily["Date"]
    .dt.dayofyear
)


# =========================================================
# ROUND NUMERICAL COLUMNS
# =========================================================

numeric_columns = [

    "Consumer_Demand",

    "Retailer_Order",

    "Wholesaler_Order",

    "Manufacturer_Order",

    "Demand_Amplification",

    "Bullwhip_Ratio",

    "Lead_Time_Days",

    "Supply_Disruption_Rate",

    "Inventory_Level",

    "Price_Stress",

    "Risk_Score",
]


for column in numeric_columns:

    daily[column] = (
        daily[column]
        .round(4)
    )


# =========================================================
# SAVE
# =========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


daily.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 65)
print("BULLWHIP SIMULATION COMPLETED")
print("=" * 65)


print(
    f"\nFinal dataset shape: {daily.shape}"
)


print(
    f"\nOutput file:\n{OUTPUT_FILE}"
)


# =========================================================
# RISK DISTRIBUTION
# =========================================================

print(
    "\n----------------------------------------"
)

print(
    "RISK DISTRIBUTION"
)

print(
    "----------------------------------------"
)

print(
    daily["Risk_Label"]
    .value_counts()
)


# =========================================================
# BULLWHIP RATIO
# =========================================================

print(
    "\n----------------------------------------"
)

print(
    "BULLWHIP RATIO"
)

print(
    "----------------------------------------"
)

print(
    daily["Bullwhip_Ratio"]
    .describe()
    .round(3)
)


# =========================================================
# DEMAND AMPLIFICATION
# =========================================================

print(
    "\n----------------------------------------"
)

print(
    "DEMAND AMPLIFICATION"
)

print(
    "----------------------------------------"
)

print(
    daily["Demand_Amplification"]
    .describe()
    .round(3)
)


# =========================================================
# SUPPLY CHAIN ORDER STATISTICS
# =========================================================

print(
    "\n----------------------------------------"
)

print(
    "SUPPLY CHAIN ORDER STATISTICS"
)

print(
    "----------------------------------------"
)

print(
    daily[
        [
            "Consumer_Demand",
            "Retailer_Order",
            "Wholesaler_Order",
            "Manufacturer_Order",
        ]
    ]
    .describe()
    .round(3)
)


# =========================================================
# SAMPLE
# =========================================================

print(
    "\n----------------------------------------"
)

print(
    "SAMPLE"
)

print(
    "----------------------------------------"
)

print(
    daily[
        [
            "Date",
            "Product_Category",
            "Consumer_Demand",
            "Retailer_Order",
            "Wholesaler_Order",
            "Manufacturer_Order",
            "Bullwhip_Ratio",
            "Risk_Score",
            "Risk_Label",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print(
    "\n✅ Kaggle + SDV + Bullwhip simulation completed!"
)