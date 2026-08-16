import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "final"
    / "augmented_supply_chain.csv"
)


# =========================================================
# Load dataset
# =========================================================

print("=" * 65)
print("BULLWHIPAI - FINAL DATASET VALIDATION")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

print(f"\nDataset shape: {df.shape}")


# =========================================================
# 1. Missing values
# =========================================================

print("\n" + "-" * 40)
print("MISSING VALUE CHECK")
print("-" * 40)

missing = df.isnull().sum()

missing = missing[missing > 0]

if missing.empty:
    print("No missing values found.")
else:
    print(missing)


# =========================================================
# 2. Duplicate rows
# =========================================================

print("\n" + "-" * 40)
print("DUPLICATE CHECK")
print("-" * 40)

duplicates = df.duplicated().sum()

print(
    f"Duplicate rows: {duplicates}"
)


# =========================================================
# 3. Negative values
# =========================================================

print("\n" + "-" * 40)
print("NEGATIVE VALUE CHECK")
print("-" * 40)


numeric_columns = [
    "Consumer_Demand",
    "Product_Price",
    "Sales",
    "Order_Total",
    "Retailer_Order",
    "Wholesaler_Order",
    "Manufacturer_Order",
    "Demand_Amplification",
    "Bullwhip_Ratio",
    "Lead_Time_Days",
    "Inventory_Level",
    "Risk_Score",
]


for column in numeric_columns:

    if column in df.columns:

        count = (
            df[column] < 0
        ).sum()

        print(
            f"{column}: {count}"
        )


# =========================================================
# 4. Bullwhip Ratio
# =========================================================

print("\n" + "-" * 40)
print("BULLWHIP RATIO CHECK")
print("-" * 40)

print(
    df["Bullwhip_Ratio"]
    .describe()
    .round(3)
)


# =========================================================
# 5. Demand amplification
# =========================================================

print("\n" + "-" * 40)
print("DEMAND AMPLIFICATION CHECK")
print("-" * 40)

print(
    df["Demand_Amplification"]
    .describe()
    .round(3)
)


# =========================================================
# 6. Risk distribution
# =========================================================

print("\n" + "-" * 40)
print("RISK DISTRIBUTION")
print("-" * 40)

print(
    df["Risk_Label"]
    .value_counts()
)


print("\nRisk percentages:")

print(
    (
        df["Risk_Label"]
        .value_counts(
            normalize=True
        )
        * 100
    )
    .round(2)
)


# =========================================================
# 7. Category check
# =========================================================

print("\n" + "-" * 40)
print("CATEGORICAL FEATURES")
print("-" * 40)

categorical_columns = [
    "Product_Category",
    "Shipping_Mode",
    "Market",
    "Order_Region",
    "Customer_Segment",
]


for column in categorical_columns:

    print(
        f"{column}: "
        f"{df[column].nunique()} unique values"
    )


# =========================================================
# 8. Date check
# =========================================================

print("\n" + "-" * 40)
print("DATE CHECK")
print("-" * 40)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

print(
    f"Minimum date: {df['Date'].min()}"
)

print(
    f"Maximum date: {df['Date'].max()}"
)

print(
    f"Invalid dates: "
    f"{df['Date'].isna().sum()}"
)


# =========================================================
# 9. Supply chain relationship
# =========================================================

print("\n" + "-" * 40)
print("SUPPLY CHAIN ORDER RELATIONSHIP")
print("-" * 40)

print(
    df[
        [
            "Consumer_Demand",
            "Retailer_Order",
            "Wholesaler_Order",
            "Manufacturer_Order",
        ]
    ]
    .mean()
    .round(3)
)


# =========================================================
# 10. Correlation
# =========================================================

print("\n" + "-" * 40)
print("IMPORTANT FEATURE CORRELATIONS")
print("-" * 40)

correlation_columns = [
    "Consumer_Demand",
    "Retailer_Order",
    "Wholesaler_Order",
    "Manufacturer_Order",
    "Bullwhip_Ratio",
    "Lead_Time_Days",
    "Supply_Disruption_Rate",
    "Inventory_Level",
    "Product_Price",
    "Risk_Score",
]


correlation_columns = [
    column
    for column in correlation_columns
    if column in df.columns
]


correlation = (
    df[correlation_columns]
    .corr()
)


print(
    correlation
    .round(2)
    .to_string()
)


# =========================================================
# 11. ML target check
# =========================================================

print("\n" + "-" * 40)
print("ML TARGET CHECK")
print("-" * 40)

print(
    "Risk_Label values:"
)

print(
    df["Risk_Label"]
    .unique()
)

print(
    "\nRisk_Level_Num values:"
)

print(
    sorted(
        df["Risk_Level_Num"]
        .unique()
    )
)


# =========================================================
# Final
# =========================================================

print("\n" + "=" * 65)
print("FINAL DATASET VALIDATION COMPLETED")
print("=" * 65)