import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "kaggle"
    / "DataCoSupplyChainDataset.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "dataco_supply_chain_clean.csv"
)


# ---------------------------------------------------------
# Columns we need
# ---------------------------------------------------------

SELECTED_COLUMNS = [
    "order date (DateOrders)",
    "Category Name",
    "Product Name",
    "Order Item Quantity",
    "Product Price",
    "Order Item Product Price",
    "Sales",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Total",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Late_delivery_risk",
    "Shipping Mode",
    "Market",
    "Order Region",
    "Customer Segment",
    "Order Status",
    "Delivery Status",
]


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

print("Loading DataCo dataset...")

df = pd.read_csv(
    INPUT_FILE,
    encoding="latin-1"
)

print(f"Original dataset shape: {df.shape}")


# ---------------------------------------------------------
# Check required columns
# ---------------------------------------------------------

missing_columns = [
    column
    for column in SELECTED_COLUMNS
    if column not in df.columns
]

if missing_columns:

    print("\nERROR: Missing columns:")

    for column in missing_columns:
        print(f" - {column}")

    raise ValueError(
        "Some required columns are missing."
    )


# ---------------------------------------------------------
# Keep required columns
# ---------------------------------------------------------

df = df[SELECTED_COLUMNS].copy()


# ---------------------------------------------------------
# Rename columns
# ---------------------------------------------------------

df.rename(
    columns={
        "order date (DateOrders)": "Order_Date",
        "Category Name": "Product_Category",
        "Product Name": "Product_Name",
        "Order Item Quantity": "Order_Quantity",
        "Product Price": "Product_Price",
        "Order Item Product Price": "Order_Item_Price",
        "Sales": "Sales",
        "Order Item Discount": "Discount",
        "Order Item Discount Rate": "Discount_Rate",
        "Order Item Total": "Order_Total",
        "Days for shipping (real)": "Actual_Shipping_Days",
        "Days for shipment (scheduled)": "Scheduled_Shipping_Days",
        "Late_delivery_risk": "Late_Delivery_Risk",
        "Shipping Mode": "Shipping_Mode",
        "Market": "Market",
        "Order Region": "Order_Region",
        "Customer Segment": "Customer_Segment",
        "Order Status": "Order_Status",
        "Delivery Status": "Delivery_Status",
    },
    inplace=True,
)


# ---------------------------------------------------------
# Convert date
# ---------------------------------------------------------

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)


# ---------------------------------------------------------
# Convert numeric columns
# ---------------------------------------------------------

numeric_columns = [
    "Order_Quantity",
    "Product_Price",
    "Order_Item_Price",
    "Sales",
    "Discount",
    "Discount_Rate",
    "Order_Total",
    "Actual_Shipping_Days",
    "Scheduled_Shipping_Days",
    "Late_Delivery_Risk",
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ---------------------------------------------------------
# Remove invalid rows
# ---------------------------------------------------------

print("\nMissing values before cleaning:")

print(
    df.isnull()
    .sum()
    .sort_values(ascending=False)
)


df.dropna(
    subset=[
        "Order_Date",
        "Order_Quantity",
        "Product_Price",
        "Order_Item_Price",
        "Sales",
        "Actual_Shipping_Days",
    ],
    inplace=True
)


# ---------------------------------------------------------
# Remove impossible values
# ---------------------------------------------------------

df = df[
    (df["Order_Quantity"] > 0)
    & (df["Product_Price"] >= 0)
    & (df["Order_Item_Price"] >= 0)
    & (df["Sales"] >= 0)
    & (df["Actual_Shipping_Days"] >= 0)
    & (df["Scheduled_Shipping_Days"] >= 0)
]


# ---------------------------------------------------------
# Remove duplicate rows
# ---------------------------------------------------------

before_duplicates = len(df)

df.drop_duplicates(
    inplace=True
)

after_duplicates = len(df)

print(
    f"\nRemoved duplicates: "
    f"{before_duplicates - after_duplicates}"
)


# ---------------------------------------------------------
# Sort by date
# ---------------------------------------------------------

df.sort_values(
    by="Order_Date",
    inplace=True
)

df.reset_index(
    drop=True,
    inplace=True
)


# ---------------------------------------------------------
# Create additional features
# ---------------------------------------------------------

df["Shipping_Delay_Days"] = (
    df["Actual_Shipping_Days"]
    - df["Scheduled_Shipping_Days"]
)


df["Revenue_Per_Item"] = (
    df["Sales"]
    / df["Order_Quantity"]
).replace(
    [float("inf"), -float("inf")],
    0
)


df["Discount_Amount_Per_Item"] = (
    df["Discount"]
    / df["Order_Quantity"]
).replace(
    [float("inf"), -float("inf")],
    0
)


# ---------------------------------------------------------
# Create output directory
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Save cleaned dataset
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Final information
# ---------------------------------------------------------

print("\n========================================")
print("PREPROCESSING COMPLETED")
print("========================================")

print(
    f"Final dataset shape: {df.shape}"
)

print("\nColumns:")

for column in df.columns:
    print(f" - {column}")


print("\nOutput file:")

print(OUTPUT_FILE)


print("\nFirst 5 rows:")

print(df.head())