import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

ORIGINAL_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "dataco_supply_chain_clean.csv"
)

SYNTHETIC_FILE = (
    BASE_DIR
    / "data"
    / "synthetic"
    / "synthetic_supply_chain.csv"
)


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

print("Loading datasets...")

original = pd.read_csv(
    ORIGINAL_FILE
)

synthetic = pd.read_csv(
    SYNTHETIC_FILE
)


print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

print(
    f"Original dataset:  {original.shape}"
)

print(
    f"Synthetic dataset: {synthetic.shape}"
)


# ---------------------------------------------------------
# Numerical columns
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
    "Shipping_Delay_Days",
    "Revenue_Per_Item",
    "Discount_Amount_Per_Item",
]


print("\n========================================")
print("NUMERICAL COMPARISON")
print("========================================")


comparison = []

for column in numeric_columns:

    original_mean = original[column].mean()
    synthetic_mean = synthetic[column].mean()

    original_std = original[column].std()
    synthetic_std = synthetic[column].std()

    comparison.append({
        "Column": column,
        "Original Mean": original_mean,
        "Synthetic Mean": synthetic_mean,
        "Original Std": original_std,
        "Synthetic Std": synthetic_std,
    })


comparison_df = pd.DataFrame(
    comparison
)

print(
    comparison_df.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Categorical comparison
# ---------------------------------------------------------

categorical_columns = [
    "Product_Category",
    "Shipping_Mode",
    "Market",
    "Order_Region",
    "Customer_Segment",
    "Order_Status",
    "Delivery_Status",
]


print("\n========================================")
print("CATEGORICAL VALUE CHECK")
print("========================================")


for column in categorical_columns:

    print(f"\n--- {column} ---")

    original_values = set(
        original[column]
        .dropna()
        .unique()
    )

    synthetic_values = set(
        synthetic[column]
        .dropna()
        .unique()
    )

    print(
        "Original unique values:",
        len(original_values)
    )

    print(
        "Synthetic unique values:",
        len(synthetic_values)
    )

    unexpected = (
        synthetic_values
        - original_values
    )

    if unexpected:

        print(
            "Unexpected synthetic values:",
            unexpected
        )

    else:

        print(
            "No unexpected categorical values."
        )


# ---------------------------------------------------------
# Invalid value checks
# ---------------------------------------------------------

print("\n========================================")
print("INVALID VALUE CHECK")
print("========================================")


checks = {
    "Negative Quantity":
        (synthetic["Order_Quantity"] < 0).sum(),

    "Negative Product Price":
        (synthetic["Product_Price"] < 0).sum(),

    "Negative Sales":
        (synthetic["Sales"] < 0).sum(),

    "Negative Order Total":
        (synthetic["Order_Total"] < 0).sum(),

    "Negative Shipping Days":
        (
            synthetic["Actual_Shipping_Days"]
            < 0
        ).sum(),

    "Negative Scheduled Shipping":
        (
            synthetic["Scheduled_Shipping_Days"]
            < 0
        ).sum(),

    "Discount Rate Below 0":
        (
            synthetic["Discount_Rate"]
            < 0
        ).sum(),

    "Discount Rate Above 1":
        (
            synthetic["Discount_Rate"]
            > 1
        ).sum(),
}


for check, count in checks.items():

    print(
        f"{check}: {count}"
    )


# ---------------------------------------------------------
# Missing values
# ---------------------------------------------------------

print("\n========================================")
print("MISSING VALUE CHECK")
print("========================================")

missing = (
    synthetic
    .isnull()
    .sum()
)

print(
    missing[
        missing > 0
    ]
)


# ---------------------------------------------------------
# Final
# ---------------------------------------------------------

print("\n========================================")
print("VALIDATION COMPLETED")
print("========================================")