import pandas as pd
from pathlib import Path

from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "dataco_supply_chain_clean.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "synthetic"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "synthetic_supply_chain.csv"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SYNTHETIC_ROWS = 10_000


# ---------------------------------------------------------
# Load cleaned dataset
# ---------------------------------------------------------

print("Loading cleaned DataCo dataset...")

df = pd.read_csv(
    INPUT_FILE
)

print(f"Input dataset shape: {df.shape}")


# ---------------------------------------------------------
# Convert date column
# ---------------------------------------------------------

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)


# ---------------------------------------------------------
# Remove rows with invalid dates
# ---------------------------------------------------------

df = df.dropna(
    subset=["Order_Date"]
).copy()


print(
    f"Dataset after date cleaning: {df.shape}"
)


# ---------------------------------------------------------
# Detect SDV metadata
# ---------------------------------------------------------

print("\nDetecting SDV metadata...")

metadata = Metadata.detect_from_dataframe(
    data=df,
    table_name="supply_chain"
)

print("SDV metadata detected successfully.")


# ---------------------------------------------------------
# Create SDV synthesizer
# ---------------------------------------------------------

print("\nCreating Gaussian Copula synthesizer...")

synthesizer = GaussianCopulaSynthesizer(
    metadata
)


# ---------------------------------------------------------
# Train SDV model
# ---------------------------------------------------------

print("\nTraining SDV model...")

synthesizer.fit(df)

print("SDV model training completed.")


# ---------------------------------------------------------
# Generate synthetic data
# ---------------------------------------------------------

print(
    f"\nGenerating {SYNTHETIC_ROWS:,} synthetic records..."
)

synthetic_df = synthesizer.sample(
    num_rows=SYNTHETIC_ROWS
)


# ---------------------------------------------------------
# Create output directory
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Save synthetic dataset
# ---------------------------------------------------------

synthetic_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n========================================")
print("SYNTHETIC DATA GENERATION COMPLETED")
print("========================================")

print(
    f"Synthetic dataset shape: "
    f"{synthetic_df.shape}"
)

print("\nOutput file:")

print(OUTPUT_FILE)

print("\nFirst 5 synthetic records:")

print(
    synthetic_df.head()
)

print("\nData types:")

print(
    synthetic_df.dtypes
)