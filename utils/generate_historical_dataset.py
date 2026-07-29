import pandas as pd
import random
import os
from datetime import datetime, timedelta
from pathlib import Path

# -------------------------------
# CONFIGURATION
# -------------------------------

NUM_ROWS = 100

START_DATE = datetime(2025, 1, 1)

PRODUCTS = [
    "Onion",
    "Tomato",
    "Potato",
    "Rice",
    "Wheat",
    "Milk",
    "Sugar",
    "Apple",
    "Banana",
    "Cooking Oil"
]

CITIES = [
    ("Mumbai", "Maharashtra"),
    ("Pune", "Maharashtra"),
    ("Nashik", "Maharashtra"),
    ("Delhi", "Delhi"),
    ("Bengaluru", "Karnataka"),
    ("Chennai", "Tamil Nadu"),
    ("Hyderabad", "Telangana"),
    ("Kolkata", "West Bengal")
]

SENTIMENTS = [
    "Positive",
    "Neutral",
    "Negative"
]

COUNTRY = "India"

# -------------------------------
# RISK CALCULATION
# -------------------------------

def calculate_risk(
    rainfall,
    commodity_price,
    demand,
    inventory,
    transport_delay,
    festival,
    government_alert
):

    score = 0

    if rainfall > 120:
        score += 2

    if commodity_price > 80:
        score += 2

    if demand > inventory:
        score += 3

    if transport_delay > 5:
        score += 2

    if festival == 1:
        score += 1

    if government_alert == 1:
        score += 2

    if score <= 2:
        return "Low"

    elif score <= 5:
        return "Medium"

    return "High"


# -------------------------------
# GENERATE DATA
# -------------------------------

rows = []

for i in range(NUM_ROWS):

    date = START_DATE + timedelta(days=i)

    city, state = random.choice(CITIES)
    product = random.choice(PRODUCTS)

    temperature = round(random.uniform(18, 42), 1)
    humidity = random.randint(35, 95)
    rainfall = random.randint(0, 220)
    windspeed = round(random.uniform(2, 30), 1)

    commodity_price = random.randint(20, 120)
    fuel_price = round(random.uniform(90, 110), 2)

    demand = random.randint(500, 3000)
    inventory = random.randint(500, 3000)

    lead_time = random.randint(1, 10)
    transport_delay = random.randint(0, 10)

    festival = random.choice([0, 1])
    holiday = random.choice([0, 1])

    sentiment = random.choice(SENTIMENTS)

    government_alert = random.choice([0, 1])
    port_delay = random.choice([0, 1])
    railway_delay = random.choice([0, 1])

    risk = calculate_risk(
        rainfall,
        commodity_price,
        demand,
        inventory,
        transport_delay,
        festival,
        government_alert
    )

    rows.append([
        date.strftime("%Y-%m-%d"),
        city,
        state,
        COUNTRY,
        product,
        temperature,
        humidity,
        rainfall,
        windspeed,
        commodity_price,
        fuel_price,
        demand,
        inventory,
        lead_time,
        transport_delay,
        festival,
        holiday,
        sentiment,
        government_alert,
        port_delay,
        railway_delay,
        risk
    ])

# -------------------------------
# CREATE DATAFRAME
# -------------------------------

columns = [
    "Date",
    "City",
    "State",
    "Country",
    "Product",
    "Temperature",
    "Humidity",
    "Rainfall",
    "WindSpeed",
    "CommodityPrice",
    "FuelPrice",
    "Demand",
    "Inventory",
    "LeadTime",
    "TransportDelay",
    "Festival",
    "Holiday",
    "MarketSentiment",
    "GovernmentAlert",
    "PortDelay",
    "RailwayDelay",
    "RiskLevel"
]

df = pd.DataFrame(rows, columns=columns)

# -------------------------------
# SAVE DATASET
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

output_dir = BASE_DIR / "datasets" / "historical"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "historical_supply_chain.csv"

# Delete existing file
if output_path.exists():
    output_path.unlink()

# Save CSV
df.to_csv(output_path, index=False)

# -------------------------------
# VERIFY FILE
# -------------------------------

print("\n========== DATAFRAME ==========")
print(df.head())

print("\nRows in DataFrame:", len(df))

print("\n========== FILE LOCATION ==========")
print(output_path.resolve())

print("\n========== VERIFY SAVED FILE ==========")

df_check = pd.read_csv(output_path)

print(df_check.head())
print("\nRows in Saved File:", len(df_check))

print("\nColumns:", len(df_check.columns))

print("\nFile Size:", os.path.getsize(output_path), "bytes")

print("\nDataset generated successfully!")