import pandas as pd
import os
from datetime import datetime


print("Starting Product Risk Agent...")


# ---------------------------------
# File Paths
# ---------------------------------

risk_file = "data/risk_score.csv"
event_file = "data/market_events.csv"


# ---------------------------------
# Check Required Files
# ---------------------------------

if not os.path.exists(risk_file):

    print("Risk score file missing")
    exit()


if not os.path.exists(event_file):

    print("Market events file missing")
    exit()



# ---------------------------------
# Read Data
# ---------------------------------

risk = pd.read_csv(risk_file)

events = pd.read_csv(event_file)



# ---------------------------------
# Get Overall Risk Score
# ---------------------------------

overall_risk = float(
    risk["Overall Score"].iloc[0]
)



# ---------------------------------
# Risk Classification
# ---------------------------------

if overall_risk >= 70:

    risk_level = "HIGH"


elif overall_risk >= 30:

    risk_level = "MEDIUM"


else:

    risk_level = "LOW"



# ---------------------------------
# Products To Monitor
# ---------------------------------

products = [

    "Rice",
    "Tomato",
    "Onion",
    "Wheat",
    "Milk",
    "Sugar",
    "Processed Food"

]



results = []



# ---------------------------------
# Convert Events Into Text
# Fix NaN Issue
# ---------------------------------

event_text = " ".join(

    events.fillna("")
          .astype(str)
          .values
          .flatten()

)



# ---------------------------------
# Analyze Each Product
# ---------------------------------

for product in products:


    reason = "Normal market conditions"



    if "Commodity" in event_text or "Price" in event_text:

        reason = "Commodity price fluctuation detected"



    if "Weather" in event_text:

        reason += ", Weather impact detected"



    if "Market News" in event_text:

        reason += ", Market news impact detected"



    results.append({

        "Date":
            datetime.now().strftime("%Y-%m-%d"),


        "Product":
            product,


        "Risk Level":
            risk_level,


        "Risk Score":
            overall_risk,


        "Reason":
            reason

    })



# ---------------------------------
# Create DataFrame
# ---------------------------------

product_risk = pd.DataFrame(results)



# ---------------------------------
# Save Output File
# ---------------------------------

os.makedirs(

    "data",

    exist_ok=True

)



product_risk.to_csv(

    "data/product_risk.csv",

    index=False

)



# ---------------------------------
# Output
# ---------------------------------

print("\nProduct Risk Generated Successfully")

print("------------------------------------")

print(product_risk)