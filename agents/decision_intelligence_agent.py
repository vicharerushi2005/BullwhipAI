"""
BullwhipAI
Decision Intelligence Agent

Purpose:
Convert AI predictions into real-world
business recommendations.

Output:
datasets/recommendations/recommendations.json
"""

import json
from pathlib import Path

import pandas as pd

from utils.preprocessing import prepare_features
from utils.model_utils import load_model
from config.decision_rules import DECISION_RULES

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
    "recommendations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "recommendations.json"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("=" * 60)
print("BullwhipAI - Decision Intelligence Agent")
print("=" * 60)

artifacts = load_model()

model = artifacts["model"]
encoder = artifacts["encoder"]
feature_columns = artifacts["features"]

print("\n✓ Model Loaded")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("✓ Dataset Loaded :", len(df))

latest_row = df.iloc[-1]

print("✓ Latest Record Selected")

# --------------------------------------------------
# PREDICT
# --------------------------------------------------

X = prepare_features(df)

X = X[feature_columns]

sample = X.iloc[[-1]]

prediction_encoded = model.predict(sample)[0]

prediction = encoder.inverse_transform(
    [prediction_encoded]
)[0]

confidence = round(
    model.predict_proba(sample).max() * 100,
    2
)

print("\nPrediction :", prediction)
print("Confidence :", confidence, "%")

# --------------------------------------------------
# DECISION ENGINE
# --------------------------------------------------

print("\nGenerating Recommendations...\n")

recommendations = []

for rule in DECISION_RULES:

    try:

        if rule["condition"](latest_row):

            recommendations.append({

                "priority": rule["priority"],

                "action": rule["action"],

                "reason": rule["reason"]

            })

    except Exception:

        pass

# --------------------------------------------------
# DEFAULT
# --------------------------------------------------

if len(recommendations) == 0:

    recommendations.append({

        "priority": "Low",

        "action":
        "Continue normal operations.",

        "reason":
        "Current supply chain appears stable."

    })

# --------------------------------------------------
# SORT PRIORITY
# --------------------------------------------------

priority_order = {

    "Critical": 0,

    "High": 1,

    "Medium": 2,

    "Low": 3

}

recommendations = sorted(

    recommendations,

    key=lambda x: priority_order[x["priority"]]

)

# --------------------------------------------------
# DASHBOARD OUTPUT
# --------------------------------------------------

output = {

    "prediction": prediction,

    "confidence": confidence,

    "recommendation_count": len(recommendations),

    "recommendations": recommendations

}

with open(OUTPUT_FILE, "w") as file:

    json.dump(output, file, indent=4)

# --------------------------------------------------
# TERMINAL OUTPUT
# --------------------------------------------------

print("=" * 60)
print("BUSINESS RECOMMENDATIONS")
print("=" * 60)

for i, rec in enumerate(recommendations, start=1):

    print(f"\n{i}. {rec['priority']} Priority")

    print("Action :")

    print(rec["action"])

    print("\nReason :")

    print(rec["reason"])

print("\nSaved To:")

print(OUTPUT_FILE)

print("\nDecision Intelligence Complete.")