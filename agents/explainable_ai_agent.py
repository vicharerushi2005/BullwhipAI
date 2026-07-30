"""
BullwhipAI
Explainable AI Agent (Version 2)

Uses:
• Shared preprocessing pipeline
• Shared model loader
• SHAP
• Business Knowledge Base

Output:
datasets/explanations/explanation.json
"""

import json
from pathlib import Path

import pandas as pd
import shap

from utils.preprocessing import prepare_features
from utils.model_utils import load_model
from config.business_rules import FEATURE_EXPLANATIONS

# ----------------------------------------------------
# PATHS
# ----------------------------------------------------

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
    "explanations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "explanation.json"

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------

print("=" * 60)
print("BullwhipAI - Explainable AI Agent")
print("=" * 60)

artifacts = load_model()

model = artifacts["model"]

encoder = artifacts["encoder"]

feature_columns = artifacts["features"]

metadata = artifacts["metadata"]

print("\n✓ Model Loaded")

print("✓ Model Accuracy :", metadata["accuracy"])

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\n✓ Dataset Loaded :", len(df))

# ----------------------------------------------------
# PREPROCESS
# ----------------------------------------------------

X = prepare_features(df)

# Ensure feature order matches training

X = X[feature_columns]

# Latest record

sample = X.iloc[[-1]]

print("\n✓ Latest Record Selected")

# ----------------------------------------------------
# PREDICTION
# ----------------------------------------------------

prediction_encoded = model.predict(sample)[0]

prediction = encoder.inverse_transform([prediction_encoded])[0]

probabilities = model.predict_proba(sample)[0]

confidence = round(max(probabilities) * 100, 2)

print("\nPrediction :", prediction)

print("Confidence :", confidence, "%")

# ----------------------------------------------------
# SHAP
# ----------------------------------------------------

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    sample,
    check_additivity=False
)

import numpy as np

print("\n========== SHAP DEBUG ==========")
print("Type:", type(shap_values))

if isinstance(shap_values, list):
    print("List Length:", len(shap_values))
    for i, arr in enumerate(shap_values):
        print(f"Class {i} Shape:", np.array(arr).shape)
else:
    print("Shape:", np.array(shap_values).shape)

print("===============================\n")

# Handle multiclass

# ----------------------------------------------------
# EXTRACT SHAP VALUES
# ----------------------------------------------------

import numpy as np

# New SHAP versions return:
# (samples, features, classes)

if isinstance(shap_values, list):

    # Older SHAP versions
    values = np.array(
        shap_values[prediction_encoded][0]
    )

else:

    arr = np.array(shap_values)

    if arr.ndim == 3:
        # (1, features, classes)
        values = arr[0, :, prediction_encoded]

    elif arr.ndim == 2:
        # (1, features)
        values = arr[0]

    else:
        raise ValueError(
            f"Unsupported SHAP shape: {arr.shape}"
        )

# Make absolutely sure it's 1D
values = values.flatten()

print("\nSHAP Vector Shape :", values.shape)
print("Feature Count     :", len(feature_columns))

# ----------------------------------------------------
# TOP FEATURES
# ----------------------------------------------------

import numpy as np

print("\n========== VALUES DEBUG ==========")
print("type(values):", type(values))
print("shape(values):", np.shape(values))
print("values:")
print(values)
print("feature_columns length:", len(feature_columns))
print("=================================\n")

importance = pd.DataFrame({

    "Feature": feature_columns,

    "Impact": np.abs(values)

})

print(importance.head())

importance = importance.sort_values(

    by="Impact",

    ascending=False

)

top5 = importance.head(5)

# ----------------------------------------------------
# BUSINESS EXPLANATIONS
# ----------------------------------------------------

results = []

for _, row in top5.iterrows():

    feature = row["Feature"]

    info = FEATURE_EXPLANATIONS.get(

        feature,

        {

            "title": feature,

            "reason": "No explanation available.",

            "impact": "Unknown"

        }

    )

    results.append({

        "feature": feature,

        "importance": round(float(row["Impact"]), 4),

        "title": info["title"],

        "reason": info["reason"],

        "impact": info["impact"]

    })

# ----------------------------------------------------
# SAVE JSON
# ----------------------------------------------------

output = {

    "prediction": prediction,

    "confidence": confidence,

    "accuracy": metadata["accuracy"],

    "top_factors": results

}

with open(OUTPUT_FILE, "w") as file:

    json.dump(output, file, indent=4)

print("\n===================================")

print("EXPLAINABLE AI COMPLETE")

print("===================================")

print("\nPrediction :", prediction)

print("Confidence :", confidence)

print("\nTop Business Factors:\n")

for factor in results:

    print(f"• {factor['title']}")

print("\nSaved to:")

print(OUTPUT_FILE)