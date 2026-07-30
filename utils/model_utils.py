"""
BullwhipAI

Shared Model Loader

Used by:
    • Explainable AI Agent
    • Prediction Agent
    • Dashboard
"""

import json
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "datasets" / "models"

MODEL_FILE = MODEL_DIR / "risk_prediction_model.pkl"

ENCODER_FILE = MODEL_DIR / "label_encoder.pkl"

FEATURE_FILE = MODEL_DIR / "feature_columns.pkl"

METADATA_FILE = MODEL_DIR / "model_metadata.json"


def load_model():

    model = joblib.load(MODEL_FILE)

    encoder = joblib.load(ENCODER_FILE)

    features = joblib.load(FEATURE_FILE)

    with open(METADATA_FILE, "r") as f:

        metadata = json.load(f)

    return {

        "model": model,

        "encoder": encoder,

        "features": features,

        "metadata": metadata

    }