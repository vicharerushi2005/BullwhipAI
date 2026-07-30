"""
BullwhipAI
Machine Learning Training Agent
"""

import json
import joblib
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.preprocessing import prepare_features

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "datasets" / "processed" / "featured_supply_chain.csv"

MODEL_DIR = BASE_DIR / "datasets" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "risk_prediction_model.pkl"
ENCODER_FILE = MODEL_DIR / "label_encoder.pkl"
FEATURE_FILE = MODEL_DIR / "feature_columns.pkl"
METADATA_FILE = MODEL_DIR / "model_metadata.json"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\nLoading processed dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows Loaded : {len(df)}")

# --------------------------------------------------
# PREPROCESS
# --------------------------------------------------

X, y = prepare_features(df, training=True)

# --------------------------------------------------
# ENCODE TARGET
# --------------------------------------------------

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

joblib.dump(label_encoder, ENCODER_FILE)

# --------------------------------------------------
# SAVE FEATURE LIST
# --------------------------------------------------

joblib.dump(list(X.columns), FEATURE_FILE)

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

print("Training Complete.")

# --------------------------------------------------
# PREDICT
# --------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n===============================")
print("MODEL PERFORMANCE")
print("===============================")

print(f"\nAccuracy : {accuracy * 100:.2f}%")

print("\nClassification Report\n")

print(classification_report(
    y_test,
    predictions,
    target_names=label_encoder.classes_,
    zero_division=0
))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, predictions))

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features\n")
print(importance.head(10))

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

joblib.dump(model, MODEL_FILE)

# --------------------------------------------------
# SAVE METADATA
# --------------------------------------------------

metadata = {
    "algorithm": "RandomForestClassifier",
    "training_rows": len(df),
    "features": len(X.columns),
    "feature_names": list(X.columns),
    "accuracy": round(float(accuracy), 4)
}

with open(METADATA_FILE, "w") as f:
    json.dump(metadata, f, indent=4)

print("\n===============================")
print("MODEL ARTIFACTS SAVED")
print("===============================")

print("✓ Model")
print(MODEL_FILE)

print("\n✓ Label Encoder")
print(ENCODER_FILE)

print("\n✓ Feature Columns")
print(FEATURE_FILE)

print("\n✓ Metadata")
print(METADATA_FILE)