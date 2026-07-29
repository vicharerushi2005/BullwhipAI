"""
BullwhipAI
Machine Learning Training Agent

Purpose:
Train a Random Forest model to predict Bullwhip Risk.
"""

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

MODEL_DIR = (
    BASE_DIR /
    "datasets" /
    "models"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "risk_prediction_model.pkl"

ENCODER_FILE = MODEL_DIR / "label_encoder.pkl"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\nLoading processed dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows Loaded : {len(df)}")

# --------------------------------------------------
# DROP UNUSED COLUMNS
# --------------------------------------------------

drop_columns = [
    "Date",
    "City",
    "State",
    "Country",
    "Product"
]

df = df.drop(columns=drop_columns)

# --------------------------------------------------
# ENCODE MARKET SENTIMENT
# --------------------------------------------------

sentiment_map = {
    "Positive": 0,
    "Neutral": 1,
    "Negative": 2
}

df["MarketSentiment"] = df["MarketSentiment"].map(sentiment_map)

# --------------------------------------------------
# ENCODE TARGET
# --------------------------------------------------

label_encoder = LabelEncoder()

df["RiskLevel"] = label_encoder.fit_transform(df["RiskLevel"])

# Save encoder
joblib.dump(label_encoder, ENCODER_FILE)

# --------------------------------------------------
# SPLIT FEATURES / TARGET
# --------------------------------------------------

X = df.drop(columns=["RiskLevel"])

y = df["RiskLevel"]

# --------------------------------------------------
# TRAIN TEST SPLIT
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
# MODEL
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
# PREDICTION
# --------------------------------------------------

predictions = model.predict(X_test)

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\n===============================")
print("MODEL PERFORMANCE")
print("===============================")

print(f"\nAccuracy : {accuracy*100:.2f}%")

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_
    )
)

print("Confusion Matrix\n")

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

print("\nModel Saved Successfully!")

print(MODEL_FILE)