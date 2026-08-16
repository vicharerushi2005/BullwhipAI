"""
BullwhipAI - ML Model Trainer v3.0
===================================

ML pipeline for:

Kaggle DataCo
      ↓
SDV Synthetic Data
      ↓
Bullwhip Simulation
      ↓
Risk Classification

Target:
    Risk_Level_Num

Classes:
    0 = LOW
    1 = MEDIUM
    2 = HIGH

Important:
    Risk_Score and Risk_Level_Num are NOT used as input
    features to prevent target leakage.
"""


import pandas as pd
import numpy as np
import json
import joblib
import os

from datetime import datetime

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = (
    "data/final/augmented_supply_chain.csv"
)

MODEL_DIR = "models"


# =========================================================
# HEADER
# =========================================================

print("=" * 65)
print("BULLWHIP AI - MODEL TRAINER v3.0")
print("=" * 65)


# =========================================================
# 1. LOAD DATA
# =========================================================

print("\nLoading final augmented dataset...")

if not os.path.exists(DATA_PATH):

    print(
        "\nERROR: Final augmented dataset not found."
    )

    print(
        "Run:"
    )

    print(
        "python scripts/apply_bullwhip_simulation.py"
    )

    exit(1)


df = pd.read_csv(DATA_PATH)


print(
    f"\nLoaded {len(df)} records"
)

print(
    f"Columns: {len(df.columns)}"
)

print(
    f"Date range: "
    f"{df['Date'].min()} "
    f"to "
    f"{df['Date'].max()}"
)


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

print("\nClass distribution:")

print(
    df["Risk_Label"]
    .value_counts()
)


# =========================================================
# 2. DATE PROCESSING
# =========================================================

print(
    "\nProcessing dates..."
)


df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


df = (
    df
    .dropna(
        subset=["Date"]
    )
    .sort_values("Date")
    .reset_index(drop=True)
)


# =========================================================
# 3. TARGET VALIDATION
# =========================================================

print(
    "\nChecking target classes..."
)


required_classes = {
    0,
    1,
    2
}


present_classes = set(
    df["Risk_Level_Num"]
    .dropna()
    .astype(int)
    .unique()
)


print(
    f"Present classes: {sorted(present_classes)}"
)


if not required_classes.issubset(
    present_classes
):

    print(
        "\nWARNING: Not all three classes are present."
    )

    print(
        "ML training requires LOW, MEDIUM and HIGH."
    )


# =========================================================
# 4. FEATURE ENGINEERING
# =========================================================

print(
    "\nEngineering features..."
)


# ---------------------------------------------------------
# Product category encoding
# ---------------------------------------------------------
#
# The new dataset contains Product_Category rather than
# the old Product field.
#
# We use one-hot encoding later rather than assigning
# arbitrary numeric values to categories.
# ---------------------------------------------------------

categorical_columns = [
    "Product_Category",
    "Shipping_Mode",
    "Market",
    "Order_Region",
    "Customer_Segment",
]


# ---------------------------------------------------------
# Rolling demand features
# ---------------------------------------------------------

df["Demand_7d_Mean"] = (
    df["Consumer_Demand"]
    .rolling(
        7,
        min_periods=1
    )
    .mean()
)


df["Demand_7d_Std"] = (
    df["Consumer_Demand"]
    .rolling(
        7,
        min_periods=1
    )
    .std()
    .fillna(0)
)


# ---------------------------------------------------------
# Rolling Bullwhip features
# ---------------------------------------------------------

df["Bullwhip_7d_Mean"] = (
    df["Bullwhip_Ratio"]
    .rolling(
        7,
        min_periods=1
    )
    .mean()
)


df["Bullwhip_7d_Std"] = (
    df["Bullwhip_Ratio"]
    .rolling(
        7,
        min_periods=1
    )
    .std()
    .fillna(0)
)


# ---------------------------------------------------------
# Rolling disruption
# ---------------------------------------------------------

df["Disruption_7d"] = (
    df["Supply_Disruption"]
    .rolling(
        7,
        min_periods=1
    )
    .sum()
)


df["Disruption_Rate_7d"] = (
    df["Supply_Disruption_Rate"]
    .rolling(
        7,
        min_periods=1
    )
    .mean()
)


# ---------------------------------------------------------
# Rolling lead time
# ---------------------------------------------------------

df["LeadTime_7d_Mean"] = (
    df["Lead_Time_Days"]
    .rolling(
        7,
        min_periods=1
    )
    .mean()
)


df["LeadTime_7d_Max"] = (
    df["Lead_Time_Days"]
    .rolling(
        7,
        min_periods=1
    )
    .max()
)


# ---------------------------------------------------------
# Price features
# ---------------------------------------------------------

df["Price_7d_Mean"] = (
    df["Product_Price"]
    .rolling(
        7,
        min_periods=1
    )
    .mean()
)


df["Price_Change_Pct"] = (
    df["Product_Price"]
    .pct_change(
        7
    )
    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )
    .fillna(0)
    * 100
)


# ---------------------------------------------------------
# Inventory features
# ---------------------------------------------------------

df["Inventory_7d_Mean"] = (
    df["Inventory_Level"]
    .rolling(
        7,
        min_periods=1
    )
    .mean()
)


df["Inventory_7d_Std"] = (
    df["Inventory_Level"]
    .rolling(
        7,
        min_periods=1
    )
    .std()
    .fillna(0)
)


# ---------------------------------------------------------
# Supply-chain amplification features
# ---------------------------------------------------------

df["Wholesale_Amplification"] = (

    df["Wholesaler_Order"]

    / df["Retailer_Order"]
    .clip(lower=0.01)
)


df["Manufacturer_Amplification"] = (

    df["Manufacturer_Order"]

    / df["Wholesaler_Order"]
    .clip(lower=0.01)
)


# ---------------------------------------------------------
# Order variability
# ---------------------------------------------------------

df["Manufacturer_Order_7d_Std"] = (
    df["Manufacturer_Order"]
    .rolling(
        7,
        min_periods=1
    )
    .std()
    .fillna(0)
)


# =========================================================
# 5. SEASONAL FEATURES
# =========================================================

def get_season(month):

    if month in [12, 1, 2]:

        return 0

    if month in [3, 4, 5]:

        return 1

    if month in [6, 7, 8, 9]:

        return 2

    return 3


df["Season"] = (
    df["Month"]
    .apply(get_season)
)


df["Is_Festival"] = (
    df["Month"]
    .isin(
        [10, 11]
    )
    .astype(int)
)


df["Is_Monsoon"] = (
    df["Month"]
    .isin(
        [6, 7, 8, 9]
    )
    .astype(int)
)


# =========================================================
# 6. CREATE DUMMY VARIABLES
# =========================================================

print(
    "\nEncoding categorical features..."
)


existing_categorical = [
    column
    for column in categorical_columns
    if column in df.columns
]


df_model = pd.get_dummies(
    df,
    columns=existing_categorical,
    drop_first=False
)


# =========================================================
# 7. FEATURE LIST
# =========================================================
#
# IMPORTANT:
#
# We deliberately exclude:
#
# Risk_Score
# Risk_Label
# Risk_Level_Num
#
# because these are target-derived values.
#
# =========================================================


BASE_NUMERIC_FEATURES = [

    # Demand
    "Consumer_Demand",

    "Demand_7d_Mean",

    "Demand_7d_Std",

    # Supply-chain orders
    "Retailer_Order",

    "Wholesaler_Order",

    "Manufacturer_Order",

    # Amplification
    "Demand_Amplification",

    "Bullwhip_Ratio",

    "Bullwhip_7d_Mean",

    "Bullwhip_7d_Std",

    "Wholesale_Amplification",

    "Manufacturer_Amplification",

    "Manufacturer_Order_7d_Std",

    # Inventory
    "Inventory_Level",

    "Inventory_7d_Mean",

    "Inventory_7d_Std",

    # Supply disruption
    "Supply_Disruption",

    "Supply_Disruption_Rate",

    "Disruption_7d",

    "Disruption_Rate_7d",

    # Lead time
    "Lead_Time_Days",

    "LeadTime_7d_Mean",

    "LeadTime_7d_Max",

    # Pricing
    "Product_Price",

    "Price_Stress",

    "Price_7d_Mean",

    "Price_Change_Pct",

    # Sales
    "Sales",

    "Order_Total",

    "Discount_Rate",

    # Date
    "Month",

    "Day_of_Year",

    "Year",

    # Season
    "Season",

    "Is_Festival",

    "Is_Monsoon",
]


# ---------------------------------------------------------
# Add encoded categorical features
# ---------------------------------------------------------

categorical_feature_columns = [
    column
    for column in df_model.columns
    if any(
        column.startswith(
            category + "_"
        )
        for category in existing_categorical
    )
]


FEATURES = [
    column
    for column in (
        BASE_NUMERIC_FEATURES
        + categorical_feature_columns
    )
    if column in df_model.columns
]


TARGET = "Risk_Level_Num"


# =========================================================
# 8. PREPARE ML DATA
# =========================================================

df_model = (
    df_model[
        FEATURES
        + [TARGET]
    ]
    .dropna()
    .copy()
)


print(
    f"\nFeatures: {len(FEATURES)}"
)

print(
    f"Rows: {len(df_model)}"
)


print(
    "\nClass distribution after feature engineering:"
)


print(
    df_model[TARGET]
    .value_counts()
    .sort_index()
    .rename(
        {
            0: "LOW",
            1: "MEDIUM",
            2: "HIGH"
        }
    )
)


# =========================================================
# 9. TRAIN / TEST SPLIT
# =========================================================
#
# We use a chronological split rather than a random split.
#
# Earlier data → training
# Later data   → testing
#
# This is more appropriate for a supply-chain forecasting
# scenario because it prevents future observations from
# leaking into the training set.
#
# =========================================================

X = (
    df_model[
        FEATURES
    ]
    .values
)


y = (
    df_model[
        TARGET
    ]
    .astype(int)
    .values
)


split_idx = int(
    len(X)
    * 0.82
)


X_train = X[
    :split_idx
]

X_test = X[
    split_idx:
]


y_train = y[
    :split_idx
]

y_test = y[
    split_idx:
]


print(
    f"\nTrain records: {len(X_train)}"
)

print(
    f"Test records: {len(X_test)}"
)


print(
    f"Test classes: "
    f"{np.unique(y_test)}"
)


# =========================================================
# 10. SCALE FEATURES
# =========================================================

print(
    "\nScaling features..."
)


scaler = StandardScaler()


X_train_sc = (
    scaler.fit_transform(
        X_train
    )
)


X_test_sc = (
    scaler.transform(
        X_test
    )
)


# =========================================================
# 11. TRAIN RANDOM FOREST
# =========================================================

print(
    "\nTraining Random Forest..."
)


rf = RandomForestClassifier(

    n_estimators=200,

    max_depth=12,

    min_samples_split=5,

    min_samples_leaf=2,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1,
)


rf.fit(
    X_train_sc,
    y_train
)


# =========================================================
# 12. PREDICTIONS
# =========================================================

y_pred = (
    rf.predict(
        X_test_sc
    )
)


y_prob = (
    rf.predict_proba(
        X_test_sc
    )
)


# =========================================================
# 13. METRICS
# =========================================================

accuracy = (
    accuracy_score(
        y_test,
        y_pred
    )
)


f1 = (
    f1_score(
        y_test,
        y_pred,
        average="weighted"
    )
)


precision = (
    precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
)


recall = (
    recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
)


print(
    "\n" + "=" * 65
)

print(
    "MODEL PERFORMANCE"
)

print(
    "=" * 65
)


print(
    f"\nAccuracy : "
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)


print(
    f"Precision: "
    f"{precision:.4f}"
)


print(
    f"Recall   : "
    f"{recall:.4f}"
)


print(
    f"F1 Score : "
    f"{f1:.4f}"
)


# =========================================================
# 14. CLASSIFICATION REPORT
# =========================================================

name_map = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH"
}


classes_in_test = sorted(
    np.unique(
        np.concatenate(
            [
                y_test,
                y_pred
            ]
        )
    )
)


names_present = [
    name_map[c]
    for c in classes_in_test
]


print(
    "\nClassification Report:"
)


print(
    classification_report(

        y_test,

        y_pred,

        labels=classes_in_test,

        target_names=names_present,

        zero_division=0,
    )
)


# =========================================================
# 15. CONFUSION MATRIX
# =========================================================

print(
    "Confusion Matrix:"
)


cm = confusion_matrix(

    y_test,

    y_pred,

    labels=classes_in_test
)


cm_df = pd.DataFrame(

    cm,

    index=[
        f"Actual {name_map[c]}"
        for c in classes_in_test
    ],

    columns=[
        f"Pred {name_map[c]}"
        for c in classes_in_test
    ],
)


print(
    cm_df
)


# =========================================================
# 16. CROSS VALIDATION
# =========================================================
#
# Cross-validation is performed only on the training data.
#
# This prevents the test set from influencing model
# evaluation.
#
# =========================================================

print(
    "\nRunning 5-Fold Cross Validation..."
)


skf = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42
)


cv_scores = cross_val_score(

    rf,

    X_train_sc,

    y_train,

    cv=skf,

    scoring="accuracy"
)


print(
    f"\n5-Fold CV Accuracy: "
    f"{cv_scores.mean():.4f}"
    f" ± "
    f"{cv_scores.std():.4f}"
)


# =========================================================
# 17. FEATURE IMPORTANCE
# =========================================================

print(
    "\n" + "=" * 65
)

print(
    "TOP FEATURES"
)

print(
    "=" * 65
)


importance = (
    rf.feature_importances_
)


feat_imp = pd.DataFrame({

    "Feature":
        FEATURES,

    "Importance":
        importance,

    "Importance_Pct":
        (
            importance
            / importance.sum()
            * 100
        ).round(2),

})


feat_imp = (
    feat_imp
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


feat_imp["Rank"] = (
    range(
        1,
        len(feat_imp) + 1
    )
)


print(
    feat_imp
    .head(15)
    [
        [
            "Rank",
            "Feature",
            "Importance_Pct"
        ]
    ]
    .to_string(
        index=False
    )
)


# =========================================================
# 18. SAVE MODEL
# =========================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


joblib.dump(
    rf,
    "models/bullwhip_model.pkl"
)


joblib.dump(
    scaler,
    "models/feature_scaler.pkl"
)


joblib.dump(
    FEATURES,
    "models/feature_names.pkl"
)


feat_imp.to_csv(
    "models/feature_importance.csv",
    index=False
)


# =========================================================
# 19. SAVE METRICS
# =========================================================

metrics = {

    "trained_on":
        datetime.now().strftime(
            "%Y-%m-%d %H:%M"
        ),

    "algorithm":
        "RandomForestClassifier",

    "n_estimators":
        200,

    "max_depth":
        12,

    "train_records":
        int(len(X_train)),

    "test_records":
        int(len(X_test)),

    "accuracy":
        round(
            float(accuracy),
            4
        ),

    "precision_weighted":
        round(
            float(precision),
            4
        ),

    "recall_weighted":
        round(
            float(recall),
            4
        ),

    "f1_weighted":
        round(
            float(f1),
            4
        ),

    "cv_mean":
        round(
            float(
                cv_scores.mean()
            ),
            4
        ),

    "cv_std":
        round(
            float(
                cv_scores.std()
            ),
            4
        ),

    "features":
        FEATURES,

    "n_features":
        len(FEATURES),

    "classes":
        [
            "LOW",
            "MEDIUM",
            "HIGH"
        ],

    "class_distribution":
        df[
            "Risk_Label"
        ]
        .value_counts()
        .to_dict(),

}


with open(
    "models/model_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )


# =========================================================
# 20. FINAL OUTPUT
# =========================================================

print(
    "\n" + "=" * 65
)

print(
    "MODEL TRAINING COMPLETED"
)

print(
    "=" * 65
)


print(
    "\nModels saved:"
)

print(
    "  models/bullwhip_model.pkl"
)

print(
    "  models/feature_scaler.pkl"
)

print(
    "  models/feature_names.pkl"
)

print(
    "  models/feature_importance.csv"
)

print(
    "  models/model_metrics.json"
)


print(
    "\n🎓 BullwhipAI ML training complete!"
)