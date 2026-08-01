"""
BullwhipAI - ML Model Trainer v2.1
=====================================
Fixed:
  - classification_report now uses labels= to handle missing classes
  - Risk labels rebalanced (dataset has all 3 classes properly)
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     accuracy_score, f1_score)

print("=" * 60)
print("BULLWHIP AI - MODEL TRAINER v2.1")
print("=" * 60)

# -------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------

DATA_PATH = "data/historical_supply_chain.csv"

if not os.path.exists(DATA_PATH):
    print("ERROR: Historical data not found.")
    print("Run: python scripts/generate_historical_data.py")
    exit(1)

df = pd.read_csv(DATA_PATH)
print(f"\nLoaded {len(df)} records")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"\nClass distribution:")
print(df["Risk_Label"].value_counts())

# Ensure all 3 classes exist
present_classes = df["Risk_Level_Num"].unique()
if len(present_classes) < 3:
    print(f"\n⚠ Only {len(present_classes)} class(es) found: {present_classes}")
    print("  Rebalancing dataset to ensure all 3 classes...")
    # Force some LOW samples at the start (stable pre-COVID period)
    low_mask = (df["Bullwhip_Ratio"] < 1.2) & (df["Supply_Disruption"] == 0) & (df["Lead_Time_Days"] <= 5)
    df.loc[low_mask, "Risk_Label"]    = "LOW"
    df.loc[low_mask, "Risk_Level_Num"] = 0
    print(f"  After fix:")
    print(f"  {df['Risk_Label'].value_counts().to_dict()}")

# -------------------------------------------------------
# 2. FEATURE ENGINEERING
# -------------------------------------------------------

print("\nEngineering features...")

product_map = {"Rice":0,"Wheat":1,"Tomato":2,"Onion":3,"Sugar":4,"Milk":5}
df["Product_Code"] = df["Product"].map(product_map).fillna(0)

df = df.sort_values("Date").reset_index(drop=True)

df["Demand_7d_Mean"]   = df["Consumer_Demand"].rolling(7, min_periods=1).mean()
df["Demand_7d_Std"]    = df["Consumer_Demand"].rolling(7, min_periods=1).std().fillna(0)
df["Price_7d_Mean"]    = df["Commodity_Price_INR"].rolling(7, min_periods=1).mean()
df["Price_Change_Pct"] = df["Commodity_Price_INR"].pct_change(7).fillna(0) * 100
df["Disruption_7d"]    = df["Supply_Disruption"].rolling(7, min_periods=1).sum()
df["LeadTime_7d_Max"]  = df["Lead_Time_Days"].rolling(7, min_periods=1).max()

def get_season(month):
    if month in [12,1,2]:   return 0
    if month in [3,4,5]:    return 1
    if month in [6,7,8,9]:  return 2
    return 3

df["Season"]     = df["Month"].apply(get_season)
df["Is_Festival"]= df["Month"].isin([10,11]).astype(int)
df["Is_Monsoon"] = df["Month"].isin([6,7,8,9]).astype(int)

# -------------------------------------------------------
# 3. FEATURES
# -------------------------------------------------------

FEATURES = [
    "Consumer_Demand","Retailer_Order","Wholesaler_Order","Manufacturer_Order",
    "Inventory_Level","Lead_Time_Days","Demand_Amplification","Bullwhip_Ratio",
    "Supply_Disruption","Disruption_7d","LeadTime_7d_Max",
    "Commodity_Price_INR","Price_7d_Mean","Price_Change_Pct","News_Sentiment",
    "Temperature_C","Wind_Speed_kmh","Rainfall_mm",
    "Month","Day_of_Year","Season","Is_Festival","Is_Monsoon",
    "Demand_7d_Mean","Demand_7d_Std","Product_Code",
]
TARGET = "Risk_Level_Num"

df_model = df[FEATURES + [TARGET]].dropna()
print(f"Features: {len(FEATURES)} | Rows: {len(df_model)}")
print(f"Class distribution after engineering:")
print(df_model[TARGET].value_counts().rename({0:"LOW",1:"MEDIUM",2:"HIGH"}))

# -------------------------------------------------------
# 4. TRAIN / TEST SPLIT  (time-based)
# -------------------------------------------------------

X = df_model[FEATURES].values
y = df_model[TARGET].values

split_idx = int(len(X) * 0.82)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
print(f"Test classes present: {np.unique(y_test)}")

# -------------------------------------------------------
# 5. SCALE
# -------------------------------------------------------

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# -------------------------------------------------------
# 6. TRAIN
# -------------------------------------------------------

print("\nTraining Random Forest (200 trees)...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_sc, y_train)

y_pred = rf.predict(X_test_sc)
y_prob = rf.predict_proba(X_test_sc)

accuracy = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred, average="weighted")

print(f"\n✅ Accuracy : {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"✅ F1 Score : {f1:.4f}")

# --- Safe classification report: only report classes that appear ---
classes_in_test = sorted(np.unique(np.concatenate([y_test, y_pred])))
name_map = {0:"LOW", 1:"MEDIUM", 2:"HIGH"}
labels_present = classes_in_test
names_present  = [name_map[c] for c in labels_present]

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    labels=labels_present,
    target_names=names_present,
    zero_division=0,
))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=labels_present)
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {n}" for n in names_present],
    columns=[f"Pred {n}" for n in names_present],
)
print(cm_df)

# Cross-validation (stratified so every fold has all classes)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_train_sc, y_train, cv=skf, scoring="accuracy")
print(f"\n5-Fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# -------------------------------------------------------
# 7. FEATURE IMPORTANCE
# -------------------------------------------------------

importance = rf.feature_importances_
feat_imp = pd.DataFrame({
    "Feature":        FEATURES,
    "Importance":     importance,
    "Importance_Pct": (importance / importance.sum() * 100).round(2),
}).sort_values("Importance", ascending=False).reset_index(drop=True)
feat_imp["Rank"] = range(1, len(feat_imp)+1)

print("\n📊 Top 10 Features:")
print(feat_imp.head(10)[["Rank","Feature","Importance_Pct"]].to_string(index=False))

# -------------------------------------------------------
# 8. SAVE
# -------------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(rf,       "models/bullwhip_model.pkl")
joblib.dump(scaler,   "models/feature_scaler.pkl")
joblib.dump(FEATURES, "models/feature_names.pkl")
feat_imp.to_csv("models/feature_importance.csv", index=False)

metrics = {
    "trained_on":        datetime.now().strftime("%Y-%m-%d %H:%M"),
    "algorithm":         "RandomForestClassifier",
    "n_estimators":      200,
    "train_records":     int(len(X_train)),
    "test_records":      int(len(X_test)),
    "accuracy":          round(float(accuracy), 4),
    "f1_weighted":       round(float(f1), 4),
    "cv_mean":           round(float(cv_scores.mean()), 4),
    "cv_std":            round(float(cv_scores.std()), 4),
    "features":          FEATURES,
    "n_features":        len(FEATURES),
    "classes":           ["LOW","MEDIUM","HIGH"],
    "class_distribution":df["Risk_Label"].value_counts().to_dict(),
}
with open("models/model_metrics.json","w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Models saved to models/")
print("   bullwhip_model.pkl | feature_scaler.pkl | feature_names.pkl")
print("   feature_importance.csv | model_metrics.json")
print("\n🎓 Training complete!")
