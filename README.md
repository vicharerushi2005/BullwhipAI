# 📦 BullwhipAI v2.0
**Multi-Agent AI System for Bullwhip Effect Reduction in Food Supply Chains**

---

## 🚀 Quick Start

### First Time Setup (run once)
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install packages
pip install -r requirements.txt

# 3. Generate historical data (2020–2025) + Train ML model
python run_system.py --setup
```

### Every Day / Run Pipeline
```bash
python run_system.py
```

### Dashboard Only
```bash
streamlit run dashboard/app.py
```

---

## 🏗️ Architecture

```
News Agent ──┐
Weather Agent─┤
Commodity Agent─► Data Processing Agent ─► Risk Scoring Agent
                                                    │
                                          ML Prediction Agent  ← NEW ★
                                                    │
                                          XAI Explainer Engine ← NEW ★
                                                    │
                                          AI Narrative (Ollama)
                                                    │
                                              Dashboard v2.0
```

---

## ✨ New Features (v2.0)

### 1. Historical Dataset (2020–2025)
- 2,192 days of realistic food supply chain data
- Features: demand, orders at every supply tier, commodity prices, weather, disruptions
- Built-in COVID-era shocks, monsoon seasonality, festival demand spikes
- File: `data/historical_supply_chain.csv`

### 2. Machine Learning Model
- **Algorithm**: Random Forest Classifier (200 trees, balanced classes)
- **Accuracy**: ~98.7% on held-out 2025 data
- **26 features** including Bullwhip Ratio, demand amplification, price trends
- **Files**: `models/bullwhip_model.pkl`, `models/feature_scaler.pkl`

### 3. Explainable AI (XAI)
- **Method**: Perturbation-based Local Sensitivity (no paid SHAP library needed)
- For every prediction, shows **exactly which features caused it**
- Natural language explanation of the Bullwhip Effect status
- Team-specific action recommendations (Procurement, Inventory, Production, Logistics)
- **File**: `data/xai_explanation.json`

---

## 📁 Project Structure

```
BullwhipAI/
├── agents/
│   ├── data_processing_agent.py    # raw → market events
│   ├── risk_scoring_agent.py       # rule-based risk scores
│   ├── product_risk_agent.py       # per-product risk
│   └── ml_prediction_agent.py      # ★ ML + XAI prediction
├── ai_agent/
│   └── market_ai_agent.py          # Ollama LLM report
├── xai/
│   └── explainer.py                # ★ XAI engine
├── scripts/
│   ├── generate_historical_data.py # ★ dataset generator
│   └── train_model.py              # ★ model trainer
├── models/                         # ★ saved ML artifacts
│   ├── bullwhip_model.pkl
│   ├── feature_scaler.pkl
│   ├── feature_names.pkl
│   ├── feature_importance.csv
│   └── model_metrics.json
├── dashboard/
│   └── app.py                      # ★ v2.0 dashboard with XAI
├── data/
│   ├── historical_supply_chain.csv # ★ training dataset
│   ├── ml_prediction.csv           # ★ ML predictions log
│   ├── xai_explanation.json        # ★ latest XAI explanation
│   ├── xai_history.csv             # ★ XAI prediction history
│   ├── weather_data.csv
│   ├── commodity_data.csv
│   ├── news_data.csv
│   ├── market_events.csv
│   ├── risk_score.csv
│   └── product_risk.csv
├── weather/
│   └── weather_collector.py
├── commodity/
│   └── commodity_collector.py
├── scraper/
│   └── news_collector.py
├── run_system.py                   # master orchestrator
└── requirements.txt
```

---

## 🔍 What is Explainable AI (XAI)?

Traditional AI is a "black box" — it gives an answer but no reason.
BullwhipAI v2.0 uses **perturbation-based local sensitivity analysis**:

1. Take today's input features (price, weather, disruption, etc.)
2. Perturb each feature by ±1 standard deviation
3. Measure how much the prediction probability changes
4. Rank features by their local impact
5. Generate plain-English explanation + team-specific actions

**Example output:**
```
Prediction: 🔴 HIGH (82% confident)

Top Drivers:
  1. [34.2%] Supply_Disruption   → ⚠️ Active disruption detected
  2. [22.1%] Bullwhip_Ratio      → 🔴 Ratio 2.8 — severe amplification upstream
  3. [15.6%] Commodity_Price_INR → 🔴 Price ₹5,800 — significantly elevated
  4. [11.2%] Lead_Time_Days      → 🔴 Lead time 10 days — critically delayed
  5. [ 8.4%] Disruption_7d       → 🔴 3 disruptions in 7 days — persistent
```

---

## 📊 What is the Bullwhip Effect?

When a retailer sees a small demand spike and orders 2x more than needed,
the wholesaler sees that 2x order and orders 4x, and the manufacturer sees
that 4x order and produces 8x — even though the original consumer only
wanted a little bit more.

**BullwhipAI detects and measures this amplification in real time.**

Bullwhip Ratio = Variance(Manufacturer Orders) / Variance(Consumer Demand)
- Ratio ~1.0 → Healthy, well-synchronized supply chain
- Ratio > 1.5 → Medium risk, some amplification present
- Ratio > 2.5 → High risk, severe bullwhip effect active

---
