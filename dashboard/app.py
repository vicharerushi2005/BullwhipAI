"""
BullwhipAI Dashboard - v2.0
============================
Complete dashboard with:
  - ML-based risk prediction (not just rule-based)
  - Explainable AI (XAI) panel showing WHY the model predicted what it did
  - Historical trend analysis
  - Feature importance visualization
  - All original panels (weather, commodity, news, product risk)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from datetime import datetime

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="BullwhipAI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 20px 30px; border-radius: 12px; margin-bottom: 20px; color: white;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

@st.cache_data(ttl=60)
def load_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame()

def load_xai():
    path = "data/xai_explanation.json"
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None

def load_metrics():
    path = "models/model_metrics.json"
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None

def risk_color(label):
    return {"LOW": "#00c851", "MEDIUM": "#ffbb33", "HIGH": "#ff4444"}.get(str(label), "#aaa")

def risk_emoji(label):
    return {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(str(label), "⚪")

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

with st.sidebar:
    st.markdown("## 📦 BullwhipAI v2.0")
    st.markdown("*Multi-Agent AI Supply Chain*")
    st.divider()

    metrics = load_metrics()
    if metrics:
        st.markdown("### 🤖 ML Model Status")
        st.success("✅ Model Trained")
        st.metric("Accuracy",  f"{metrics['accuracy']*100:.1f}%")
        st.metric("F1 Score",  f"{metrics['f1_weighted']*100:.1f}%")
        st.metric("Features",  metrics['n_features'])
        st.caption(f"Trained: {metrics['trained_on']}")
        st.caption(f"Algo: {metrics['algorithm']}")
    else:
        st.warning("⚠️ Model not trained.\nRun: `python scripts/train_model.py`")

    st.divider()
    st.markdown("### 📁 Data Status")
    for label, path in [
        ("Weather",    "data/weather_data.csv"),
        ("Commodity",  "data/commodity_data.csv"),
        ("News",       "data/news_data.csv"),
        ("ML Predict", "data/ml_prediction.csv"),
        ("XAI",        "data/xai_explanation.json"),
        ("Historical", "data/historical_supply_chain.csv"),
    ]:
        st.markdown(f"{'✅' if os.path.exists(path) else '❌'} {label}")

    st.divider()
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1 style="margin:0;font-size:2rem;">📦 BullwhipAI</h1>
    <p style="margin:5px 0 0 0;opacity:0.8;">AI-Powered Multi-Agent Food Supply Chain Intelligence</p>
    <small style="opacity:0.6;">Bullwhip Effect Reduction | Explainable AI | Real-Time Monitoring</small>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# LOAD ALL DATA
# -------------------------------------------------------

xai         = load_xai()
risk_df     = load_csv("data/risk_score.csv")
product_df  = load_csv("data/product_risk.csv")
weather_df  = load_csv("data/weather_data.csv")
commodity_df= load_csv("data/commodity_data.csv")
news_df     = load_csv("data/news_data.csv")
ml_pred_df  = load_csv("data/ml_prediction.csv")
hist_df     = load_csv("data/historical_supply_chain.csv")
feat_imp_df = load_csv("models/feature_importance.csv")
xai_hist    = load_csv("data/xai_history.csv")

rb_supply    = float(risk_df["Supply Risk Score"].iloc[0])    if not risk_df.empty else 15.0
rb_demand    = float(risk_df["Demand Risk Score"].iloc[0])    if not risk_df.empty else 50.0
rb_inventory = float(risk_df["Inventory Risk Score"].iloc[0]) if not risk_df.empty else 10.0
rb_overall   = float(risk_df["Overall Score"].iloc[0])        if not risk_df.empty else 25.0

ml_label = "N/A"; ml_confidence = 0
ml_low_pct = ml_med_pct = ml_high_pct = 0.0

if xai:
    ml_label      = xai["prediction"]["risk_label"]
    ml_confidence = xai["prediction"]["confidence_pct"]
    cb = xai["prediction"]["confidence_breakdown"]
    ml_low_pct  = cb.get("LOW",    0)
    ml_med_pct  = cb.get("MEDIUM", 0)
    ml_high_pct = cb.get("HIGH",   0)
elif not ml_pred_df.empty:
    r = ml_pred_df.iloc[-1]
    ml_label      = r.get("Risk_Label", "N/A")
    ml_confidence = float(r.get("Confidence_Pct", 0))
    ml_low_pct    = float(r.get("Risk_LOW_Pct",    0))
    ml_med_pct    = float(r.get("Risk_MEDIUM_Pct", 0))
    ml_high_pct   = float(r.get("Risk_HIGH_Pct",   0))

# -------------------------------------------------------
# SECTION 1 — KPI CARDS
# -------------------------------------------------------

st.markdown("## 📊 Real-Time Risk Dashboard")
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    col = risk_color(ml_label); em = risk_emoji(ml_label)
    st.markdown(f"""
    <div style="background:{col}22;border:2px solid {col};border-radius:10px;
                padding:15px;text-align:center;">
        <div style="font-size:2rem;">{em}</div>
        <div style="font-size:0.75rem;color:#aaa;">ML Risk Level</div>
        <div style="font-size:1.4rem;font-weight:bold;color:{col};">{ml_label}</div>
        <div style="font-size:0.75rem;color:#888;">{ml_confidence:.0f}% confident</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.metric("Supply Risk",   f"{rb_supply:.0f}%",   delta_color="inverse")
with c3:
    st.metric("Demand Risk",   f"{rb_demand:.0f}%",   delta_color="inverse")
with c4:
    st.metric("Inventory Risk",f"{rb_inventory:.0f}%",delta_color="inverse")
with c5:
    st.metric("Overall Score", f"{rb_overall:.1f}")

st.divider()

# -------------------------------------------------------
# SECTION 2 — ML CONFIDENCE + GAUGE
# -------------------------------------------------------

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 🎯 ML Prediction Confidence")
    fig_conf = go.Figure()
    for lbl, val, clr in [("LOW", ml_low_pct,"#00c851"),
                           ("MEDIUM",ml_med_pct,"#ffbb33"),
                           ("HIGH", ml_high_pct,"#ff4444")]:
        fig_conf.add_trace(go.Bar(
            x=[lbl], y=[val], marker_color=clr, name=lbl,
            text=f"{val:.1f}%", textposition="outside"
        ))
    fig_conf.update_layout(
        title="Probability per Risk Class", yaxis_range=[0,110],
        showlegend=False, height=300, margin=dict(t=40,b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white"
    )
    st.plotly_chart(fig_conf, use_container_width=True)

with col_b:
    st.markdown("### 🌡️ Bullwhip Risk Gauge")
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rb_overall,
        gauge={
            "axis": {"range":[0,100]},
            "bar":  {"color": risk_color(ml_label), "thickness":0.3},
            "steps":[{"range":[0,30],"color":"#00380a"},
                     {"range":[30,70],"color":"#3d3200"},
                     {"range":[70,100],"color":"#3d0000"}],
            "threshold":{"line":{"color":"white","width":4},"value":rb_overall,"thickness":0.75}
        },
        title={"text":"Overall Risk Score","font":{"color":"white"}},
        number={"font":{"color":"white"}},
    ))
    fig_g.update_layout(height=300, margin=dict(t=30,b=10),
                        paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig_g, use_container_width=True)

st.divider()

# -------------------------------------------------------
# SECTION 3 — EXPLAINABLE AI PANEL
# -------------------------------------------------------

st.markdown("## 🔍 Explainable AI — Why This Prediction?")
st.caption("No black box. Every decision is explained with evidence.")

if xai:
    narrative = xai.get("bullwhip_narrative", "")
    col = risk_color(ml_label)
    st.markdown(f"""
    <div style="background:{col}11;border-left:4px solid {col};
                padding:14px 18px;border-radius:0 8px 8px 0;margin-bottom:16px;">
        <strong style="color:{col};">📝 Bullwhip Effect Analysis</strong><br>
        <span style="color:#ccc;">{narrative}</span>
    </div>""", unsafe_allow_html=True)

    drivers = xai.get("top_drivers", [])
    if drivers:
        st.markdown("### 📊 Top Driving Factors")
        st.markdown("*Features ranked by their influence on today's prediction:*")

        for i, d in enumerate(drivers[:6]):
            feat   = d["feature"]
            impact = d["local_impact"]
            assess = d["assessment"]
            desc   = d["description"]
            val    = d["value"]
            bar_w  = min(int(impact * 3), 100)
            bar_c  = ("#ff4444" if "🔴" in assess else "#ffbb33" if "🟡" in assess else "#00c851")

            c1x, c2x, c3x, c4x = st.columns([0.06, 0.26, 0.45, 0.23])
            c1x.markdown(f"**#{i+1}**")
            c2x.markdown(f"**{feat}**<br><small style='color:#888'>{desc}</small>",
                         unsafe_allow_html=True)
            c3x.markdown(f"<div style='padding-top:6px'>{assess}</div>",
                         unsafe_allow_html=True)
            c4x.markdown(f"""
                <div style='margin-top:6px;background:{bar_c}33;border-radius:4px;height:14px;'>
                  <div style='background:{bar_c};width:{bar_w}%;height:14px;border-radius:4px;'></div>
                </div>
                <small style='color:#888'>{impact:.1f}% influence | val: {val}</small>
            """, unsafe_allow_html=True)

    st.divider()

    actions = xai.get("actions", {})
    if actions:
        st.markdown("### 🎯 AI-Generated Recommended Actions")
        team_cols = st.columns(len(actions))
        for tcol, (team, acts) in zip(team_cols, actions.items()):
            with tcol:
                st.markdown(f"**{team}**")
                for a in acts:
                    st.markdown(f"• {a}")

    st.markdown(f"""
    <div style="background:#1a1a2e;border:1px solid #333;border-radius:8px;
                padding:10px 16px;margin-top:12px;">
        <small style="color:#555;">
        🤖 XAI: Perturbation-based Local Sensitivity |
        Generated: {xai.get('generated_at','N/A')} |
        {xai.get('model_info',{}).get('algorithm','Random Forest')} |
        {xai.get('model_info',{}).get('features_used','N/A')} features
        </small>
    </div>""", unsafe_allow_html=True)
else:
    st.info("XAI not yet generated. Run: `python agents/ml_prediction_agent.py`")

st.divider()

# -------------------------------------------------------
# SECTION 4 — GLOBAL FEATURE IMPORTANCE
# -------------------------------------------------------

if not feat_imp_df.empty:
    st.markdown("## 🧠 Global Feature Importance")
    st.caption("What the model learned from 6 years of supply chain data:")
    top_f = feat_imp_df.head(15).sort_values("Importance_Pct")
    fig_fi = go.Figure(go.Bar(
        x=top_f["Importance_Pct"], y=top_f["Feature"],
        orientation="h",
        marker=dict(color=top_f["Importance_Pct"], colorscale="RdYlGn_r", showscale=True),
        text=[f"{v:.1f}%" for v in top_f["Importance_Pct"]], textposition="outside",
    ))
    fig_fi.update_layout(
        title="Feature Importance — Random Forest",
        xaxis_title="Importance (%)", height=480,
        margin=dict(l=190,t=40,b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white"
    )
    st.plotly_chart(fig_fi, use_container_width=True)
    st.divider()

# -------------------------------------------------------
# SECTION 5 — HISTORICAL TRENDS
# -------------------------------------------------------

if not hist_df.empty:
    st.markdown("## 📈 Historical Supply Chain Analysis (2020–2025)")
    tab1, tab2, tab3 = st.tabs(["📦 Bullwhip Ratio", "💰 Commodity Prices", "⚠️ Risk Timeline"])

    hist_df["Date"] = pd.to_datetime(hist_df["Date"])

    with tab1:
        monthly = (hist_df.groupby(hist_df["Date"].dt.to_period("M"))
                   .agg(Bullwhip_Ratio=("Bullwhip_Ratio","mean"))
                   .reset_index())
        monthly["Date"] = monthly["Date"].astype(str)
        fig_bwr = go.Figure()
        fig_bwr.add_trace(go.Scatter(x=monthly["Date"], y=monthly["Bullwhip_Ratio"],
                                     fill="tozeroy", line_color="#e94560", name="BWR"))
        fig_bwr.add_hline(y=1.5, line_dash="dash", line_color="#ffbb33",
                          annotation_text="MEDIUM (1.5)")
        fig_bwr.add_hline(y=2.5, line_dash="dash", line_color="#ff4444",
                          annotation_text="HIGH (2.5)")
        fig_bwr.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_bwr, use_container_width=True)
        st.caption("COVID-19 (2020–21) caused major order amplification spikes.")

    with tab2:
        mp = (hist_df.groupby(hist_df["Date"].dt.to_period("M"))
              .agg(Price=("Commodity_Price_INR","mean")).reset_index())
        mp["Date"] = mp["Date"].astype(str)
        fig_p = px.line(mp, x="Date", y="Price", color_discrete_sequence=["#f5a623"],
                        title="Commodity Price — Monthly Average (INR/quintal)")
        fig_p.add_hline(y=4500, line_dash="dash", line_color="#ffbb33")
        fig_p.add_hline(y=5500, line_dash="dash", line_color="#ff4444")
        fig_p.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_p, use_container_width=True)

    with tab3:
        rc = (hist_df.groupby([hist_df["Date"].dt.to_period("M"), "Risk_Label"])
              .size().unstack(fill_value=0).reset_index())
        rc["Date"] = rc["Date"].astype(str)
        fig_r = go.Figure()
        for lbl, clr in [("LOW","#00c851"),("MEDIUM","#ffbb33"),("HIGH","#ff4444")]:
            if lbl in rc.columns:
                fig_r.add_trace(go.Bar(x=rc["Date"], y=rc[lbl], name=lbl, marker_color=clr))
        fig_r.update_layout(barmode="stack", height=350,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_r, use_container_width=True)
    st.divider()

# -------------------------------------------------------
# SECTION 6 — PREDICTION HISTORY
# -------------------------------------------------------

if not xai_hist.empty and len(xai_hist) > 1:
    st.markdown("## ⏱️ ML Prediction History")
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=xai_hist["Timestamp"], y=xai_hist["Confidence_Pct"],
                               mode="lines+markers", line_color="#e94560",
                               name="Confidence %"))
    fig_h.update_layout(height=250, title="Model Confidence Over Time",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig_h, use_container_width=True)
    st.dataframe(xai_hist.tail(10), use_container_width=True)
    st.divider()

# -------------------------------------------------------
# SECTION 7 — PRODUCT RISK
# -------------------------------------------------------

if not product_df.empty:
    st.markdown("## 📦 Product-Wise Risk")
    pc1, pc2 = st.columns([3,2])
    with pc1:
        fig_pr = px.bar(product_df, x="Product", y="Risk Score", color="Risk Level",
                        color_discrete_map={"LOW":"#00c851","MEDIUM":"#ffbb33","HIGH":"#ff4444"},
                        title="Product Risk Scores")
        fig_pr.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)",
                             paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_pr, use_container_width=True)
    with pc2:
        cols_show = [c for c in ["Product","Risk Level","Risk Score","Reason"] if c in product_df.columns]
        st.dataframe(product_df[cols_show], use_container_width=True, height=350)
    st.divider()

# -------------------------------------------------------
# SECTION 8 — LIVE DATA
# -------------------------------------------------------

cw, cc = st.columns(2)
with cw:
    st.markdown("### 🌦️ Weather Monitoring")
    if not weather_df.empty:
        w = weather_df.iloc[-1]
        wc1, wc2 = st.columns(2)
        wc1.metric("🌡️ Temperature", f"{w.get('Temperature_C','N/A')}°C")
        wc2.metric("💨 Wind", f"{w.get('Wind_Speed_kmh','N/A')} km/h")
        st.caption(f"📍 {w.get('City','')} {w.get('State','')} | {w.get('Date','')}")
        st.dataframe(weather_df.tail(3), use_container_width=True)

with cc:
    st.markdown("### 📈 Commodity Market")
    if not commodity_df.empty:
        for _, row in commodity_df.iterrows():
            try:
                price_f = float(str(row.get("Price","4200")).replace(",",""))
                badge = "🔴 HIGH" if price_f > 5000 else "🟡 ELEVATED" if price_f > 4500 else "🟢 NORMAL"
                st.metric(f"{row.get('Product','')} Price",
                          f"₹{price_f:,.0f} {row.get('Unit','')}", delta=badge, delta_color="off")
            except Exception:
                st.metric(str(row.get("Product","")), str(row.get("Price","")))

st.divider()

# -------------------------------------------------------
# SECTION 9 — NEWS
# -------------------------------------------------------

st.markdown("### 📰 Market Intelligence")
if not news_df.empty:
    for _, row in news_df.head(6).iterrows():
        title = row.get("Title",""); link = row.get("Link","#")
        neg = any(w in title.lower() for w in ["crisis","shortage","flood","strike","ban","inflation"])
        pos = any(w in title.lower() for w in ["growth","expand","invest","stable","recover"])
        icon = "🔴" if neg else "🟢" if pos else "🔵"
        st.markdown(f"{icon} [{title}]({link})")

st.divider()


# -------------------------------------------------------
# SECTION 10 — AI NARRATIVE REPORT
# -------------------------------------------------------

st.markdown("### 🤖 AI Supply Chain Report")
ai_report_path = "data/ai_report.txt"
if os.path.exists(ai_report_path):
    with open(ai_report_path, encoding="utf-8") as f:
        ai_report_text = f.read()
    st.text_area("Full AI Report", ai_report_text, height=300)
else:
    st.info("AI report not yet generated. Run the full pipeline to create it.\n\n"
            "`python run_system.py`")

st.divider()

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------

st.markdown("""
<div style="text-align:center;padding:20px;opacity:0.4;font-size:0.75rem;">
BullwhipAI v2.0 | Multi-Agent AI Supply Chain | ML: Random Forest (200 trees, 26 features) |
XAI: Perturbation-based Local Sensitivity | Final Year Engineering Project
</div>""", unsafe_allow_html=True)
