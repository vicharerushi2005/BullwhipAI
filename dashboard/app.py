import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Bullwhip AI Dashboard",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Bullwhip AI - Food Supply Chain Dashboard")

st.markdown("---")

# -------------------------------
# LOAD DATA
# -------------------------------

risk = pd.read_csv("data/risk_score.csv")
events = pd.read_csv("data/market_events.csv")
weather = pd.read_csv("data/weather_data.csv")
commodity = pd.read_csv("data/commodity_data.csv")
news = pd.read_csv("data/news_data.csv")

# -------------------------------
# RISK LEVEL
# -------------------------------

overall = float(risk["Overall Score"][0])

if overall < 30:
    color = "green"
    status = "🟢 LOW"

elif overall < 70:
    color = "orange"
    status = "🟡 MEDIUM"

else:
    color = "red"
    status = "🔴 HIGH"

st.header("Overall Bullwhip Risk")

st.metric(
    label="Current Risk",
    value=status
)

st.progress(int(overall)/100)

st.write(f"Overall Score : **{overall}/100**")

st.markdown("---")

# -------------------------------
# RISK METERS
# -------------------------------

st.header("Risk Breakdown")

c1,c2,c3 = st.columns(3)

with c1:

    supply = int(risk["Supply Risk Score"][0])

    st.subheader("Supply Risk")

    st.progress(supply/100)

    st.write(f"{supply}/100")

with c2:

    demand = int(risk["Demand Risk Score"][0])

    st.subheader("Demand Risk")

    st.progress(demand/100)

    st.write(f"{demand}/100")

with c3:

    inventory = int(risk["Inventory Risk Score"][0])

    st.subheader("Inventory Risk")

    st.progress(inventory/100)

    st.write(f"{inventory}/100")

st.markdown("---")

# -------------------------------
# PRODUCTS
# -------------------------------

st.header("Products Being Monitored")

st.dataframe(commodity,use_container_width=True)

st.markdown("---")

# -------------------------------
# WEATHER
# -------------------------------

st.header("Weather Status")

st.dataframe(weather,use_container_width=True)

st.markdown("---")

# -------------------------------
# MARKET EVENTS
# -------------------------------

st.header("Detected Market Events")

st.dataframe(events,use_container_width=True)

st.markdown("---")

# -------------------------------
# NEWS
# -------------------------------

st.header("Latest Food Industry News")

for i,row in news.head(5).iterrows():

    st.write("📰",row["Title"])

st.markdown("---")

# -------------------------------
# AI DECISION
# -------------------------------

st.header("AI Recommendation")

if overall < 30:

    st.success("""
Current market conditions are stable.

Recommended Action:

• Continue normal procurement.

• Maintain current inventory.

• Monitor daily news.
""")

elif overall < 70:

    st.warning("""
Medium Risk Detected

Recommended Action

• Increase monitoring.

• Prepare backup suppliers.

• Review inventory weekly.
""")

else:

    st.error("""
High Risk Detected

Recommended Action

• Increase safety stock.

• Contact suppliers immediately.

• Reduce unnecessary production.

• Monitor market every few hours.
""")
