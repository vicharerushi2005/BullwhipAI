import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


st.set_page_config(
    page_title="BullwhipAI",
    page_icon="📦",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("📦 BullwhipAI - Food Supply Chain Intelligence")

st.write(
    "AI Based Multi-Agent System for Bullwhip Effect Reduction"
)


# -----------------------------
# Load Data
# -----------------------------

risk = pd.read_csv(
    "data/risk_score.csv"
)

product = pd.read_csv(
    "data/product_risk.csv"
)

weather = pd.read_csv(
    "data/weather_data.csv"
)

commodity = pd.read_csv(
    "data/commodity_data.csv"
)

news = pd.read_csv(
    "data/news_data.csv"
)


# -----------------------------
# Risk Values
# -----------------------------

overall = float(
    risk["Overall Score"][0]
)

supply = float(
    risk["Supply Risk Score"][0]
)

demand = float(
    risk["Demand Risk Score"][0]
)

inventory = float(
    risk["Inventory Risk Score"][0]
)


if overall < 30:

    level = "LOW 🟢"

elif overall < 70:

    level = "MEDIUM 🟡"

else:

    level = "HIGH 🔴"



# -----------------------------
# Dashboard Cards
# -----------------------------

col1,col2,col3,col4 = st.columns(4)


with col1:

    st.metric(
        "Overall Risk",
        level
    )


with col2:

    st.metric(
        "Supply Risk",
        f"{supply}%"
    )


with col3:

    st.metric(
        "Demand Risk",
        f"{demand}%"
    )


with col4:

    st.metric(
        "Inventory Risk",
        f"{inventory}%"
    )



st.divider()



# -----------------------------
# Risk Gauge
# -----------------------------


st.subheader("🎯 Overall Bullwhip Risk")


fig = go.Figure(

    go.Indicator(

        mode="gauge+number",

        value=overall,

        gauge={

            "axis":{
                "range":[0,100]
            },

            "bar":{
                "color":"red"
            },

            "steps":[

                {
                    "range":[0,30],
                    "color":"green"
                },

                {
                    "range":[30,70],
                    "color":"yellow"
                },

                {
                    "range":[70,100],
                    "color":"red"
                }

            ]

        }

    )

)


st.plotly_chart(
    fig,
    use_container_width=True
)



# -----------------------------
# Product Risk
# -----------------------------


st.subheader("📦 Product Wise Risk")


fig2 = px.bar(

    product,

    x="Product",

    y="Risk Score",

    color="Risk Level",

    title="Product Risk Analysis"

)


st.plotly_chart(

    fig2,

    use_container_width=True

)


st.dataframe(

    product,

    use_container_width=True

)



# -----------------------------
# Commodity
# -----------------------------


st.subheader("📈 Commodity Market Data")


st.dataframe(

    commodity,

    use_container_width=True

)



# -----------------------------
# Weather
# -----------------------------


st.subheader("🌦 Weather Monitoring")


st.dataframe(

    weather,

    use_container_width=True

)



# -----------------------------
# News
# -----------------------------


st.subheader("📰 Market Intelligence")


for i,row in news.head(5).iterrows():

    st.info(
        row["Title"]
    )



# -----------------------------
# AI Decision
# -----------------------------


st.divider()

st.subheader("🤖 AI Supply Chain Recommendation")


if overall < 30:

    st.success(
        """
        Current supply chain condition is stable.

        Recommended:
        - Maintain inventory levels
        - Continue monitoring market signals
        """
    )


elif overall < 70:

    st.warning(
        """
        Moderate Bullwhip Risk Detected.

        Recommended:
        - Monitor suppliers
        - Avoid unnecessary stock accumulation
        - Review demand changes
        """
    )


else:

    st.error(
        """
        High Bullwhip Risk Detected.

        Recommended:
        - Increase safety stock
        - Activate alternate suppliers
        - Review production planning
        """
    )


st.divider()


st.caption(
    "BullwhipAI | Multi-Agent Artificial Intelligence Supply Chain System"
)
