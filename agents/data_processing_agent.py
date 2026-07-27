import pandas as pd
import os


print("Starting Data Processing...")


# Load Data

news = pd.read_csv(
    "data/news_data.csv"
)

weather = pd.read_csv(
    "data/weather_data.csv"
)

commodity = pd.read_csv(
    "data/commodity_data.csv"
)



# Create Market Event File

events = []


# Weather Analysis

temperature = weather["Temperature_C"][0]

city = weather["City"][0]


if temperature > 35:

    events.append({
        "Event": "High Temperature",
        "Location": city,
        "Impact": "Possible increase in cooling product demand",
        "Risk": "Medium"
    })


elif temperature < 15:

    events.append({
        "Event": "Low Temperature",
        "Location": city,
        "Impact": "Possible change in food demand",
        "Risk": "Low"
    })


else:

    events.append({
        "Event": "Normal Weather",
        "Location": city,
        "Impact": "No major weather risk",
        "Risk": "Low"
    })



# Commodity Analysis

for index,row in commodity.iterrows():

    if row["Price"] > 50:

        events.append({

            "Event": "High Commodity Price",

            "Product": row["Product"],

            "Impact": "Possible cost increase",

            "Risk": "Medium"

        })



# News Events

for title in news["Title"].head(5):

    events.append({

        "Event": "Market News",

        "Information": title,

        "Risk": "Needs Analysis"

    })



event_df = pd.DataFrame(events)



event_df.to_csv(

    "data/market_events.csv",

    index=False

)


print("Market Events Created")

print(event_df)
