import pandas as pd


news = pd.read_csv(
    "data/news_data.csv"
)


weather = pd.read_csv(
    "data/weather_data.csv"
)


commodity = pd.read_csv(
    "data/commodity_data.csv"
)


print("NEWS")
print(news.head())


print("\nWEATHER")
print(weather.head())


print("\nCOMMODITY")
print(commodity.head())


print("\nMarket Data Loaded Successfully")
