import pandas as pd
from datetime import datetime


commodity_prices = [
    {
        "Product": "Tomato",
        "Price": 40,
        "Unit": "Kg"
    },
    {
        "Product": "Wheat",
        "Price": 35,
        "Unit": "Kg"
    },
    {
        "Product": "Rice",
        "Price": 60,
        "Unit": "Kg"
    },
    {
        "Product": "Sugar",
        "Price": 45,
        "Unit": "Kg"
    }
]


df = pd.DataFrame(commodity_prices)

df["Date"] = datetime.now()


df.to_csv(
    "data/commodity_data.csv",
    index=False
)


print("Commodity Data Saved")
print(df)
