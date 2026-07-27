import pandas as pd


weather = pd.read_csv(
    "data/weather_data.csv"
)


temperature = weather["Temperature"][0]


print("Market Risk Analysis")
print("--------------------")


if temperature > 35:

    print("HIGH RISK")
    print("Possible increase in beverage demand")

elif temperature < 20:

    print("MEDIUM RISK")
    print("Weather may affect demand")

else:

    print("LOW RISK")
    print("Weather conditions normal")
