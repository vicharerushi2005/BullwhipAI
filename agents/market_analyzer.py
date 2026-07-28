import pandas as pd

# Read weather data
weather = pd.read_csv("data/weather_data.csv")

# Read values
temperature = weather["Temperature_C"][0]
city = weather["City"][0]
state = weather["State"][0]
wind = weather["Wind_Speed_kmh"][0]

print("\n========== MARKET ANALYSIS ==========\n")

print(f"Location : {city}, {state}")
print(f"Temperature : {temperature} °C")
print(f"Wind Speed : {wind} km/h")

print("\nMarket Risk Analysis")
print("--------------------------")

if temperature >= 35:
    print("🔴 HIGH RISK")
    print("Reason : Extreme Heat")
    print("Possible Impact : Higher beverage demand, cold storage pressure.")

elif temperature <= 20:
    print("🟡 MEDIUM RISK")
    print("Reason : Low Temperature")
    print("Possible Impact : Transportation delays and demand changes.")

else:
    print("🟢 LOW RISK")
    print("Reason : Normal Weather")
    print("Possible Impact : Stable supply chain.")
