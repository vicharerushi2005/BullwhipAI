import requests
import pandas as pd
from datetime import datetime


# Location Details

city = "Panvel"
state = "Maharashtra"
country = "India"


latitude = 18.9894
longitude = 73.1175


# Weather API

url = (
    "https://api.open-meteo.com/v1/forecast?"
    f"latitude={latitude}"
    f"&longitude={longitude}"
    "&current_weather=true"
)


response = requests.get(url)

weather = response.json()


current = weather["current_weather"]


data = {

    "Date": datetime.now(),

    "City": city,

    "State": state,

    "Country": country,

    "Latitude": latitude,

    "Longitude": longitude,

    "Temperature_C": current["temperature"],

    "Wind_Speed_kmh": current["windspeed"]

}


df = pd.DataFrame([data])


df.to_csv(
    "data/weather_data.csv",
    index=False
)


print("Weather Updated Successfully")
print(df)
