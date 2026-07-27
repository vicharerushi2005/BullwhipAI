import pandas as pd


events = pd.read_csv(
    "data/market_events.csv"
)


supply_risk = 0
demand_risk = 0
inventory_risk = 0


for index,row in events.iterrows():

    risk = str(row.get("Risk"))


    if risk == "High":
        supply_risk += 30
        inventory_risk += 20


    elif risk == "Medium":
        supply_risk += 15
        inventory_risk += 10


    elif risk == "Needs Analysis":
        demand_risk += 10



# Limit scores to 100

supply_risk = min(supply_risk,100)

demand_risk = min(demand_risk,100)

inventory_risk = min(inventory_risk,100)



overall = (
    supply_risk +
    demand_risk +
    inventory_risk
) / 3



if overall >= 70:
    level = "HIGH"

elif overall >= 40:
    level = "MEDIUM"

else:
    level = "LOW"



report = {

    "Supply Risk Score": supply_risk,

    "Demand Risk Score": demand_risk,

    "Inventory Risk Score": inventory_risk,

    "Overall Score": round(overall,2),

    "Risk Level": level

}



df = pd.DataFrame([report])


df.to_csv(
    "data/risk_score.csv",
    index=False
)


print("Risk Score Generated")
print(df)
