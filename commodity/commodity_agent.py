import pandas as pd
import datetime
import os


def collect_commodity_data():

    print("Starting Commodity Agent...")


    # Current date

    today = datetime.datetime.now()


    # Temporary real-data structure
    # We will connect live mandi API next

    data = {

        "Date":[today],

        "Product":[
            "Rice"
        ],

        "Price":[
            4200
        ],

        "Unit":[
            "per quintal"
        ],

        "Source":[
            "Commodity Market Data"
        ]

    }


    df = pd.DataFrame(data)


    # Create data folder if missing

    os.makedirs(
        "data",
        exist_ok=True
    )


    df.to_csv(
        "data/commodity_data.csv",
        index=False
    )


    print("Commodity Data Updated")

    print(df)



if __name__=="__main__":

    collect_commodity_data()
