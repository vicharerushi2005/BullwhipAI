"""
BullwhipAI
Decision Intelligence Rules

Every rule generates an operational recommendation
based on current supply-chain conditions.
"""

DECISION_RULES = [

    {
        "name": "Increase Inventory",

        "condition": lambda row:
            row["InventoryGap"] > 500,

        "priority": "Critical",

        "action":
            "Increase inventory by approximately 20% to avoid stock shortages.",

        "reason":
            "Current demand is significantly higher than available inventory."
    },

    {
        "name": "Delay Shipment",

        "condition": lambda row:
            row["Rainfall"] > 150,

        "priority": "High",

        "action":
            "Delay non-essential shipments until weather improves.",

        "reason":
            "Heavy rainfall increases transportation disruptions and product damage."
    },

    {
        "name": "Optimize Transportation",

        "condition": lambda row:
            row["FuelPrice"] > 105,

        "priority": "High",

        "action":
            "Use optimized transportation routes and consolidate deliveries.",

        "reason":
            "Fuel prices are unusually high, increasing logistics costs."
    },

    {
        "name": "Increase Procurement",

        "condition": lambda row:
            row["DemandSupplyRatio"] > 1.3,

        "priority": "Critical",

        "action":
            "Increase procurement from suppliers before shortages occur.",

        "reason":
            "Demand is growing faster than available inventory."
    },

    {
        "name": "Build Safety Stock",

        "condition": lambda row:
            row["GovernmentAlert"] == 1,

        "priority": "Medium",

        "action":
            "Build additional safety stock for essential commodities.",

        "reason":
            "Government alerts may indicate future supply disruptions."
    },

    {
        "name": "Reduce Overstock",

        "condition": lambda row:
            row["InventoryRatio"] > 2,

        "priority": "Medium",

        "action":
            "Reduce procurement until inventory returns to healthy levels.",

        "reason":
            "Inventory is considerably higher than expected demand."
    }

]