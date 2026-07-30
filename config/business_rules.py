"""
BullwhipAI

Business Knowledge Base

These rules convert ML features into
human-readable business explanations.
"""

FEATURE_EXPLANATIONS = {

    "Rainfall": {
        "title": "Heavy Rainfall",
        "reason":
            "Heavy rainfall can delay harvesting, damage crops, slow transportation, and reduce market supply. These disruptions often increase supply uncertainty and amplify the Bullwhip Effect.",
        "impact":
            "High"
    },

    "CommodityPrice": {
        "title": "Commodity Price Increase",
        "reason":
            "Higher commodity prices increase procurement costs and usually indicate unstable market conditions. Businesses may place larger safety orders, increasing demand variability.",
        "impact":
            "High"
    },

    "Demand": {
        "title": "Demand Surge",
        "reason":
            "Demand is significantly higher than normal. If inventory cannot satisfy customer demand, panic ordering and stock shortages may occur.",
        "impact":
            "High"
    },

    "Inventory": {
        "title": "Inventory Level",
        "reason":
            "Inventory levels directly influence the ability to satisfy customer demand. Low inventory increases stock-out risk and creates unstable replenishment cycles.",
        "impact":
            "Medium"
    },

    "InventoryGap": {
        "title": "Demand Inventory Gap",
        "reason":
            "Demand currently exceeds available inventory. This imbalance is a strong indicator of future shortages and increased Bullwhip risk.",
        "impact":
            "Very High"
    },

    "DemandSupplyRatio": {
        "title": "Demand Supply Ratio",
        "reason":
            "A higher demand-to-supply ratio indicates demand is growing faster than available stock. Suppliers often react with over-ordering, increasing volatility.",
        "impact":
            "Very High"
    },

    "FuelImpact": {
        "title": "Fuel Cost Impact",
        "reason":
            "Higher fuel prices increase transportation costs and may delay deliveries throughout the supply chain.",
        "impact":
            "Medium"
    },

    "DelayScore": {
        "title": "Transportation Delay",
        "reason":
            "Long transportation delays reduce replenishment speed, increase uncertainty, and force distributors to maintain larger inventories.",
        "impact":
            "High"
    },

    "GovernmentAlert": {
        "title": "Government Advisory",
        "reason":
            "Government advisories may indicate export restrictions, policy changes, food safety issues, or emergency situations that affect the supply chain.",
        "impact":
            "Medium"
    },

    "WeatherSeverity": {
        "title": "Severe Weather",
        "reason":
            "Extreme weather conditions increase operational uncertainty and frequently disrupt logistics, farming, and distribution.",
        "impact":
            "Very High"
    }
}