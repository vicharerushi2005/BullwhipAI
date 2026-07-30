"""
BullwhipAI
Shared Data Preprocessing Pipeline

Every ML-related agent should use this file.

Used by:
    • ML Training Agent
    • Prediction Agent
    • Explainable AI Agent
    • Future Forecasting Agent
"""

import pandas as pd

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

DROP_COLUMNS = [
    "Date",
    "City",
    "State",
    "Country",
    "Product"
]

SENTIMENT_MAP = {
    "Positive": 0,
    "Neutral": 1,
    "Negative": 2
}


# --------------------------------------------------
# PREPARE FEATURES
# --------------------------------------------------

def prepare_features(df, training=False):
    """
    Converts raw dataset into ML-ready features.

    Parameters
    ----------
    df : pandas.DataFrame

    training : bool
        True when training model.
        False when making predictions.

    Returns
    -------
    X : Feature dataframe

    y : Target column (only during training)
    """

    df = df.copy()

    # ------------------------------------------
    # Remove unnecessary columns
    # ------------------------------------------

    for column in DROP_COLUMNS:

        if column in df.columns:

            df.drop(columns=column, inplace=True)

    # ------------------------------------------
    # Encode Market Sentiment
    # ------------------------------------------

    if "MarketSentiment" in df.columns:

        df["MarketSentiment"] = (
            df["MarketSentiment"]
            .map(SENTIMENT_MAP)
            .fillna(1)
            .astype(int)
        )

    # ------------------------------------------
    # TRAINING MODE
    # ------------------------------------------

    if training:

        X = df.drop(columns=["RiskLevel"])

        y = df["RiskLevel"]

        return X, y

    # ------------------------------------------
    # PREDICTION MODE
    # ------------------------------------------

    if "RiskLevel" in df.columns:

        df = df.drop(columns=["RiskLevel"])

    return df