"""
BullwhipAI
Central AI Prediction Engine

Used by:
• Explainable AI Agent
• Decision Intelligence Agent
• Inventory Optimization Agent
• Simulation Agent
• Demand Forecast Agent
"""

from utils.model_utils import load_model
from utils.preprocessing import prepare_features


class AIEngine:

    def __init__(self):

        artifacts = load_model()

        self.model = artifacts["model"]
        self.encoder = artifacts["encoder"]
        self.features = artifacts["features"]
        self.metadata = artifacts["metadata"]

    def predict(self, dataframe):

        X = prepare_features(dataframe)

        X = X[self.features]

        prediction_encoded = self.model.predict(X)[0]

        prediction = self.encoder.inverse_transform(
            [prediction_encoded]
        )[0]

        probabilities = self.model.predict_proba(X)[0]

        confidence = round(
         float(max(probabilities)) * 100,
         2
        )

        probability_map = {}

        for label, probability in zip(
            self.encoder.classes_,
            probabilities
        ):

            probability_map[label] = round(
                float(probability),
                4
            )

        return {

            "prediction": prediction,

            "confidence": confidence,

            "probabilities": probability_map
        }