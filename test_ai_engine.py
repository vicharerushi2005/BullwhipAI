import pandas as pd
from pathlib import Path

from core.ai_engine import AIEngine

BASE_DIR = Path(__file__).resolve().parent

FILE = (
    BASE_DIR /
    "datasets" /
    "processed" /
    "featured_supply_chain.csv"
)

df = pd.read_csv(FILE)

latest = df.tail(1)

engine = AIEngine()

result = engine.predict(latest)

print(result)