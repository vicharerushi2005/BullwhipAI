"""
BullwhipAI Backend API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json

import pandas as pd
from pathlib import Path

from core.ai_engine import AIEngine


app = FastAPI(
    title="BullwhipAI API",
    description="AI powered Bullwhip Effect Reduction System",
    version="1.0"
)


# Allow dashboard connection

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent.parent


DATA_FILE = (
    BASE_DIR /
    "datasets" /
    "processed" /
    "featured_supply_chain.csv"
)


# Initialize AI Engine

ai_engine = AIEngine()


# ----------------------------------
# ROOT
# ----------------------------------

@app.get("/")
def root():

    return {

        "project":"BullwhipAI",

        "status":"Running",

        "message":
        "Autonomous Multi Agent Supply Chain Intelligence System"

    }



# ----------------------------------
# HEALTH CHECK
# ----------------------------------

@app.get("/health")
def health():

    return {

        "status":"healthy"

    }



# ----------------------------------
# AI PREDICTION
# ----------------------------------

@app.get("/prediction")
def prediction():

    df = pd.read_csv(DATA_FILE)

    result = ai_engine.predict(df.tail(1))

    return result



# ----------------------------------
# DATA SUMMARY
# ----------------------------------

@app.get("/summary")
def summary():

    df = pd.read_csv(DATA_FILE)


    return {

        "total_records":len(df),

        "products":
        df["Product"].nunique(),

        "cities":
        df["City"].nunique(),

        "latest_date":
        str(df["Date"].iloc[-1])

    }
# ----------------------------------
# EXPLAINABLE AI
# ----------------------------------

@app.get("/explanation")
def explanation():

    file = (
        BASE_DIR /
        "datasets" /
        "explanations" /
        "explanation.json"
    )

    with open(file, "r") as f:
        return json.load(f)


# ----------------------------------
# RECOMMENDATIONS
# ----------------------------------

@app.get("/recommendation")
def recommendation():

    file = (
        BASE_DIR /
        "datasets" /
        "recommendations" /
        "recommendations.json"
    )

    with open(file, "r") as f:
        return json.load(f)


# ----------------------------------
# INVENTORY
# ----------------------------------

@app.get("/inventory")
def inventory():

    file = (
        BASE_DIR /
        "datasets" /
        "optimization" /
        "inventory_optimization.json"
    )

    with open(file, "r") as f:
        return json.load(f)