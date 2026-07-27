import ollama
import pandas as pd


# ==============================
# LOAD PROCESSED MARKET DATA
# ==============================

events = pd.read_csv(
    "data/market_events.csv"
)


risk = pd.read_csv(
    "data/risk_score.csv"
)


# Convert data into readable format

market_information = events.to_string()

risk_information = risk.to_string()


# ==============================
# CREATE AI PROMPT
# ==============================

prompt = f"""

You are an AI Food Supply Chain Risk Analyst.

Your objective is to reduce the Bullwhip Effect.

Analyze the following processed market information.


MARKET EVENTS:

{market_information}



RISK SCORE:

{risk_information}



Prepare a practical business report.


Follow this exact format:


===== SUPPLY CHAIN ALERT =====


1. Overall Risk Level:

Mention:
- Risk Level
- Risk Score


2. Main Supply Chain Threats:

Identify:
- Supply risks
- Demand risks
- Inventory risks


3. Products Potentially Affected:

Mention possible food products.


4. Bullwhip Effect Impact:

Explain how this situation can create:
- Overstocking
- Shortages
- Wrong production planning
- Demand fluctuation


5. Recommended Actions:


Procurement Team:
What should they do?


Inventory Team:
What should they do?


Production Team:
What should they do?


6. Final Decision:

Give a short action recommendation.


Important:
Do not only summarize information.
Think like a supply chain manager making decisions.

"""


# ==============================
# SEND TO LOCAL AI MODEL
# ==============================

response = ollama.chat(

    model="qwen2.5:1.5b",

    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]

)


# ==============================
# DISPLAY RESULT
# ==============================

print("\n")
print("========== AI SUPPLY CHAIN REPORT ==========")
print("\n")


print(
    response["message"]["content"]
)
