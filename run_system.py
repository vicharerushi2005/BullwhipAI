import subprocess
import sys


print("="*60)
print("BULLWHIP AI AUTOMATED SYSTEM")
print("="*60)


agents = [

    "weather/weather_collector.py",

    "scraper/news_collector.py",

    "commodity/commodity_agent.py",

    "agents/data_processing_agent.py",

    "agents/risk_scoring_agent.py",

    "agents/product_risk_agent.py",

    "ai_agent/market_ai_agent.py"

]


for agent in agents:

    print("\nRunning:", agent)

    result = subprocess.run(
        [sys.executable, agent]
    )


    if result.returncode != 0:

        print("\nERROR IN:", agent)

        break



print("\n")

print("="*60)
print("BULLWHIP AI EXECUTION COMPLETED")
print("="*60)
