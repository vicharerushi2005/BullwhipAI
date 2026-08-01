"""
BullwhipAI - Market AI Agent v2.1
===================================
Generates LLM narrative using Ollama (local, free).
Gracefully handles Ollama being offline — falls back to
a rule-based report so the pipeline never crashes.
"""

import pandas as pd
import os
import json
from datetime import datetime

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------

def load_csv_safe(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame()

events      = load_csv_safe("data/market_events.csv")
risk        = load_csv_safe("data/risk_score.csv")
xai_data    = {}
xai_path    = "data/xai_explanation.json"
if os.path.exists(xai_path):
    try:
        with open(xai_path) as f:
            xai_data = json.load(f)
    except Exception:
        pass

market_info = events.to_string() if not events.empty else "No market events data."
risk_info   = risk.to_string()   if not risk.empty   else "No risk score data."

# Include XAI summary in prompt if available
xai_summary = ""
if xai_data:
    pred = xai_data.get("prediction", {})
    drivers = xai_data.get("top_drivers", [])[:3]
    xai_summary = f"""
ML MODEL PREDICTION:
Risk Level : {pred.get('risk_label','N/A')} (Confidence: {pred.get('confidence_pct','N/A')}%)
Top Drivers: {', '.join([d['feature'] for d in drivers])}
Analysis   : {xai_data.get('bullwhip_narrative','')}
"""

# -------------------------------------------------------
# PROMPT
# -------------------------------------------------------

prompt = f"""
You are an AI Food Supply Chain Risk Analyst for India.
Your objective is to reduce the Bullwhip Effect.
Analyze the following processed market information.

MARKET EVENTS:
{market_info}

RISK SCORE:
{risk_info}
{xai_summary}

Prepare a practical business report in this exact format:

===== SUPPLY CHAIN ALERT =====

1. Overall Risk Level:
   - Risk Level and Score

2. Main Supply Chain Threats:
   - Supply risks
   - Demand risks
   - Inventory risks

3. Products Potentially Affected:
   List specific food products.

4. Bullwhip Effect Impact:
   Explain how this causes overstocking, shortages, or wrong planning.

5. Recommended Actions:

   Procurement Team: (2-3 actions)
   Inventory Team:   (2-3 actions)
   Production Team:  (2-3 actions)

6. Final Decision:
   One clear action recommendation.

Think like a supply chain manager making real decisions.
"""

# -------------------------------------------------------
# RULE-BASED FALLBACK REPORT
# -------------------------------------------------------

def generate_fallback_report():
    """Generate a structured report without Ollama."""
    risk_level   = "LOW"
    risk_score   = 25.0
    ml_risk      = "N/A"
    ml_conf      = 0
    narrative    = ""

    if not risk.empty:
        risk_level = str(risk.get("Risk Level", pd.Series(["LOW"])).iloc[0])
        risk_score = float(risk.get("Overall Score", pd.Series([25.0])).iloc[0])

    if xai_data:
        pred     = xai_data.get("prediction", {})
        ml_risk  = pred.get("risk_label", "N/A")
        ml_conf  = pred.get("confidence_pct", 0)
        narrative= xai_data.get("bullwhip_narrative", "")

    actions = xai_data.get("actions", {}) if xai_data else {}

    report_lines = [
        "",
        "========== AI SUPPLY CHAIN REPORT (Rule-Based) ==========",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "(Ollama offline — using rule-based report)",
        "",
        "===== SUPPLY CHAIN ALERT =====",
        "",
        f"1. Overall Risk Level:",
        f"   Rule-Based : {risk_level} (Score: {risk_score})",
        f"   ML Model   : {ml_risk} ({ml_conf}% confidence)",
        "",
        f"2. Bullwhip Effect Analysis:",
        f"   {narrative}" if narrative else "   Monitoring supply chain signals.",
        "",
        "3. Products Potentially Affected:",
        "   Rice, Wheat, Tomato, Onion, Sugar, Milk, Processed Food",
        "",
        "4. Recommended Actions:",
    ]

    if actions:
        for team, acts in actions.items():
            report_lines.append(f"\n   {team}:")
            for a in acts:
                report_lines.append(f"     • {a}")
    else:
        report_lines += [
            "   Procurement : Monitor supplier lead times and commodity prices",
            "   Inventory   : Maintain safety stock, avoid panic over-ordering",
            "   Production  : Use 7-day rolling demand average for planning",
        ]

    report_lines += [
        "",
        "6. Final Decision:",
        f"   {'Immediate intervention required — activate backup suppliers and increase safety stock.' if risk_level == 'HIGH' else 'Continue monitoring. Avoid reactive ordering to prevent bullwhip amplification.' if risk_level == 'MEDIUM' else 'Stable operations. Use this period to optimise inventory and renegotiate contracts.'}",
        "",
        "=" * 50,
    ]

    return "\n".join(report_lines)


# -------------------------------------------------------
# TRY OLLAMA, FALL BACK GRACEFULLY
# -------------------------------------------------------

report_text = None

try:
    import ollama
    print("Connecting to Ollama...")
    response = ollama.chat(
        model="qwen2.5:1.5b",
        messages=[{"role": "user", "content": prompt}]
    )
    report_text = response["message"]["content"]
    print("\n========== AI SUPPLY CHAIN REPORT (Ollama) ==========\n")
    print(report_text)

except ImportError:
    print("⚠️  ollama package not installed. Using rule-based report.")
    report_text = generate_fallback_report()
    print(report_text)

except Exception as e:
    err = str(e)
    if "ConnectionError" in err or "connect" in err.lower() or "ollama" in err.lower():
        print("⚠️  Ollama is not running. Using rule-based report.")
        print("   To enable AI narratives: start Ollama and run: ollama pull qwen2.5:1.5b")
    else:
        print(f"⚠️  Ollama error: {err}. Using rule-based report.")
    report_text = generate_fallback_report()
    print(report_text)

# -------------------------------------------------------
# SAVE REPORT
# -------------------------------------------------------

if report_text:
    os.makedirs("data", exist_ok=True)
    report_path = "data/ai_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(report_text)
    print(f"\n✅ Report saved → {report_path}")
