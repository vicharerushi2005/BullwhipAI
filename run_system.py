"""
BullwhipAI - Master Run Script v2.2
=====================================
Fixed:
  - Uses 'python -m streamlit' instead of bare 'streamlit' command
  - Graceful handling of Ollama offline
  - Pre-flight check tells you exactly what's missing

Usage:
  python run_system.py              # full pipeline + dashboard
  python run_system.py --setup      # generate data + train model (FIRST TIME)
  python run_system.py --dashboard  # dashboard only
  python run_system.py --check      # check environment, don't run anything
"""

import subprocess
import sys
import os
import time
from datetime import datetime

BANNER = """
╔══════════════════════════════════════════════════════════╗
║           BULLWHIP AI  v2.2 — FULL PIPELINE             ║
║   Multi-Agent AI Food Supply Chain Intelligence          ║
╚══════════════════════════════════════════════════════════╝
"""

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def run(label, cmd, critical=False):
    print(f"\n{'─'*55}")
    print(f"  ▶  {label}")
    print(f"{'─'*55}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  ✅ Done in {elapsed:.1f}s")
        return True
    else:
        print(f"  ⚠️  Finished with warnings ({elapsed:.1f}s)")
        if critical:
            print("  ❌ Critical step failed. Stopping.")
            sys.exit(1)
        return False


def check_package(pkg):
    """Return True if a Python package is importable."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {pkg}"],
        capture_output=True
    )
    return result.returncode == 0


def check_ollama():
    """Return True if Ollama is running and reachable."""
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False


def preflight_check():
    """Check environment and print a clear status table."""
    print("\n📋 PRE-FLIGHT CHECK")
    print("─" * 55)

    all_ok = True

    # Required packages
    required = ["pandas", "numpy", "sklearn", "joblib", "streamlit", "plotly"]
    optional = ["ollama"]

    for pkg in required:
        ok = check_package(pkg)
        status = "✅" if ok else "❌ MISSING"
        print(f"  {status}  {pkg}")
        if not ok:
            all_ok = False

    for pkg in optional:
        ok = check_package(pkg)
        status = "✅" if ok else "⚠️  optional"
        print(f"  {status}  {pkg} (Ollama LLM)")

    # Ollama server
    ollama_running = check_ollama()
    print(f"  {'✅' if ollama_running else '⚠️  offline'}  Ollama server (optional — fallback report used if offline)")

    # Model files
    model_trained = os.path.exists("models/bullwhip_model.pkl")
    print(f"  {'✅' if model_trained else '❌ NOT TRAINED'}  ML model (run --setup if missing)")
    if not model_trained:
        all_ok = False

    # Data files
    hist_exists = os.path.exists("data/historical_supply_chain.csv")
    print(f"  {'✅' if hist_exists else '❌ MISSING'}  Historical dataset (run --setup if missing)")

    print("─" * 55)

    if not all_ok:
        print("\n⚠️  Issues found. Fix them before running the pipeline:")
        if not check_package("pandas"):
            print("   pip install -r requirements.txt")
        if not model_trained:
            print("   python run_system.py --setup")
        print()
    else:
        print("  ✅ All systems ready!\n")

    return all_ok


def launch_dashboard():
    """Launch Streamlit using python -m streamlit (works inside venv)."""
    print("\n🌐 Launching Dashboard...")
    print("   URL: http://localhost:8501")
    print("   Press Ctrl+C to stop.\n")
    # Use sys.executable so it always uses the active venv's Python
    os.system(f'"{sys.executable}" -m streamlit run dashboard/app.py')


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    print(BANNER)
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python  : {sys.executable}")
    print(f"  Mode    : {' '.join(sys.argv[1:]) or 'full pipeline'}")

    # ── CHECK ONLY ─────────────────────────────────────────
    if "--check" in sys.argv:
        preflight_check()
        return

    # ── DASHBOARD ONLY ─────────────────────────────────────
    if "--dashboard" in sys.argv:
        launch_dashboard()
        return

    # ── SETUP MODE ─────────────────────────────────────────
    if "--setup" in sys.argv:
        print("\n🔧 SETUP MODE — Generate historical data + train model")
        run("Generate Historical Dataset (2020-2025)",
            f'"{sys.executable}" scripts/generate_historical_data.py', critical=True)
        run("Train ML Model (Random Forest, 200 trees)",
            f'"{sys.executable}" scripts/train_model.py', critical=True)
        print("\n✅ Setup complete!")
        print("   Now run: python run_system.py")
        return

    # ── FULL PIPELINE ──────────────────────────────────────

    # Pre-flight
    print()
    ok = preflight_check()
    if not ok:
        print("Run 'python run_system.py --setup' to fix missing components.\n")
        sys.exit(1)

    # PHASE 1 — Data collection
    print("\n📡 PHASE 1: DATA COLLECTION")
    run("Weather Collector",
        f'"{sys.executable}" weather/weather_collector.py')
    run("Commodity Price Collector",
        f'"{sys.executable}" commodity/commodity_collector.py')
    run("News Collector",
        f'"{sys.executable}" scraper/news_collector.py')

    # PHASE 2 — Processing
    print("\n⚙️  PHASE 2: DATA PROCESSING")
    run("Data Processing Agent  → market_events.csv",
        f'"{sys.executable}" agents/data_processing_agent.py')
    run("Risk Scoring Agent     → risk_score.csv",
        f'"{sys.executable}" agents/risk_scoring_agent.py')
    run("Product Risk Agent     → product_risk.csv",
        f'"{sys.executable}" agents/product_risk_agent.py')

    # PHASE 3 — ML + XAI
    print("\n🤖 PHASE 3: ML PREDICTION + EXPLAINABLE AI")
    run("ML Prediction Agent    → ml_prediction.csv + xai_explanation.json",
        f'"{sys.executable}" agents/ml_prediction_agent.py', critical=True)

    # PHASE 4 — LLM (optional, won't crash if Ollama is offline)
    print("\n📝 PHASE 4: AI NARRATIVE (Ollama — optional)")
    run("Market AI Agent        → AI supply chain report",
        f'"{sys.executable}" ai_agent/market_ai_agent.py')

    # PHASE 5 — Dashboard
    print("\n🌐 PHASE 5: DASHBOARD")
    launch_dashboard()


if __name__ == "__main__":
    main()
