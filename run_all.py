"""
run_all.py — يشغل كل المؤشرات ويحفظ CSV لكل واحد
يُشغَّل تلقائياً كل يوم على GitHub Actions
"""
import subprocess, sys, os

SCRIPTS = [
    # SPX Breadth
    "scripts/spx_regime.py",
    "scripts/rsi_spread_live.py",
    "scripts/breadth_rsi50_from_2003.py",
    "scripts/breadth_rsi60_from_2003.py",
    "scripts/spx_mas_breadth_chart.py",
    "scripts/spx_mas_breadth_chart_no10_v2.py",
    "scripts/MacdBreadth_Yearly_2003.py",
    "scripts/AdxDIPlusBreadth.py",
    "scripts/breadth_m50.py",
    "scripts/M21.py",
    # Market Environment
    "scripts/market_env_composite.py",
    "scripts/MarketEnvComposite_QQQ.py",
    "scripts/spx_price_oscillator.py",
    # QQQ / NYSE
    "scripts/MacdNQ.py",
    "scripts/MacdNY.py",
    "scripts/SpreadrsiNY.py",
    # Seasonality & Cycles
    "scripts/spx500.py",
    "scripts/spx500_2014.py",
    "scripts/presidential_cycle.py",
    "scripts/qqqe_seasonality.py",
]

os.makedirs("data", exist_ok=True)

for script in SCRIPTS:
    print(f"\n{'='*50}")
    print(f"▶ Running: {script}")
    print('='*50)
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False
    )
    if result.returncode != 0:
        print(f"⚠️  {script} انتهى بخطأ — نكمل بالباقي")
    else:
        print(f"✅ {script} اكتمل")

print("\n✅ انتهى تشغيل كل المؤشرات")
