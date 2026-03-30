# spx_price_oscillator.py — يحفظ data/price_oscillator.csv
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import yfinance as yf
from pathlib import Path

DATA_DIR=Path(__file__).resolve().parent.parent/"data"; DATA_DIR.mkdir(exist_ok=True)
BASE_DIR=Path(__file__).resolve().parent

START="2014-01-01"; SMA_FAST=10; THRESH_UP=0.60; THRESH_DN=0.20
INDEX_TICKER="^GSPC"

# ابحث عن ملف الكون
for name in ["universe.csv","sp500_tickers.csv","spx_tickers.csv"]:
    p=BASE_DIR/name
    if p.exists():
        universe_path=p; break
else:
    universe_path=BASE_DIR/"universe.csv"

def read_universe(path):
    out=[]
    for line in path.read_text().splitlines():
        t=line.strip().upper()
        if t and not t.startswith("#"): out.append(t.replace(".","-"))
    return sorted(set(out))

universe=read_universe(universe_path)
all_tickers=list(set(universe+[INDEX_TICKER]))

raw=yf.download(all_tickers,start=START,auto_adjust=True,progress=False)
close=raw["Close"].copy()

px=close.reindex(columns=universe).ffill().bfill()
px_sma=px.rolling(SMA_FAST).mean()
above=(px>px_sma)&px_sma.notna()
count_above=above.sum(axis=1)
count_total=px_sma.notna().sum(axis=1).replace(0,np.nan)
osc=(count_above/count_total).dropna()

idx_price=close[INDEX_TICKER].dropna() if INDEX_TICKER in close.columns else None

out=pd.DataFrame({"SPX":idx_price.reindex(osc.index) if idx_price is not None else np.nan,
                  "Oscillator":osc,"Threshold_Up":THRESH_UP,"Threshold_Down":THRESH_DN})
out.index.name="Date"
out.to_csv(DATA_DIR/"price_oscillator.csv")
print(f"✅ price_oscillator.csv — آخر قيمة: {osc.iloc[-1]:.3f}")
