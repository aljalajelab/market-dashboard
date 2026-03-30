# market_env_composite.py — يحفظ data/market_env_spx.csv
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import yfinance as yf
from pathlib import Path
import time

START = "2014-01-01"
THRESHOLD = 5
SMOOTH_DAYS = 1
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
T_SPY="SPY"; T_RSP="RSP"; T_IWM="IWM"; T_SPHB="SPHB"; T_SPLV="SPLV"
T_HYG="HYG"; T_LQD="LQD"; T_GLD="GLD"; T_JJC="JJC"; T_VIX="^VIX"
tickers=[T_SPY,T_RSP,T_IWM,T_SPHB,T_SPLV,T_HYG,T_LQD,T_GLD,T_JJC,T_VIX]

def download_in_batches(tickers, start, batch_size=5, delay=3):
    all_data = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, start=start, auto_adjust=True, progress=False)
            all_data.append(data)
        except Exception as e:
            print(f"خطأ في الدفعة {i}: {e}")
        time.sleep(delay)
    return pd.concat(all_data, axis=1) if all_data else pd.DataFrame()

raw = download_in_batches(tickers, START)
data = raw["Close"].ffill().bfill()

def sma(s,n): return s.rolling(n).mean()
def roc(s,n): return s.pct_change(n)
def above(x,y):
    out=(x>y).astype(int)
    return out.where(np.isfinite(out),0)
spy=data[T_SPY]
signals=pd.DataFrame({
    "SPY>50":    above(spy, sma(spy,50)),
    "SPY>200":   above(spy, sma(spy,200)),
    "RSP/SPY>50": above(data[T_RSP]/spy, sma(data[T_RSP]/spy,50)),
    "IWM/SPY>50": above(data[T_IWM]/spy, sma(data[T_IWM]/spy,50)),
    "SPHB/SPLV>50": above(data[T_SPHB]/data[T_SPLV], sma(data[T_SPHB]/data[T_SPLV],50)),
    "HYG/LQD>50": above(data[T_HYG]/data[T_LQD], sma(data[T_HYG]/data[T_LQD],50)),
    "VIX<MA50&20": ((data[T_VIX]<sma(data[T_VIX],50))&(data[T_VIX]<20)).astype(int),
    "ROC20>0":   (roc(spy,20)>0).astype(int),
    "JJC/GLD>50": above(data[T_JJC]/data[T_GLD], sma(data[T_JJC]/data[T_GLD],50)),
    "MOM6M>0":   (roc(spy,126)>0).astype(int),
}).dropna()
composite=signals.sum(axis=1).rolling(SMOOTH_DAYS,min_periods=1).mean()
out=pd.DataFrame({
    "SPY": spy.reindex(composite.index),
    "Composite": composite,
    "Threshold": THRESHOLD
})
out.index.name="Date"
out.to_csv(DATA_DIR/"market_env_spx.csv")
print(f"✅ market_env_spx.csv — آخر قيمة: {composite.iloc[-1]:.1f}/10")
