# MarketEnvComposite_QQQ.py — يحفظ data/market_env_qqq.csv
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import yfinance as yf
from pathlib import Path

START="2014-01-01"; THRESHOLD=5; SMOOTH_DAYS=1
DATA_DIR=Path(__file__).resolve().parent.parent/"data"; DATA_DIR.mkdir(exist_ok=True)

T_QQQ="QQQ"; T_QQQE="QQQE"; T_IWM="IWM"; T_SPHB="SPHB"; T_SPLV="SPLV"
T_HYG="HYG"; T_LQD="LQD"; T_GLD="GLD"; T_JJC="JJC"; T_VXN="^VXN"
tickers=[T_QQQ,T_QQQE,T_IWM,T_SPHB,T_SPLV,T_HYG,T_LQD,T_GLD,T_JJC,T_VXN]

data=yf.download(tickers,start=START,auto_adjust=True,progress=False)["Close"].ffill().bfill()

def sma(s,n): return s.rolling(n).mean()
def roc(s,n): return s.pct_change(n)
def above(x,y):
    out=(x>y).astype(int)
    return out.where(np.isfinite(out),0)

qqq=data[T_QQQ]; vxn=data[T_VXN]
signals=pd.DataFrame({
    "QQQ>50":    above(qqq,sma(qqq,50)),
    "QQQ>200":   above(qqq,sma(qqq,200)),
    "QQQE/QQQ>50": above(data[T_QQQE]/qqq,sma(data[T_QQQE]/qqq,50)),
    "IWM/QQQ>50":  above(data[T_IWM]/qqq,sma(data[T_IWM]/qqq,50)),
    "SPHB/SPLV>50":above(data[T_SPHB]/data[T_SPLV],sma(data[T_SPHB]/data[T_SPLV],50)),
    "HYG/LQD>50":  above(data[T_HYG]/data[T_LQD],sma(data[T_HYG]/data[T_LQD],50)),
    "VXN<MA50&25": ((vxn<sma(vxn,50))&(vxn<25)).astype(int),
    "ROC20>0":     (roc(qqq,20)>0).astype(int),
    "JJC/GLD>50":  above(data[T_JJC]/data[T_GLD],sma(data[T_JJC]/data[T_GLD],50)),
    "MOM6M>0":     (roc(qqq,126)>0).astype(int),
}).dropna()

composite=signals.sum(axis=1).rolling(SMOOTH_DAYS,min_periods=1).mean()
out=pd.DataFrame({"QQQ":qqq.reindex(composite.index),"Composite":composite,"Threshold":THRESHOLD})
out.index.name="Date"
out.to_csv(DATA_DIR/"market_env_qqq.csv")
print(f"✅ market_env_qqq.csv — آخر قيمة: {composite.iloc[-1]:.1f}/10")
