# rsi_spread_live.py — يحفظ data/rsi_spread_spx.csv
import os, time, datetime as dt
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import yfinance as yf

BASE_DIR=os.path.abspath(os.path.dirname(__file__))
DATA_DIR=os.path.join(BASE_DIR,"..","data"); os.makedirs(DATA_DIR,exist_ok=True)
START_DATE=dt.date(2003,1,1); TODAY=dt.date.today()
RSI_PERIOD=14; BATCH_SIZE=60; PAUSE_SEC=3; MIN_UNIVERSE=200

def read_universe():
    for name in ["universe.csv","sp500_tickers.csv","spx_tickers.csv"]:
        p=os.path.join(BASE_DIR,name)
        if os.path.isfile(p):
            df=pd.read_csv(p); col=df.columns[0]
            s=df[col].astype(str).str.strip().str.upper().replace({"NAN":None}).dropna().drop_duplicates()
            s=s[s.str.match(r"^[A-Z0-9.\-^]+$")].str.replace(".","-",regex=False)
            return sorted(s.unique().tolist())
    return []

def rsi_wilder(close,period=14):
    delta=close.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    ag=gain.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
    al=loss.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
    return 100-(100/(1+ag/al))

def chunks(lst,n):
    for i in range(0,len(lst),n): yield lst[i:i+n]

def download_rsi_batch(batch):
    df=yf.download(batch,start=START_DATE.isoformat(),end=(TODAY+dt.timedelta(days=1)).isoformat(),progress=False,group_by="ticker",threads=True)
    out=pd.DataFrame()
    if df is None or df.empty: return out
    if isinstance(df.columns,pd.MultiIndex):
        for t in batch:
            col=(t,"Close") if (t,"Close") in df.columns else ((t,"Adj Close") if (t,"Adj Close") in df.columns else None)
            if not col: continue
            close=df[col].dropna()
            if len(close)>=RSI_PERIOD*3: out[t]=rsi_wilder(close)
    else:
        col="Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
        if col:
            close=df[col].dropna()
            if len(close)>=RSI_PERIOD*3: out[batch[0]]=rsi_wilder(close)
    return out

def main():
    union=read_universe()
    print(f"عدد الأسهم: {len(union)}")
    frames=[]
    for batch in chunks(union,BATCH_SIZE):
        r=download_rsi_batch(batch)
        if not r.empty: frames.append(r)
        time.sleep(PAUSE_SEC)
    if not frames:
        print("❌ ما نزلت أي بيانات")
        return
    rsi_df=pd.concat(frames,axis=1).sort_index()
    universe=rsi_df.notna().sum(axis=1)
    pct70=(rsi_df>=70).sum(axis=1)/universe*100.0
    pct30=(rsi_df<=30).sum(axis=1)/universe*100.0
    spread=(pct70-pct30).rename("RSI_Spread_SPX")
    spread[universe<MIN_UNIVERSE]=np.nan
    out=spread.dropna().to_frame(); out.index.name="Date"
    out.to_csv(os.path.join(DATA_DIR,"rsi_spread_spx.csv"))
    print(f"✅ rsi_spread_spx.csv — آخر قيمة: {spread.dropna().iloc[-1]:.2f}")

if __name__=="__main__":
    main()
