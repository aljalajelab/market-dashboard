# breadth_rsi60_from_2003.py — يحفظ data/rsi60.csv
import os, time
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import yfinance as yf

START_DATE="2003-01-01"; RSI_PERIOD=14; THRESHOLD=60; BATCH_SIZE=50; PAUSE_SEC=1.0
BASE_DIR=Path(__file__).resolve().parent
DATA_DIR=BASE_DIR.parent/"data"; DATA_DIR.mkdir(exist_ok=True)

def read_all_tickers(folder):
    tickers=[]
    for csv_file in sorted(folder.glob("*.csv")):
        try:
            s=pd.read_csv(csv_file,header=None)[0].dropna().astype(str)
            tickers.extend(s.str.strip().str.upper().str.replace(".","-",regex=False).tolist())
        except: continue
    return sorted({t for t in tickers if t})

def compute_rsi(close,period=RSI_PERIOD):
    delta=close.diff(); gain=delta.clip(lower=0.0); loss=-delta.clip(upper=0.0)
    ag=gain.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
    al=loss.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
    return 100-(100/(1+ag/al.replace(0,pd.NA)))

def chunks(lst,n):
    for i in range(0,len(lst),n): yield lst[i:i+n]

def download_rsi_batch(batch):
    df=yf.download(batch,start=START_DATE,end=pd.Timestamp.today().normalize()+pd.Timedelta(days=1),progress=False,group_by="ticker",threads=True,auto_adjust=True)
    out=pd.DataFrame()
    if df is None or df.empty: return out
    if isinstance(df.columns,pd.MultiIndex):
        for t in batch:
            col=(t,"Close") if (t,"Close") in df.columns else ((t,"Adj Close") if (t,"Adj Close") in df.columns else None)
            if not col: continue
            close=df[col].dropna()
            if len(close)>=RSI_PERIOD+2: out[t]=compute_rsi(close)
    else:
        col="Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
        if col and len(df[col].dropna())>=RSI_PERIOD+2: out[batch[0]]=compute_rsi(df[col].dropna())
    return out

def main():
    tickers=read_all_tickers(BASE_DIR); frames=[]
    for i,batch in enumerate(chunks(tickers,BATCH_SIZE),1):
        r=download_rsi_batch(batch)
        if not r.empty: frames.append(r)
        time.sleep(PAUSE_SEC)
    rsi_df=pd.concat(frames,axis=1).sort_index()
    signals=(rsi_df>=THRESHOLD).astype(float)
    percent=(signals.sum(axis=1,min_count=1)/signals.notna().sum(axis=1))*100.0
    percent.name="RSI60_Breadth"
    out=percent.to_frame(); out.index.name="Date"
    out.to_csv(DATA_DIR/"rsi60.csv")
    print(f"✅ rsi60.csv — آخر قيمة: {percent.dropna().iloc[-1]:.2f}")

if __name__=="__main__": main()
