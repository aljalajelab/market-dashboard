# MacdBreadth_Yearly_2003.py — يحفظ data/macd_spx.csv
import os, glob, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import yfinance as yf

START_DATE=pd.Timestamp("2003-01-01"); FAST_EMA=12; SLOW_EMA=26; MIN_UNIVERSE=350
BATCH_SIZE=150; MAX_RETRIES=3; RETRY_SLEEP=3
BASE_DIR=Path(__file__).resolve().parent
DATA_DIR=BASE_DIR.parent/"data"; DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR=BASE_DIR/"_cache_macd"; CACHE_DIR.mkdir(exist_ok=True)

def normalize_ticker(t):
    t=str(t).strip().upper()
    if t in ("","TICKER","SYMBOL","TICKERS","SECID","RIC","ISIN"): return None
    t=t.replace(" ","").replace(".PR","-P").replace(".","-").replace("^","")
    return None if any(bad in t for bad in ("TICKERN","TICKER_")) else t

def read_membership():
    by_year={}
    for p in sorted(glob.glob(str(BASE_DIR/"*.csv"))+glob.glob(str(BASE_DIR/"*.txt"))):
        name=Path(p).stem
        if not name.isdigit(): continue
        year=int(name)
        df=pd.read_csv(p,header=None,usecols=[0],engine="python",on_bad_lines="skip")
        vals=sorted({normalize_ticker(x) for x in df.iloc[:,0].tolist() if normalize_ticker(x) and normalize_ticker(x)!="TICKER"})
        if vals: by_year[year]=vals
    if not by_year: raise RuntimeError("ما وجدت ملفات سنوية.")
    return by_year

def download_close(tickers, start):
    prices,need={},[]
    for t in sorted(set(tickers)):
        cp=CACHE_DIR/f"{t}.parquet"
        if cp.exists():
            try:
                df=pd.read_parquet(cp)
                if not df.empty: prices[t]=df; continue
            except: pass
        need.append(t)
    fetch_start=start-pd.Timedelta(days=400); end=pd.Timestamp.today().normalize()
    for i in range(0,len(need),BATCH_SIZE):
        batch=need[i:i+BATCH_SIZE]; tries=0
        while tries<MAX_RETRIES:
            try:
                df=yf.download(tickers=batch,start=fetch_start,end=end+pd.Timedelta(days=1),group_by="ticker",auto_adjust=True,progress=False,threads=True)
                for t in batch:
                    try:
                        sub=df[t] if isinstance(df.columns,pd.MultiIndex) else df
                        base="Close" if "Close" in sub.columns else "Adj Close"
                        close=sub[[base]].rename(columns={base:"close"}).copy()
                        close.index=pd.to_datetime(close.index); close=close.loc[~close.index.duplicated()].sort_index()
                        if not close.empty: prices[t]=close; close.to_parquet(CACHE_DIR/f"{t}.parquet")
                    except: pass
                break
            except: tries+=1; time.sleep(RETRY_SLEEP*tries)
    return prices

def compute_breadth(pr, membership):
    recs=[]
    for t,df in pr.items():
        ema_f=df["close"].ewm(span=FAST_EMA,adjust=False,min_periods=FAST_EMA).mean()
        ema_s=df["close"].ewm(span=SLOW_EMA,adjust=False,min_periods=SLOW_EMA).mean()
        macd=ema_f-ema_s
        recs.append(pd.DataFrame({"date":macd.index,"symbol":t,"macd":macd.values,"vf":ema_f.isna().values,"vs":ema_s.isna().values}).dropna(subset=["macd"]))
    long=pd.concat(recs,ignore_index=True)
    rows=[(y,s) for y,lst in membership.items() for s in lst]
    universe=pd.DataFrame(rows,columns=["year","symbol"]).drop_duplicates()
    long["year"]=pd.to_datetime(long["date"]).dt.year
    in_u=long.merge(universe,on=["year","symbol"],how="inner")
    valid=(~in_u["vf"])&(~in_u["vs"]); cond=(in_u["macd"]>0)&valid
    daily=in_u.assign(valid=valid,cond=cond).groupby("date").agg(vc=("valid","sum"),ok=("cond","sum")).reset_index()
    daily["pct"]=(daily["ok"]/daily["vc"].replace(0,np.nan))*100.0
    daily.loc[daily["vc"]<MIN_UNIVERSE,"pct"]=np.nan
    return daily.set_index("date")["pct"].sort_index()[lambda s:s.index>=START_DATE].rename("MACD_Breadth")

def main():
    membership=read_membership(); all_tickers=sorted({t for v in membership.values() for t in v})
    prices=download_close(all_tickers,START_DATE); series=compute_breadth(prices,membership)
    out=series.to_frame(); out.index.name="Date"
    out.to_csv(DATA_DIR/"macd_spx.csv")
    print(f"✅ macd_spx.csv — آخر قيمة: {series.dropna().iloc[-1]:.2f}")

if __name__=="__main__": main()
