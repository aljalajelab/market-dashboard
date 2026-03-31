import os, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import yfinance as yf

BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "..", "data")
CACHE_DIR  = os.path.join(BASE_DIR, "_cache_di")
START_DATE = "2003-01-01"
END_DATE   = pd.Timestamp.today().strftime("%Y-%m-%d")
DI_LEN=13; THRESHOLD=25.0; BATCH_SIZE=150; MAX_RETRIES=3; RETRY_SLEEP=3

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)

def load_tickers():
    for name in ["sp500_tickers.csv","universe.csv"]:
        p = os.path.join(BASE_DIR, name)
        if os.path.isfile(p):
            df = pd.read_csv(p)
            col = next((c for c in df.columns if str(c).strip().lower() in
                       ("ticker","tickers","symbol","symbols")), df.columns[0])
            tickers = df[col].astype(str).str.strip().str.upper()
            tickers = tickers.replace("NAN",np.nan).dropna().drop_duplicates()
            tickers = tickers[tickers.str.match(r"^[A-Z0-9.\-^]+$")]
            return tickers.str.replace(".","-",regex=False).unique().tolist()
    raise FileNotFoundError("ما وجدت sp500_tickers.csv أو universe.csv")

def download_prices(tickers):
    prices, need = {}, []
    for t in sorted(set(tickers)):
        cp = os.path.join(CACHE_DIR, f"{t}.parquet")
        if os.path.exists(cp):
            try:
                df = pd.read_parquet(cp)
                if not df.empty: prices[t]=df; continue
            except: pass
        need.append(t)
    for i in range(0, len(need), BATCH_SIZE):
        batch = need[i:i+BATCH_SIZE]
        for attempt in range(MAX_RETRIES):
            try:
                hist = yf.download(batch, start=START_DATE, end=END_DATE,
                                   progress=False, group_by="ticker",
                                   auto_adjust=False, threads=True)
                for t in batch:
                    try:
                        sub = hist[t] if isinstance(hist.columns,pd.MultiIndex) else hist
                        sub = sub.rename(columns=str)[["High","Low","Close"]].copy()
                        sub.index = pd.to_datetime(sub.index)
                        sub = sub.loc[~sub.index.duplicated()].sort_index()
                        sub.columns = ["high","low","close"]
                        if not sub.empty:
                            prices[t]=sub
                            sub.to_parquet(os.path.join(CACHE_DIR,f"{t}.parquet"))
                    except: pass
                break
            except: time.sleep(RETRY_SLEEP*(attempt+1))
    return prices

def wilder_rma(s, n):
    alpha = 1.0/n
    sma = s.rolling(n, min_periods=n).mean()
    r = s.copy()*np.nan
    idx = np.where(~sma.isna())[0]
    if len(idx)==0: return r
    i0=idx[0]; r.iloc[i0]=sma.iloc[i0]
    for i in range(i0+1, len(s)):
        r.iloc[i] = alpha*s.iloc[i] + (1-alpha)*r.iloc[i-1]
    return r

def di_plus(df, n=DI_LEN):
    g = df.sort_index().copy()
    up = g["high"].diff(); dn = g["low"].diff().abs()
    plus_dm = np.where((up>dn)&(up>0), up, 0.0)
    tr = pd.concat([g["high"]-g["low"],
                    (g["high"]-g["close"].shift()).abs(),
                    (g["low"]-g["close"].shift()).abs()], axis=1).max(axis=1)
    return 100.0*(wilder_rma(pd.Series(plus_dm,index=g.index),n)/wilder_rma(tr,n))

def main():
    tickers = load_tickers()
    print(f"✅ عدد التيكرات: {len(tickers)}")
    prices  = download_prices(tickers)
    if not prices: raise RuntimeError("فشل تحميل البيانات")

    recs = []
    for t, df in prices.items():
        dip = di_plus(df)
        recs.append(pd.DataFrame({"date":dip.index,"symbol":t,"di_plus":dip.values}).dropna())
    if not recs: raise RuntimeError("ما في بيانات بعد الحساب")

    di_long = pd.concat(recs, ignore_index=True)
    di_long["over"] = di_long["di_plus"] > THRESHOLD
    daily = di_long.groupby("date").agg(
        count=("symbol","nunique"), over=("over","sum")
    ).reset_index()
    daily["pct_over"] = (daily["over"]/daily["count"])*100.0
    daily = daily.sort_values("date").set_index("date")

    out = daily[["pct_over"]].rename(columns={"pct_over":"ADX_DI_Plus_Breadth"})
    out.index.name = "Date"
    out.to_csv(os.path.join(DATA_DIR,"adx.csv"))
    print(f"✅ adx.csv — آخر قيمة: {out['ADX_DI_Plus_Breadth'].iloc[-1]:.2f}")

if __name__=="__main__": main()
