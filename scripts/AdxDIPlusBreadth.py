# AdxDIPlusBreadth.py — يحفظ data/adx.csv
import os, glob, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import yfinance as yf

BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "..", "data")
CACHE_DIR  = os.path.join(BASE_DIR, "_cache_di")
START_DATE = "2003-01-01"
END_DATE   = pd.Timestamp.today().strftime("%Y-%m-%d")
DI_LEN     = 13
THRESHOLD  = 25.0
BATCH_SIZE = 150
MAX_RETRIES= 3
RETRY_SLEEP= 3

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def normalize_ticker(t):
    t = str(t).strip().upper()
    if t in ("","TICKER","SYMBOL","TICKERS","SECID","RIC","ISIN"): return None
    t = t.replace(" ","").replace(".PR","-P").replace(".","-").replace("^","")
    if any(bad in t for bad in ("TICKERN","TICKER_")): return None
    return t

def read_membership_by_year():
    files = sorted(glob.glob(os.path.join(BASE_DIR,"*.csv")) + glob.glob(os.path.join(BASE_DIR,"*.txt")))
    by_year = {}
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        if not name.isdigit(): continue
        year = int(name)
        df = pd.read_csv(p, header=None, usecols=[0], engine="python", on_bad_lines="skip")
        vals = [normalize_ticker(x) for x in df.iloc[:,0].tolist() if normalize_ticker(x) and normalize_ticker(x) != "TICKER"]
        vals = sorted(set(vals))
        if vals: by_year[year] = vals
    if not by_year: raise RuntimeError("No yearly lists found.")
    return by_year

def download_prices(tickers):
    prices, need = {}, []
    for t in sorted(set(tickers)):
        cp = os.path.join(CACHE_DIR, f"{t}.parquet")
        if os.path.exists(cp):
            try:
                df = pd.read_parquet(cp)
                if not df.empty: prices[t] = df; continue
            except: pass
        need.append(t)
    for i in range(0, len(need), BATCH_SIZE):
        batch = need[i:i+BATCH_SIZE]
        for attempt in range(MAX_RETRIES):
            try:
                hist = yf.download(batch, start=START_DATE, end=END_DATE, progress=False, group_by="ticker", auto_adjust=False, threads=True)
                for t in batch:
                    try:
                        sub = hist[t] if isinstance(hist.columns, pd.MultiIndex) else hist
                        sub = sub.rename(columns=str)[["High","Low","Close"]].copy()
                        sub.index = pd.to_datetime(sub.index)
                        sub = sub.loc[~sub.index.duplicated()].sort_index()
                        sub.columns = ["high","low","close"]
                        if not sub.empty:
                            prices[t] = sub
                            sub.to_parquet(os.path.join(CACHE_DIR, f"{t}.parquet"))
                    except: pass
                break
            except:
                time.sleep(RETRY_SLEEP*(attempt+1))
    return prices

def wilder_rma(s, n):
    alpha = 1.0/n
    sma = s.rolling(n, min_periods=n).mean()
    r = s.copy()*np.nan
    idx = np.where(~sma.isna())[0]
    if len(idx)==0: return r
    i0 = idx[0]; r.iloc[i0] = sma.iloc[i0]
    for i in range(i0+1, len(s)):
        r.iloc[i] = alpha*s.iloc[i] + (1-alpha)*r.iloc[i-1]
    return r

def di_plus(df, n=DI_LEN):
    g = df.sort_index().copy()
    up = g["high"].diff(); dn = g["low"].diff().abs()
    plus_dm = np.where((up>dn)&(up>0), up, 0.0)
    tr = pd.concat([g["high"]-g["low"], (g["high"]-g["close"].shift()).abs(), (g["low"]-g["close"].shift()).abs()], axis=1).max(axis=1)
    return 100.0*(wilder_rma(pd.Series(plus_dm, index=g.index), n)/wilder_rma(tr, n))

def main():
    membership = read_membership_by_year()
    all_tickers = sorted({t for lst in membership.values() for t in lst})
    prices = download_prices(all_tickers)

    recs = []
    for t, df in prices.items():
        dip = di_plus(df)
        recs.append(pd.DataFrame({"date": dip.index, "symbol": t, "di_plus": dip.values}).dropna())
    di_long = pd.concat(recs, ignore_index=True)

    rows = [(y,s) for y,lst in membership.items() for s in lst]
    universe = pd.DataFrame(rows, columns=["year","symbol"]).drop_duplicates()
    di_long["year"] = di_long["date"].dt.year
    in_u = di_long.merge(universe, on=["year","symbol"], how="inner")
    in_u["over"] = in_u["di_plus"] > THRESHOLD

    daily = in_u.groupby("date").agg(count=("symbol","nunique"), over=("over","sum")).reset_index()
    daily["pct_over"] = (daily["over"]/daily["count"])*100.0
    daily = daily.sort_values("date").set_index("date")

    out = daily[["pct_over"]].rename(columns={"pct_over": "ADX_DI_Plus_Breadth"})
    out.index.name = "Date"
    out.to_csv(os.path.join(DATA_DIR, "adx.csv"))
    print(f"✅ adx.csv — آخر قيمة: {out['ADX_DI_Plus_Breadth'].iloc[-1]:.2f}")

if __name__ == "__main__":
    main()
