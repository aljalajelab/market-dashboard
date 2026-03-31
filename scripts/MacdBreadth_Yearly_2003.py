import time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import yfinance as yf

START_DATE=pd.Timestamp("2003-01-01"); FAST_EMA=12; SLOW_EMA=26; MIN_UNIVERSE=50
BATCH_SIZE=150; MAX_RETRIES=3; RETRY_SLEEP=3
BASE_DIR=Path(__file__).resolve().parent
DATA_DIR=BASE_DIR.parent/"data"; DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR=BASE_DIR/"_cache_macd"; CACHE_DIR.mkdir(exist_ok=True)

def load_tickers():
    for name in ["sp500_tickers.csv","universe.csv"]:
        p = BASE_DIR/name
        if p.is_file():
            df = pd.read_csv(p)
            col = next((c for c in df.columns if str(c).strip().lower() in
                       ("ticker","tickers","symbol","symbols")), df.columns[0])
            tickers = df[col].astype(str).str.strip().str.upper()
            tickers = tickers.replace("NAN",np.nan).dropna().drop_duplicates()
            tickers = tickers[tickers.str.match(r"^[A-Z0-9.\-^]+$")]
            return tickers.str.replace(".","-",regex=False).unique().tolist()
    raise FileNotFoundError("ما وجدت sp500_tickers.csv أو universe.csv")

def download_close(tickers, start):
    prices, need = {}, []
    for t in sorted(set(tickers)):
        cp = CACHE_DIR/f"{t}.parquet"
        if cp.exists():
            try:
                df = pd.read_parquet(cp)
                if not df.empty: prices[t]=df; continue
            except: pass
        need.append(t)
    fetch_start = start - pd.Timedelta(days=400)
    end = pd.Timestamp.today().normalize()
    for i in range(0, len(need), BATCH_SIZE):
        batch = need[i:i+BATCH_SIZE]; tries = 0
        while tries < MAX_RETRIES:
            try:
                df = yf.download(tickers=batch, start=fetch_start,
                                 end=end+pd.Timedelta(days=1),
                                 group_by="ticker", auto_adjust=True,
                                 progress=False, threads=True)
                for t in batch:
                    try:
                        sub = df[t] if isinstance(df.columns,pd.MultiIndex) else df
                        base = "Close" if "Close" in sub.columns else "Adj Close"
                        close = sub[[base]].rename(columns={base:"close"}).copy()
                        close.index = pd.to_datetime(close.index)
                        close = close.loc[~close.index.duplicated()].sort_index()
                        if not close.empty:
                            prices[t]=close
                            close.to_parquet(CACHE_DIR/f"{t}.parquet")
                    except: pass
                break
            except: tries+=1; time.sleep(RETRY_SLEEP*tries)
    return prices

def compute_breadth(prices):
    recs = []
    for t, df in prices.items():
        ema_f = df["close"].ewm(span=FAST_EMA,adjust=False,min_periods=FAST_EMA).mean()
        ema_s = df["close"].ewm(span=SLOW_EMA,adjust=False,min_periods=SLOW_EMA).mean()
        macd = ema_f - ema_s
        recs.append(pd.DataFrame({
            "date": macd.index, "macd": macd.values,
            "vf": ema_f.isna().values, "vs": ema_s.isna().values
        }).dropna(subset=["macd"]))
    if not recs: raise RuntimeError("ما في بيانات للحساب")
    long = pd.concat(recs, ignore_index=True)
    valid = (~long["vf"]) & (~long["vs"])
    cond  = (long["macd"] > 0) & valid
    daily = long.assign(valid=valid, cond=cond).groupby("date").agg(
        vc=("valid","sum"), ok=("cond","sum")
    ).reset_index()
    daily["pct"] = (daily["ok"] / daily["vc"].replace(0,np.nan)) * 100.0
    daily.loc[daily["vc"] < MIN_UNIVERSE, "pct"] = np.nan
    return daily.set_index("date")["pct"].sort_index()[
        lambda s: s.index >= START_DATE
    ].rename("MACD_Breadth")

def main():
    tickers = load_tickers()
    print(f"✅ عدد التيكرات: {len(tickers)}")
    prices = download_close(tickers, START_DATE)
    if not prices: raise RuntimeError("فشل تحميل البيانات")
    series = compute_breadth(prices)
    if series.dropna().empty: raise RuntimeError("series فارغة")
    out = series.to_frame(); out.index.name="Date"
    out.to_csv(DATA_DIR/"macd_spx.csv")
    print(f"✅ macd_spx.csv — آخر قيمة: {series.dropna().iloc[-1]:.2f}")

if __name__=="__main__": main()
