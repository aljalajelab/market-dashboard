import time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import yfinance as yf

START_DATE="2017-01-01"; RSI_PERIOD=14; BATCH_SIZE=60; PAUSE_SEC=0.8; MIN_UNIVERSE=50
BASE_DIR=Path(__file__).resolve().parent
DATA_DIR=BASE_DIR.parent/"data"; DATA_DIR.mkdir(exist_ok=True)

def load_tickers():
    # يبحث عن ملفات التيكرات بالترتيب
    for name in ["sp500_tickers.csv", "universe.csv"]:
        p = BASE_DIR / name
        if p.is_file():
            df = pd.read_csv(p)
            col = next((c for c in df.columns if str(c).strip().lower() in
                       ("ticker","tickers","symbol","symbols")), df.columns[0])
            tickers = df[col].astype(str).str.strip().str.upper()
            tickers = tickers.replace("NAN", np.nan).dropna().drop_duplicates()
            tickers = tickers[tickers.str.match(r"^[A-Z0-9.\-^]+$")]
            return tickers.str.replace(".", "-", regex=False).unique().tolist()
    raise FileNotFoundError("ما وجدت sp500_tickers.csv أو universe.csv")

def rsi_wilder(close, period=RSI_PERIOD):
    delta=close.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    ag=gain.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
    al=loss.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
    return 100-(100/(1+ag/al.replace(0,np.nan)))

def chunks(lst, n):
    for i in range(0, len(lst), n): yield lst[i:i+n]

def download_rsi_batch(batch):
    df = yf.download(batch, start=START_DATE, progress=False, group_by="ticker", threads=True)
    out = pd.DataFrame()
    if df is None or df.empty: return out
    if isinstance(df.columns, pd.MultiIndex):
        for t in batch:
            col = next(((t,c) for c in ("Close","Adj Close") if (t,c) in df.columns), None)
            if not col: continue
            close = df[col].dropna()
            if len(close) >= RSI_PERIOD*3: out[t] = rsi_wilder(close)
    else:
        col = next((c for c in ("Close","Adj Close") if c in df.columns), None)
        if col:
            close = df[col].dropna()
            if len(close) >= RSI_PERIOD*3: out[batch[0]] = rsi_wilder(close)
    return out

def main():
    tickers = load_tickers()
    print(f"✅ عدد التيكرات: {len(tickers)}")
    frames = []
    for batch in chunks(tickers, BATCH_SIZE):
        r = download_rsi_batch(batch)
        if not r.empty: frames.append(r)
        time.sleep(PAUSE_SEC)
    if not frames: raise RuntimeError("فشل تحميل أي بيانات")
    rsi_df = pd.concat(frames, axis=1).sort_index()
    universe = rsi_df.notna().sum(axis=1)
    pct70 = (rsi_df>=70).sum(axis=1)/universe*100.0
    pct30 = (rsi_df<=30).sum(axis=1)/universe*100.0
    spread = (pct70-pct30).rename("RSI_Spread_NY")
    spread[universe<MIN_UNIVERSE] = np.nan
    out = spread.dropna().to_frame(); out.index.name="Date"
    out.to_csv(DATA_DIR/"rsi_spread_ny.csv")
    print(f"✅ rsi_spread_ny.csv — آخر قيمة: {spread.dropna().iloc[-1]:.2f}")

if __name__=="__main__": main()
