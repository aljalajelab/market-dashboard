import os, time, datetime as dt
import pandas as pd
import yfinance as yf
import matplotlib; matplotlib.use("Agg")

BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "..", "data")
START_DATE = dt.date(2003,1,1)
TODAY      = dt.date.today()
BATCH_SIZE=60; PAUSE_SEC=0.8; RSI_PERIOD=14

def load_tickers():
    for name in ["sp500_tickers.csv","universe.csv"]:
        p = os.path.join(BASE_DIR, name)
        if os.path.isfile(p):
            df = pd.read_csv(p)
            col = next((c for c in df.columns if str(c).strip().lower() in
                       ("ticker","tickers","symbol","symbols")), df.columns[0])
            tickers = df[col].astype(str).str.strip().str.upper()
            tickers = tickers.replace({"NAN":None}).dropna().drop_duplicates()
            tickers = tickers[tickers.str.match(r"^[A-Z0-9.\-^]+$")]
            return sorted(tickers.str.replace(".","-",regex=False).unique().tolist())
    raise FileNotFoundError("ما وجدت sp500_tickers.csv أو universe.csv")

def chunks(lst, n):
    for i in range(0, len(lst), n): yield lst[i:i+n]

def download_close(tickers):
    frames = []
    for batch in chunks(tickers, BATCH_SIZE):
        for attempt in (1,2):
            try:
                data = yf.download(batch, start=START_DATE.isoformat(),
                                   end=(TODAY+dt.timedelta(days=1)).isoformat(),
                                   progress=False, auto_adjust=False,
                                   group_by="ticker", threads=True)
                break
            except:
                if attempt==2: raise
                time.sleep(2)
        if isinstance(data.columns, pd.MultiIndex):
            close = pd.DataFrame({t: data[(t,"Close")] for t in batch if (t,"Close") in data})
        else:
            close = pd.DataFrame({batch[0]: data["Close"]})
        frames.append(close)
        time.sleep(PAUSE_SEC)
    if not frames: return pd.DataFrame()
    close = pd.concat(frames, axis=1)
    close.index = pd.to_datetime(close.index)
    return close.sort_index().dropna(axis=1, how="all")

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0.0); loss = -delta.clip(upper=0.0)
    ag = gain.rolling(period, min_periods=period).mean()
    al = loss.rolling(period, min_periods=period).mean()
    return 100 - (100 / (1 + ag/al))

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    tickers = load_tickers()
    print(f"✅ عدد التيكرات: {len(tickers)}")
    close_all = download_close(tickers)
    if close_all.empty: raise RuntimeError("فشل التنزيل")

    sma200    = close_all.rolling(200, min_periods=200).mean()
    above200  = (close_all > sma200)
    rsi50_flags = pd.DataFrame(index=close_all.index)
    rsi60_flags = pd.DataFrame(index=close_all.index)

    for t in close_all.columns:
        c = close_all[t].dropna()
        if c.empty: continue
        rsi = compute_rsi(c, RSI_PERIOD).reindex(close_all.index)
        rsi50_flags[t] = (rsi >= 50)
        rsi60_flags[t] = (rsi >= 60)

    valid = (~close_all.isna()).sum(axis=1).replace(0, pd.NA)
    LT = (above200.sum(axis=1)   / valid) * 100.0
    MT = (rsi50_flags.sum(axis=1)/ valid) * 100.0
    ST = (rsi60_flags.sum(axis=1)/ valid) * 100.0

    score      = 0.5*LT + 0.3*MT + 0.2*ST
    score_ema10= score.ewm(span=10, min_periods=5).mean()

    out = pd.DataFrame({
        "LT_pct_above_SMA200" : LT,
        "MT_pct_RSI_ge_50"    : MT,
        "ST_pct_RSI_ge_60"    : ST,
        "RegimeScore"         : score,
        "RegimeScore_EMA10"   : score_ema10,
    })
    out.index.name = "Date"
    out.to_csv(os.path.join(DATA_DIR,"regime.csv"))
    print(f"✅ regime.csv — آخر قيمة: {score.dropna().iloc[-1]:.2f}")

if __name__=="__main__": main()
