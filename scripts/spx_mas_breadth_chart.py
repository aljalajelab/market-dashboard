import os, time, datetime as dt
import pandas as pd
import matplotlib; matplotlib.use("Agg")

START_DATE = dt.date(2003,1,1)
TODAY = dt.date.today()
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR,"..","data")
os.makedirs(DATA_DIR, exist_ok=True)

def load_tickers():
    for name in ["sp500_tickers.csv","universe.csv"]:
        p = os.path.join(BASE_DIR, name)
        if os.path.isfile(p):
            df = pd.read_csv(p)
            col = next((c for c in df.columns if str(c).strip().lower() in
                       ("ticker","tickers","symbol","symbols")), df.columns[0])
            tickers = df[col].astype(str).str.strip().str.upper().replace({"NAN":None}).dropna().drop_duplicates()
            return sorted(tickers[tickers.str.match(r"^[A-Z0-9.\-^]+$")].str.replace(".","-",regex=False).unique().tolist())
    return []

def download_prices(tickers, start, end):
    import yfinance as yf
    parts = []
    for i in range(0, len(tickers), 60):
        chunk = tickers[i:i+60]
        for attempt in (1,2):
            try:
                data = yf.download(chunk, start=start.isoformat(), end=(end+dt.timedelta(days=1)).isoformat(),
                                   progress=False, auto_adjust=False, group_by="ticker", threads=True)
                break
            except:
                if attempt == 2: raise
                time.sleep(2)
        parts.append(pd.DataFrame({t: data[(t,"Close")] for t in chunk if (t,"Close") in data})
                     if isinstance(data.columns, pd.MultiIndex)
                     else pd.DataFrame({chunk[0]: data["Close"]}))
    if not parts: return pd.DataFrame()
    close = pd.concat(parts, axis=1)
    close.index = pd.to_datetime(close.index)
    return close.sort_index().dropna(axis=1, how="all")

def sma(df, n): return df.rolling(n, min_periods=n).mean()

def main():
    tickers = load_tickers()
    if not tickers: raise RuntimeError("ما وجدت تيكر في sp500_tickers.csv أو universe.csv")
    close = download_prices(tickers, START_DATE, TODAY)
    if close.empty: raise RuntimeError("فشل التنزيل.")
    s10=sma(close,10); s21=sma(close,21); s50=sma(close,50); s200=sma(close,200)
    cnd = (s10>s21) & (s21>s50) & (s50>s200)
    breadth = (cnd.sum(axis=1) / (~close.isna()).sum(axis=1).replace(0,pd.NA)) * 100.0
    breadth = breadth.sort_index(); breadth.name = "MAS_10_21_50_200"
    out = breadth.to_frame(); out.index.name = "Date"
    out.to_csv(os.path.join(DATA_DIR,"mas_10_21_50_200.csv"))
    print(f"✅ mas_10_21_50_200.csv — آخر قيمة: {breadth.dropna().iloc[-1]:.2f}")

if __name__ == "__main__": main()
