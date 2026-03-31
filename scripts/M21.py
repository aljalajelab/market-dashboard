from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import yfinance as yf

START_YEAR=2003; W_SHORT=21; W_LONG=50; BREADTH_MA=10; MIN_UNIVERSE=50
BASE_DIR=Path(__file__).resolve().parent
DATA_DIR=BASE_DIR.parent/"data"; DATA_DIR.mkdir(exist_ok=True)

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

def download_close(tickers):
    start = pd.Timestamp(f"{START_YEAR}-01-01") - pd.Timedelta(days=450)
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    df = yf.download(tickers=sorted(set(tickers)), start=start, end=end,
                     group_by="column", auto_adjust=True, progress=False, threads=True)
    if df.empty: raise RuntimeError("فشل التنزيل")
    if isinstance(df.columns, pd.MultiIndex):
        base = "Close" if "Close" in df.columns.levels[0] else "Adj Close"
        close = df[base].copy()
        close.columns = [str(c).upper() for c in close.columns]
    else:
        base = "Close" if "Close" in df.columns else "Adj Close"
        close = df[[base]].rename(columns={base: tickers[0].upper()})
    return close.dropna(how="all").sort_index()

def compute_breadth(close_wide):
    ma_s = close_wide.rolling(W_SHORT, min_periods=W_SHORT).mean()
    ma_l = close_wide.rolling(W_LONG, min_periods=W_LONG).mean()
    cond  = ma_s > ma_l
    valid = (~ma_s.isna()) & (~ma_l.isna())
    valid_counts = valid.sum(axis=1)
    ok_counts    = (cond & valid).sum(axis=1)
    breadth = (ok_counts / valid_counts.replace(0,np.nan)) * 100.0
    breadth[valid_counts < MIN_UNIVERSE] = np.nan
    return breadth[breadth.index >= pd.Timestamp(f"{START_YEAR}-01-01")].dropna()

def main():
    tickers = load_tickers()
    print(f"✅ عدد التيكرات: {len(tickers)}")
    close   = download_close(tickers)
    series  = compute_breadth(close)
    if series.empty: raise RuntimeError("series فارغة")
    series_ma = series.rolling(BREADTH_MA, min_periods=BREADTH_MA).mean()
    out = pd.DataFrame({"M21_Breadth": series, "M21_MA10": series_ma})
    out.index.name = "Date"
    out.to_csv(DATA_DIR/"m21.csv")
    print(f"✅ m21.csv — آخر قيمة: {series.iloc[-1]:.2f}")

if __name__=="__main__": main()
