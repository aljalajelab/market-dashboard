from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import yfinance as yf

START_DATE=pd.Timestamp("2017-01-01"); MIN_UNIVERSE=50; FAST_EMA=12; SLOW_EMA=26
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

def download_close(tickers, start):
    fetch_start = start - pd.Timedelta(days=400)
    end = pd.Timestamp.today().normalize()
    df = yf.download(tickers=tickers, start=fetch_start,
                     end=end+pd.Timedelta(days=1),
                     group_by="column", auto_adjust=True,
                     progress=False, threads=True)
    if isinstance(df.columns, pd.MultiIndex):
        base = "Close" if "Close" in df.columns.levels[0] else "Adj Close"
        close = df[base].copy()
        close.columns = [str(c) for c in close.columns]
    else:
        base = "Close" if "Close" in df.columns else "Adj Close"
        close = df[[base]].rename(columns={base: tickers[0]})
    return close.dropna(how="all").sort_index()

def compute_macd_breadth(close_wide):
    ema_f = close_wide.ewm(span=FAST_EMA,adjust=False,min_periods=FAST_EMA).mean()
    ema_s = close_wide.ewm(span=SLOW_EMA,adjust=False,min_periods=SLOW_EMA).mean()
    macd = ema_f - ema_s
    cond = macd > 0
    valid = (~ema_f.isna()) & (~ema_s.isna())
    vc = valid.sum(axis=1)
    ok = (cond & valid).sum(axis=1)
    breadth = (ok / vc.replace(0,np.nan)) * 100.0
    breadth[vc < MIN_UNIVERSE] = np.nan
    return breadth[breadth.index >= START_DATE].dropna().rename("MACD_NQ_Breadth")

def main():
    tickers = load_tickers()
    print(f"✅ عدد التيكرات: {len(tickers)}")
    close = download_close(tickers, START_DATE)
    if close.empty: raise RuntimeError("فشل تحميل البيانات")
    series = compute_macd_breadth(close)
    if series.empty: raise RuntimeError("series فارغة بعد الحساب")
    out = series.to_frame(); out.index.name="Date"
    out.to_csv(DATA_DIR/"macd_nq.csv")
    print(f"✅ macd_nq.csv — آخر قيمة: {series.iloc[-1]:.2f}")

if __name__=="__main__": main()
