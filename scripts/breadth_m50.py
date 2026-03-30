# breadth_m50.py — يحفظ data/m50.csv
from pathlib import Path
import re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import yfinance as yf

START_YEAR = 2003
W_SHORT = 50; W_LONG = 200; BREADTH_MA = 10
MIN_UNIVERSE_FRAC = 0.60

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def read_yearly_universes(base_dir):
    universes = {}
    for f in sorted(p for p in base_dir.glob("*.csv") if re.match(r"^(19|20)\d{2}\.csv$", p.name, re.I)):
        year = int(f.stem)
        s = pd.read_csv(f, header=None, names=["ticker"])["ticker"]
        tickers = s.astype(str).str.strip().str.upper().replace("",np.nan).dropna().drop_duplicates().tolist()
        if tickers: universes[year] = tickers
    if not universes: raise SystemExit("ما لقيت ملفات سنوات.")
    return universes

def download_close(all_tickers, start_year):
    start = pd.Timestamp(f"{start_year}-01-01") - pd.Timedelta(days=450)
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    df = yf.download(tickers=sorted(set(all_tickers)), start=start, end=end, group_by="column", auto_adjust=True, progress=False, threads=True)
    if df.empty: raise SystemExit("فشل تحميل الأسعار.")
    if isinstance(df.columns, pd.MultiIndex):
        base = "Close" if "Close" in df.columns.levels[0] else "Adj Close"
        close = df[base].copy(); close.columns = [str(c).upper() for c in close.columns]
    else:
        base = "Close" if "Close" in df.columns else "Adj Close"
        close = df[[base]].rename(columns={base: all_tickers[0].upper()})
    return close.dropna(how="all").sort_index()

def compute_breadth(close_wide, universes, min_frac):
    ma50 = close_wide.rolling(W_SHORT, min_periods=W_SHORT).mean()
    ma200 = close_wide.rolling(W_LONG, min_periods=W_LONG).mean()
    cond = ma50 > ma200; valid = (~ma50.isna()) & (~ma200.isna())
    idx = cond.index[cond.index >= pd.Timestamp(f"{min(universes)}-01-01")]
    out = pd.Series(index=idx, dtype="float64")
    for year in sorted(universes):
        mask = (idx >= pd.Timestamp(f"{year}-01-01")) & (idx <= pd.Timestamp(f"{year+1}-01-01") - pd.Timedelta(days=1))
        if not mask.any(): continue
        cols = [c for c in universes[year] if c in cond.columns]
        if not cols: continue
        cond_y = cond.loc[idx[mask], cols]; valid_y = valid.loc[idx[mask], cols]
        valid_counts = valid_y.sum(axis=1); ok_counts = (cond_y & valid_y).sum(axis=1)
        b = (ok_counts / valid_counts.replace(0, np.nan)) * 100.0
        b[valid_counts < len(cols)*min_frac] = np.nan
        out.loc[b.index] = b
    return out[out.index >= pd.Timestamp(f"{max(START_YEAR, min(universes))}-01-01")].dropna()

def main():
    universes = read_yearly_universes(BASE_DIR)
    all_tickers = [t for tl in universes.values() for t in tl]
    close = download_close(all_tickers, min(universes))
    series = compute_breadth(close, universes, MIN_UNIVERSE_FRAC)
    series_ma = series.rolling(BREADTH_MA, min_periods=BREADTH_MA).mean()
    out = pd.DataFrame({"M50_Breadth": series, "M50_MA10": series_ma})
    out.index.name = "Date"
    out.to_csv(DATA_DIR / "m50.csv")
    print(f"✅ m50.csv — آخر قيمة: {series.iloc[-1]:.2f}")

if __name__ == "__main__":
    main()
