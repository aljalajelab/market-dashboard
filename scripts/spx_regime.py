# spx_regime.py — يحفظ data/regime.csv تلقائياً
import os, time, datetime as dt
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")  # بدون شاشة على السيرفر
import matplotlib.pyplot as plt

BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "..", "data")
START_DATE = dt.date(2003, 1, 1)
TODAY      = dt.date.today()
BATCH_SIZE = 60
PAUSE_SEC  = 0.8
RSI_PERIOD = 14
RISK_OFF   = 28
WEAK_ON    = 40
STRONG_ON  = 75

def _read_year_tickers(year):
    for p in [os.path.join(BASE_DIR, f"{year}.csv"), os.path.join(BASE_DIR, f"{year}.CSV")]:
        if os.path.isfile(p):
            df = pd.read_csv(p)
            col = next((c for c in df.columns if str(c).strip().lower() in ("ticker","tickers","symbol","symbols")), df.columns[0])
            s = df[col].astype(str).str.strip().str.upper().replace({"NAN": None}).dropna().drop_duplicates()
            s = s[s.str.match(r"^[A-Z0-9.\-^]+$")].str.replace(".", "-", regex=False)
            return sorted(s.unique().tolist())
    return []

def load_universe_by_year(start_y=2003, end_y=TODAY.year):
    year_to_list, union = {}, set()
    for y in range(start_y, end_y + 1):
        lst = _read_year_tickers(y)
        if lst:
            year_to_list[y] = lst
            union.update(lst)
    return year_to_list, sorted(union)

def chunks(lst, n):
    for i in range(0, len(lst), n): yield lst[i:i+n]

def download_close(tickers, start, end):
    frames = []
    for batch in chunks(tickers, BATCH_SIZE):
        for attempt in (1, 2):
            try:
                data = yf.download(batch, start=start.isoformat(), end=(end+dt.timedelta(days=1)).isoformat(),
                                   progress=False, auto_adjust=False, group_by="ticker", threads=True)
                break
            except Exception:
                if attempt == 2: raise
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
    ag = ag.combine_first(gain.expanding().mean())
    al = al.combine_first(loss.expanding().mean())
    return 100 - (100 / (1 + ag/al))

def pct_from_signals(df_bool):
    if df_bool is None or df_bool.empty: return pd.Series(dtype=float)
    return (df_bool.sum(axis=1, skipna=True) / df_bool.notna().sum(axis=1)) * 100.0

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    year_to_list, union = load_universe_by_year(START_DATE.year, TODAY.year)
    if not union: raise RuntimeError("ما لقيت أي تيكر.")
    close_all = download_close(union, START_DATE, TODAY)
    if close_all.empty: raise RuntimeError("فشل التنزيل.")

    sma200 = close_all.rolling(200, min_periods=200).mean()
    above200 = (close_all > sma200)
    rsi50_flags = pd.DataFrame(index=close_all.index)
    rsi60_flags = pd.DataFrame(index=close_all.index)
    for t in close_all.columns:
        c = close_all[t].dropna()
        if c.empty: continue
        rsi = compute_rsi(c, RSI_PERIOD).reindex(close_all.index)
        rsi50_flags[t] = (rsi >= 50)
        rsi60_flags[t] = (rsi >= 60)

    lt_list, mt_list, st_list = [], [], []
    for y, tickers in year_to_list.items():
        idx = (close_all.index.year == y)
        cols = [t for t in tickers if t in close_all.columns]
        if not any(idx) or not cols: continue
        lt_list.append(pct_from_signals(above200.loc[idx, cols]))
        mt_list.append(pct_from_signals(rsi50_flags.loc[idx, cols]))
        st_list.append(pct_from_signals(rsi60_flags.loc[idx, cols]))

    LT = pd.concat(lt_list).sort_index()
    MT = pd.concat(mt_list).sort_index()
    ST = pd.concat(st_list).sort_index()
    score = 0.5*LT + 0.3*MT + 0.2*ST
    score_ema10 = score.ewm(span=10, min_periods=5).mean()

    out = pd.DataFrame({
        "LT_pct_above_SMA200": LT,
        "MT_pct_RSI_ge_50": MT,
        "ST_pct_RSI_ge_60": ST,
        "RegimeScore": score,
        "RegimeScore_EMA10": score_ema10,
    })
    out.index.name = "Date"
    out.to_csv(os.path.join(DATA_DIR, "regime.csv"))
    print(f"✅ regime.csv — آخر قيمة: {score.iloc[-1]:.2f}")

if __name__ == "__main__":
    main()
