# spx_mas_breadth_chart.py — يحفظ data/mas_10_21_50_200.csv
import os, time, datetime as dt
import pandas as pd
import matplotlib; matplotlib.use("Agg")

START_DATE=dt.date(2003,1,1); TODAY=dt.date.today()
BASE_DIR=os.path.abspath(os.path.dirname(__file__))
DATA_DIR=os.path.join(BASE_DIR,"..","data"); os.makedirs(DATA_DIR,exist_ok=True)

def load_year_list(year):
    for p in [os.path.join(BASE_DIR,f"{year}.csv"),os.path.join(BASE_DIR,f"{year}.CSV")]:
        if os.path.isfile(p):
            df=pd.read_csv(p)
            col=next((c for c in df.columns if str(c).strip().lower() in ("ticker","tickers","symbol","symbols")),df.columns[0])
            tickers=df[col].astype(str).str.strip().str.upper().replace({"NAN":None}).dropna().drop_duplicates()
            tickers=tickers[tickers.str.match(r"^[A-Z0-9.\-^]+$")].str.replace(".","-",regex=False).unique().tolist()
            return sorted(tickers)
    return []

def download_prices(tickers,start,end):
    import yfinance as yf
    parts=[]; batch=60
    for i in range(0,len(tickers),batch):
        chunk=tickers[i:i+batch]
        for attempt in (1,2):
            try:
                data=yf.download(chunk,start=start.isoformat(),end=(end+dt.timedelta(days=1)).isoformat(),progress=False,auto_adjust=False,group_by="ticker",threads=True)
                break
            except:
                if attempt==2: raise
                time.sleep(2)
        if isinstance(data.columns,pd.MultiIndex):
            close=pd.DataFrame({t:data[(t,"Close")] for t in chunk if (t,"Close") in data})
        else:
            close=pd.DataFrame({chunk[0]:data["Close"]})
        parts.append(close)
    if not parts: return pd.DataFrame()
    close=pd.concat(parts,axis=1); close.index=pd.to_datetime(close.index)
    return close.sort_index().dropna(axis=1,how="all")

def sma(df,n): return df.rolling(n,min_periods=n).mean()

def main():
    dates=pd.bdate_range(start=START_DATE,end=TODAY); needed_years=sorted(set(dates.year))
    year_to_tickers,union={},set()
    for y in needed_years:
        lst=load_year_list(y)
        if lst: year_to_tickers[y]=lst; union.update(lst)
    if not union: raise RuntimeError("ما وجدت تيكر.")
    close=download_prices(sorted(union),START_DATE,TODAY)
    if close.empty: raise RuntimeError("فشل التنزيل.")
    s10=sma(close,10); s21=sma(close,21); s50=sma(close,50); s200=sma(close,200)
    parts=[]
    for y in needed_years:
        if y not in year_to_tickers: continue
        idx=close.index[close.index.year==y]
        if not len(idx): continue
        cols=[t for t in year_to_tickers[y] if t in close.columns]
        if not cols: continue
        cnd=(s10.loc[idx,cols]>s21.loc[idx,cols])&(s21.loc[idx,cols]>s50.loc[idx,cols])&(s50.loc[idx,cols]>s200.loc[idx,cols])
        parts.append((cnd.sum(axis=1)/(~close.loc[idx,cols].isna()).sum(axis=1).replace(0,pd.NA))*100.0)
    breadth=pd.concat(parts).sort_index(); breadth.name="MAS_10_21_50_200"
    out=breadth.to_frame(); out.index.name="Date"
    out.to_csv(os.path.join(DATA_DIR,"mas_10_21_50_200.csv"))
    print(f"✅ mas_10_21_50_200.csv — آخر قيمة: {breadth.dropna().iloc[-1]:.2f}")

if __name__=="__main__": main()
