# spx500.py — يحفظ data/seasonality_22y.csv
import os, glob, re
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
from pathlib import Path

MAX_TDOY=252
DATA_DIR=Path(__file__).resolve().parent.parent/"data"; DATA_DIR.mkdir(exist_ok=True)
BASE_DIR=Path(__file__).resolve().parent

def pick_csv(path):
    for name in ["SPY.csv","spy.csv","SPY.csv.csv"]:
        p=path/name
        if p.is_file(): return p
    cands=sorted(path.glob("*SPY*.csv"),key=lambda x:x.stat().st_size,reverse=True)
    if cands: return cands[0]
    # fallback: spy_breadth_series.csv
    p=path/"spy_breadth_series.csv"
    if p.is_file(): return p
    cands=sorted(path.glob("*.csv"),key=lambda x:x.stat().st_size,reverse=True)
    if cands: return cands[0]
    raise FileNotFoundError("CSV not found.")

def read_spy_csv(path):
    df=None
    for enc in ("utf-8","utf-8-sig","latin-1"):
        try: df=pd.read_csv(path,encoding=enc); break
        except: pass
    cols={c.strip():c for c in df.columns}
    date_col=cols.get("Date",df.columns[0])
    close_col=None
    for k in ["Price","Close","Adj Close","Adj close","Last","Close*"]:
        if k in cols: close_col=cols[k]; break
    if close_col is None: close_col=df.columns[1]
    out=df[[date_col,close_col]].rename(columns={date_col:"date",close_col:"close"})
    def clean_date(s):
        s=str(s)
        m=re.search(r"(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})",s)
        if m:
            m1,d1,y1=m.groups(); y=int(y1); y=2000+y if y<100 else y
            return f"{y:04d}-{int(m1):02d}-{int(d1):02d}"
        return s
    out["date"]=pd.to_datetime(out["date"].map(clean_date),errors="coerce")
    out["close"]=pd.to_numeric(out["close"].astype(str).str.replace(",",""),errors="coerce")
    return out.dropna(subset=["date","close"]).sort_values("date").set_index("date")

csv_path=pick_csv(BASE_DIR)
df=read_spy_csv(csv_path)
end_ts=df.index.max(); start_ts=end_ts-pd.DateOffset(years=22)
df=df.loc[start_ts:end_ts].copy()
df["ret"]=df["close"].pct_change().fillna(0.0)
df["year"]=df.index.year
df["tdoy"]=df.groupby("year").cumcount()+1
df=df[df["tdoy"]<=MAX_TDOY]
daily_avg=df.groupby("tdoy")["ret"].mean().reindex(range(1,MAX_TDOY+1),fill_value=0.0)
cum_pct=((1.0+daily_avg).cumprod()-1.0)*100.0
df_month=df[["close"]].copy(); df_month["ym"]=df_month.index.to_period("M")
monthly=df_month.groupby("ym")["close"].agg(["first","last"])
monthly["mret"]=monthly["last"]/monthly["first"]-1.0
avg_month=monthly.groupby(monthly.index.month)["mret"].mean()

out=pd.DataFrame({"tdoy":range(1,MAX_TDOY+1),"avg_ret":daily_avg.values,"cum_return_pct":cum_pct.values})
out.to_csv(DATA_DIR/"seasonality_22y.csv",index=False)
monthly_out=pd.DataFrame({"month":range(1,13),"avg_return_pct":[avg_month.get(m,np.nan)*100 for m in range(1,13)]})
monthly_out.to_csv(DATA_DIR/"seasonality_22y_monthly.csv",index=False)
print(f"✅ seasonality_22y.csv")
