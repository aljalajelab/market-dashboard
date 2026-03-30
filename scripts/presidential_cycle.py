# presidential_cycle.py — يحفظ data/presidential_cycle_Y1..Y4.csv
import re
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
from pathlib import Path

MAX_TDOY=252; START_YEAR=1933
DATA_DIR=Path(__file__).resolve().parent.parent/"data"; DATA_DIR.mkdir(exist_ok=True)
BASE_DIR=Path(__file__).resolve().parent

def pick_csv(path):
    for name in ["SPY.csv","spy.csv"]:
        p=path/name
        if p.is_file(): return p
    p=path/"spy_breadth_series.csv"
    if p.is_file(): return p
    cands=sorted(path.glob("*SPY*.csv"),key=lambda x:x.stat().st_size,reverse=True)
    if cands: return cands[0]
    cands=sorted(path.glob("*.csv"),key=lambda x:x.stat().st_size,reverse=True)
    if cands: return cands[0]
    raise FileNotFoundError("CSV not found.")

df=pd.read_csv(pick_csv(BASE_DIR))
df.columns=[c.strip() for c in df.columns]
date_col=next((c for c in df.columns if "date" in c.lower()),df.columns[0])
price_col=next((c for c in df.columns if c in ["Price","Close","Adj Close","Last"]),df.columns[1])
df["Date"]=pd.to_datetime(df[date_col],errors="coerce")
df["Price"]=pd.to_numeric(df[price_col].astype(str).str.replace(",",""),errors="coerce")
df=df.dropna(subset=["Date","Price"]).sort_values("Date")
df["ret"]=df["Price"].pct_change().fillna(0)
df["year"]=df["Date"].dt.year
df["tdoy"]=df.groupby("year").cumcount()+1
df=df[df["tdoy"]<=MAX_TDOY]
df["cycle_year"]=((df["year"]-START_YEAR)%4)+1

all_cycles=[]
for cycle in range(1,5):
    subset=df[df["cycle_year"]==cycle]
    avg=subset.groupby("tdoy")["ret"].mean().reindex(range(1,MAX_TDOY+1),fill_value=0)
    cum=((1+avg).cumprod()-1)*100
    cycle_df=pd.DataFrame({"tdoy":range(1,MAX_TDOY+1),"avg_ret":avg.values,f"cum_pct_Y{cycle}":cum.values})
    cycle_df.to_csv(DATA_DIR/f"presidential_cycle_Y{cycle}.csv",index=False)
    all_cycles.append(cum.rename(f"Year_{cycle}"))

# ملف موحد أيضاً
combined=pd.DataFrame({"tdoy":range(1,MAX_TDOY+1)})
for s in all_cycles: combined[s.name]=s.values
combined.to_csv(DATA_DIR/"presidential_cycle_all.csv",index=False)
print(f"✅ presidential_cycle_all.csv + 4 ملفات منفصلة")
