def load_tickers():
    p = os.path.join(BASE_DIR, "sp500_tickers.csv")
    if not os.path.isfile(p):
        p = os.path.join(BASE_DIR, "universe.csv")
    df = pd.read_csv(p)
    col = next((c for c in df.columns if str(c).strip().lower() in 
               ("ticker","tickers","symbol","symbols")), df.columns[0])
    tickers = df[col].astype(str).str.strip().str.upper().replace({"NAN":None}).dropna().drop_duplicates()
    return sorted(tickers[tickers.str.match(r"^[A-Z0-9.\-^]+$")].str.replace(".","-",regex=False).unique().tolist())

def main():
    tickers = load_tickers()
    if not tickers:
        raise FileNotFoundError("sp500_tickers.csv فارغ أو مو موجود")
    
    close = download_prices(tickers, START_DATE, TODAY)
    if close.empty:
        raise ValueError("لم يتم تحميل أي بيانات")

    s21 = sma(close,21); s50 = sma(close,50); s200 = sma(close,200)
    
    cnd = (s21 > s50) & (s50 > s200)
    breadth = (cnd.sum(axis=1) / (~close.isna()).sum(axis=1).replace(0,pd.NA)) * 100.0
    breadth = breadth.sort_index(); breadth.name = "MAS_21_50_200"
    
    out = breadth.to_frame(); out.index.name = "Date"
    out.to_csv(os.path.join(DATA_DIR,"mas_21_50_200.csv"))
    print(f"✅ mas_21_50_200.csv — آخر قيمة: {breadth.dropna().iloc[-1]:.2f}")

if __name__=="__main__": main()
