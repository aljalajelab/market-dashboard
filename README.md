# Market Dashboard — دليل الإعداد

## هيكل المجلدات

```
project/
├── .github/
│   └── workflows/
│       └── daily_update.yml    ← التشغيل التلقائي
├── scripts/                    ← سكريبتات Python
│   ├── spx_regime.py
│   ├── AdxDIPlusBreadth.py
│   ├── breadth_m50.py
│   ├── M21.py
│   ├── MacdBreadth_Yearly_2003.py
│   ├── MacdNQ.py
│   ├── MacdNY.py
│   ├── SpreadrsiNY.py
│   ├── spx_mas_breadth_chart.py
│   ├── spx_mas_breadth_chart_no10_v2.py
│   ├── rsi_spread_live.py
│   ├── breadth_rsi50_from_2003.py
│   └── breadth_rsi60_from_2003.py
├── data/                       ← ملفات CSV تُولَّد تلقائياً
│   ├── regime.csv
│   ├── adx.csv
│   ├── m50.csv
│   ├── m21.csv
│   ├── macd_spx.csv
│   ├── macd_nq.csv
│   ├── macd_ny.csv
│   ├── rsi_spread_ny.csv
│   ├── mas_10_21_50_200.csv
│   ├── mas_21_50_200.csv
│   ├── rsi_spread_spx.csv
│   ├── rsi50.csv
│   └── rsi60.csv
├── index.html                  ← الداشبورد
├── requirements.txt
└── README.md
```

## ملفات البيانات المطلوبة لكل سكريبت

ضع ملفات السنوات (2003.csv .. 2026.csv) داخل مجلد scripts/:
- spx_regime.py → يحتاج 2003.csv .. 2026.csv
- AdxDIPlusBreadth.py → نفس الملفات
- breadth_m50.py → نفس الملفات  
- MacdNQ.py → يحتاج ملف Nasdaq.csv
- MacdNY.py → يحتاج ملف nyse.csv
- SpreadrsiNY.py → يحتاج ملف nyse.csv
