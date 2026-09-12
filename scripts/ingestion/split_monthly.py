import pandas as pd
import os

SRC = 'data/landing/Online Retail.xlsx'
OUT_DIR = 'data/landing/monthly'

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_excel(SRC, sheet_name='Online Retail')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['year_month'] = df['InvoiceDate'].dt.to_period('M').astype(str)

for ym, group in df.groupby('year_month'):
    out_path = os.path.join(OUT_DIR, f'orders_{ym}.csv')
    group.drop(columns=['year_month']).to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(group)} rows")

print(f"\nTotal months: {df['year_month'].nunique()}")
