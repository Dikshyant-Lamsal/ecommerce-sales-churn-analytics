import pandas as pd
import glob

files = sorted(glob.glob('data/raw/*.csv'))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

print("Total rows:", len(df))
print("Cancelled invoices (starts with C):", df['InvoiceNo'].astype(str).str.startswith('C').sum())
print("Negative quantity:", (df['Quantity'] < 0).sum())
print("Zero/negative unit price:", (df['UnitPrice'] <= 0).sum())
print("Missing CustomerID:", df['CustomerID'].isnull().sum())
print("Missing Description:", df['Description'].isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
