import pandas as pd

df = pd.read_excel('data/landing/Online Retail.xlsx', sheet_name='Online Retail')

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nDtypes:\n", df.dtypes)
print("\nNull counts:\n", df.isnull().sum())
print("\nDate range:", df['InvoiceDate'].min(), "to", df['InvoiceDate'].max())
print("\nSample rows:\n", df.head(3))
