import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)

def main():
    engine = get_engine()
    df = pd.read_csv('data/staging/cleaned_transactions.csv')

    df = df.rename(columns={
        'InvoiceNo': 'invoice_no',
        'StockCode': 'stock_code',
        'Description': 'description',
        'Quantity': 'quantity',
        'InvoiceDate': 'invoice_date',
        'UnitPrice': 'unit_price',
        'CustomerID': 'customer_id',
        'Country': 'country'
    })

    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df['line_revenue'] = df['quantity'] * df['unit_price']
    df['source_file'] = 'cleaned_transactions.csv'

    # customer_id: convert to nullable int; keep NaN as-is for now (will be resolved during dim load)
    df['customer_id'] = df['customer_id'].astype('Int64')

    cols = ['invoice_no', 'stock_code', 'description', 'quantity', 'invoice_date',
            'unit_price', 'customer_id', 'country', 'line_revenue', 'source_file']
    df = df[cols]

    # Clear existing staging data (repeatable load)
    with engine.begin() as conn:
        conn.exec_driver_sql("TRUNCATE TABLE staging_orders;")

    df.to_sql('staging_orders', engine, if_exists='append', index=False, method='multi', chunksize=5000)
    print(f"Loaded {len(df)} rows into staging_orders")

if __name__ == "__main__":
    main()
