import pandas as pd
import glob
import os

RAW_DIR = 'data/raw'
STAGING_DIR = 'data/staging'
REJECTED_DIR = 'data/rejected'

os.makedirs(STAGING_DIR, exist_ok=True)
os.makedirs(REJECTED_DIR, exist_ok=True)

def load_raw():
    files = sorted(glob.glob(os.path.join(RAW_DIR, '*.csv')))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df

def clean(df):
    df = df.copy()
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    rejected_frames = []

    # Rule 1: cancelled orders (InvoiceNo starts with 'C')
    is_cancelled = df['InvoiceNo'].str.startswith('C')
    rejected = df[is_cancelled].copy()
    rejected['rejection_reason'] = 'Cancelled order'
    rejected_frames.append(rejected)
    df = df[~is_cancelled]

    # Rule 2: invalid (non-positive) quantity
    is_bad_qty = df['Quantity'] <= 0
    rejected = df[is_bad_qty].copy()
    rejected['rejection_reason'] = 'Non-positive quantity'
    rejected_frames.append(rejected)
    df = df[~is_bad_qty]

    # Rule 3: invalid (non-positive) price
    is_bad_price = df['UnitPrice'] <= 0
    rejected = df[is_bad_price].copy()
    rejected['rejection_reason'] = 'Non-positive unit price'
    rejected_frames.append(rejected)
    df = df[~is_bad_price]

    # Rule 4: drop exact duplicates
    dupe_mask = df.duplicated()
    rejected = df[dupe_mask].copy()
    rejected['rejection_reason'] = 'Duplicate row'
    rejected_frames.append(rejected)
    df = df[~dupe_mask]

    # Fill missing description
    df['Description'] = df['Description'].fillna('UNKNOWN')

    # Standardize date format
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Standardize country text
    df['Country'] = df['Country'].str.strip().str.title()

    rejected_df = pd.concat(rejected_frames, ignore_index=True) if rejected_frames else pd.DataFrame()
    return df, rejected_df

def main():
    print("Loading raw data...")
    df = load_raw()
    print(f"Loaded {len(df)} raw rows")

    cleaned_df, rejected_df = clean(df)

    staging_path = os.path.join(STAGING_DIR, 'cleaned_transactions.csv')
    cleaned_df.to_csv(staging_path, index=False)
    print(f"Wrote {len(cleaned_df)} cleaned rows to {staging_path}")

    rejected_path = os.path.join(REJECTED_DIR, 'rejected_records.csv')
    rejected_df.to_csv(rejected_path, index=False)
    print(f"Wrote {len(rejected_df)} rejected rows to {rejected_path}")

    print("\nRejection breakdown:")
    print(rejected_df['rejection_reason'].value_counts())

if __name__ == "__main__":
    main()
