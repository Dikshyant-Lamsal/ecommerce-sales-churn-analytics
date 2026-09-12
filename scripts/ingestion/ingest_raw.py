import os
import shutil
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

LANDING_DIR = 'data/landing/monthly'
RAW_DIR = 'data/raw'

def get_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)

def log_control(engine, source_file, row_count, status, error_message=None, rejected=0):
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO etl_control_log
                    (source_file, extraction_date, row_count, status, error_message, rejected_row_count)
                VALUES
                    (:source_file, :extraction_date, :row_count, :status, :error_message, :rejected)
            """),
            {
                "source_file": source_file,
                "extraction_date": datetime.now(),
                "row_count": row_count,
                "status": status,
                "error_message": error_message,
                "rejected": rejected
            }
        )

def ingest_file(engine, filename):
    src_path = os.path.join(LANDING_DIR, filename)
    raw_path = os.path.join(RAW_DIR, filename)

    try:
        df = pd.read_csv(src_path)
        row_count = len(df)

        if row_count == 0:
            log_control(engine, filename, 0, "FAILED", "Empty file")
            print(f"[FAILED] {filename}: empty file")
            return

        os.makedirs(RAW_DIR, exist_ok=True)
        shutil.copy2(src_path, raw_path)

        log_control(engine, filename, row_count, "SUCCESS")
        print(f"[SUCCESS] {filename}: {row_count} rows copied to raw layer")

    except Exception as e:
        log_control(engine, filename, 0, "FAILED", str(e))
        print(f"[FAILED] {filename}: {e}")

def main():
    engine = get_engine()
    files = sorted([f for f in os.listdir(LANDING_DIR) if f.endswith('.csv')])

    if not files:
        print("No files found in landing directory.")
        return

    for f in files:
        ingest_file(engine, f)

if __name__ == "__main__":
    main()
