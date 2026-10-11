from datetime import datetime
from airflow.sdk import dag, task
import subprocess
import os

PROJECT_DIR = "/home/diks/Projects/ecommerce-sales-churn-analytics-main"
PYTHON_BIN = os.path.join(PROJECT_DIR, "venv", "bin", "python")

def run_script(relative_path):
    script_path = os.path.join(PROJECT_DIR, relative_path)
    result = subprocess.run(
        [PYTHON_BIN, script_path],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Script failed: {relative_path}")

@dag(
    dag_id="ecommerce_sales_etl_pipeline",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ecommerce", "etl", "part1"],
)
def ecommerce_etl_pipeline():

    @task
    def split_monthly():
        run_script("scripts/ingestion/split_monthly.py")

    @task
    def ingest_raw():
        run_script("scripts/ingestion/ingest_raw.py")

    @task
    def clean_to_staging():
        run_script("scripts/quality_checks/clean_to_staging.py")

    @task
    def load_staging_db():
        run_script("scripts/transformation/load_staging.py")

    @task
    def load_dimensions():
        sql_files = [
            "sql/marts/01_load_dim_geography.sql",
            "sql/marts/02_load_dim_product.sql",
            "sql/marts/03_load_dim_customer.sql",
            "sql/marts/04_load_dim_date.sql",
        ]
        for f in sql_files:
            path = os.path.join(PROJECT_DIR, f)
            cmd = f"psql -U etl_user -d ecommerce_dw -h localhost -f {path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                     env={**os.environ, "PGPASSWORD": "etl_pass"})
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)
                raise RuntimeError(f"SQL failed: {f}")

    @task
    def load_facts():
        sql_files = [
            "sql/marts/05_load_fact_sales.sql",
            "sql/marts/06_update_customer_rfm.sql",
            "sql/marts/07_load_fact_customer_retention.sql",
        ]
        for f in sql_files:
            path = os.path.join(PROJECT_DIR, f)
            cmd = f"psql -U etl_user -d ecommerce_dw -h localhost -f {path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                     env={**os.environ, "PGPASSWORD": "etl_pass"})
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)
                raise RuntimeError(f"SQL failed: {f}")

    t1 = split_monthly()
    t2 = ingest_raw()
    t3 = clean_to_staging()
    t4 = load_staging_db()
    t5 = load_dimensions()
    t6 = load_facts()

    t1 >> t2 >> t3 >> t4 >> t5 >> t6

ecommerce_etl_pipeline()
