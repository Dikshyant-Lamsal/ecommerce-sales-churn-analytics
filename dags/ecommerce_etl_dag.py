from datetime import datetime
import os
import subprocess
import sys

from airflow.sdk import dag, task


# Project paths inside the Airflow Docker container
PROJECT_DIR = "/opt/airflow/project"
PYTHON_BIN = sys.executable


def run_script(relative_path):
    """Run a Python ETL script and raise an error if it fails."""
    script_path = os.path.join(PROJECT_DIR, relative_path)

    result = subprocess.run(
        [PYTHON_BIN, script_path],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        raise RuntimeError(f"Script failed: {relative_path}")


def run_sql_file(relative_path):
    """Execute a SQL file against the PostgreSQL service."""
    sql_path = os.path.join(PROJECT_DIR, relative_path)

    if not os.path.isfile(sql_path):
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    required_variables = ["DB_HOST", "DB_PORT", "DB_USER", "DB_NAME"]
    missing_variables = [
        variable for variable in required_variables
        if not os.environ.get(variable)
    ]

    if missing_variables:
        raise RuntimeError(
            "Missing database environment variables: "
            + ", ".join(missing_variables)
        )

    env = os.environ.copy()

    # Use the password supplied to the Airflow container.
    if not env.get("PGPASSWORD"):
        env["PGPASSWORD"] = env.get("DB_PASSWORD", "")

    if not env["PGPASSWORD"]:
        raise RuntimeError("Database password is not configured.")

    command = [
        "psql",
        "-h", env["DB_HOST"],
        "-p", env["DB_PORT"],
        "-U", env["DB_USER"],
        "-d", env["DB_NAME"],
        "-v", "ON_ERROR_STOP=1",
        "-f", sql_path,
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        env=env,
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        raise RuntimeError(f"SQL execution failed: {relative_path}")


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

        for sql_file in sql_files:
            run_sql_file(sql_file)

    @task
    def load_facts():
        sql_files = [
            "sql/marts/05_load_fact_sales.sql",
            "sql/marts/06_update_customer_rfm.sql",
            "sql/marts/07_load_fact_customer_retention.sql",
        ]

        for sql_file in sql_files:
            run_sql_file(sql_file)

    # Define the task dependencies
    t1 = split_monthly()
    t2 = ingest_raw()
    t3 = clean_to_staging()
    t4 = load_staging_db()
    t5 = load_dimensions()
    t6 = load_facts()

    t1 >> t2 >> t3 >> t4 >> t5 >> t6


ecommerce_etl_pipeline()