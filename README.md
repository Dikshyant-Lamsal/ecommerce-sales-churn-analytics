# E-Commerce Sales Analytics and Customer Churn Prediction — Part 1

## Project Overview

This project builds a retail sales data warehouse and interactive dashboard using the UCI Online Retail dataset. It implements a full ETL (Extract, Transform, Load) pipeline (landing → raw → staging → star schema) orchestrated with Apache Airflow, loaded into PostgreSQL, and visualized with a Streamlit dashboard.

## Dataset

- **Source**: UCI (University of California, Irvine) Machine Learning Repository — Online Retail Dataset
- **Access**: <https://archive.ics.uci.edu/dataset/352/online+retail> (or search "UCI Online Retail dataset")
- **Description**: Transaction-level online retail data (Dec 2010–Dec 2011) for a UK-based gift retailer, including invoice, product, quantity, price, customer, and country.
- **License**: Public, free for academic/research use.

## Architecture

```
Online Retail.xlsx (UCI dataset)
        │
        ▼
  [Landing Layer]  data/landing/monthly/*.csv  (split into 13 monthly extracts — simulates incremental source)
        │
        ▼
  [Raw Layer]      data/raw/*.csv  (untouched copies) + etl_control_log (Postgres audit table)
        │
        ▼
  [Staging Layer]  data/staging/cleaned_transactions.csv → staging_orders (Postgres table)
        │           (cancelled orders, invalid qty/price, duplicates removed & logged to data/rejected/)
        ▼
  [Star Schema]     dim_customer, dim_product, dim_date, dim_geography, fact_sales, fact_customer_retention
        │
        ▼
  [Analytics Layer] Streamlit dashboard (5 interactive views)
```

Orchestration: Apache Airflow DAG (Directed Acyclic Graph) named `ecommerce_sales_etl_pipeline` automates the full pipeline: split → ingest → clean → load staging → load dimensions → load facts.

## Tech Stack

- **Ingestion/Transformation**: Python, Pandas
- **Orchestration**: Apache Airflow 3.3.1
- **Database**: PostgreSQL
- **Dashboard**: Streamlit + Plotly
- **Environment**: Python 3.14, virtualenv

## Project Structure

```
ecommerce-sales-churn-analytics/
├── data/
│   ├── landing/monthly/     # simulated monthly source extracts
│   ├── raw/                 # raw copy of ingested files
│   ├── staging/             # cleaned data (pre-DB load)
│   └── rejected/            # rejected records log
├── dags/                    # Airflow DAG (Directed Acyclic Graph) definitions
├── scripts/
│   ├── ingestion/           # split_monthly.py, ingest_raw.py
│   ├── quality_checks/      # clean_to_staging.py, profile_raw.py
│   └── transformation/      # load_staging.py
├── sql/
│   ├── schema/              # DDL (Data Definition Language) for control log + star schema
│   └── marts/                # dimension/fact load scripts
├── dashboard/app.py          # Streamlit dashboard
├── airflow_home/             # Airflow metadata (not committed)
├── requirements.txt
├── .env                      # DB (database) credentials (not committed)
└── README.md
```

## Setup Instructions

### 1. Clone/copy the project and create a virtual environment

```
git clone <repo-url>
cd ecommerce-sales-churn-analytics
python3 -m venv venv
source venv/bin/activate        # bash/zsh
# source venv/bin/activate.fish  # fish shell
pip install -r requirements.txt
```

### 2. PostgreSQL setup

```
sudo systemctl enable --now postgresql
sudo -iu postgres psql
```

Inside `psql`:
```sql
CREATE USER etl_user WITH PASSWORD 'etl_pass';
CREATE DATABASE ecommerce_dw OWNER etl_user;
GRANT ALL PRIVILEGES ON DATABASE ecommerce_dw TO etl_user;
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ecommerce_dw
DB_USER=etl_user
DB_PASSWORD=etl_pass
```

### 4. Place the dataset

Download the UCI Online Retail dataset and place it at:

```
data/landing/Online Retail.xlsx
```

### 5. Apply the database schema

The schema **must** be applied before running any pipeline scripts — `sql/schema/00_control_table.sql` creates the `etl_control_log` audit table and the `staging_orders` staging table, and `sql/schema/01_star_schema.sql` creates the full star schema (`dim_customer`, `dim_product`, `dim_date`, `dim_geography`, `fact_sales`, `fact_customer_retention`).

```
psql -U etl_user -d ecommerce_dw -h localhost -f sql/schema/00_control_table.sql
psql -U etl_user -d ecommerce_dw -h localhost -f sql/schema/01_star_schema.sql
```

## Running the Pipeline

### Option A — Run scripts manually (step by step)

```
python scripts/ingestion/split_monthly.py
python scripts/ingestion/ingest_raw.py
python scripts/quality_checks/clean_to_staging.py
python scripts/transformation/load_staging.py
psql -U etl_user -d ecommerce_dw -h localhost -f sql/marts/01_load_dim_geography.sql
psql -U etl_user -d ecommerce_dw -h localhost -f sql/marts/02_load_dim_product.sql
psql -U etl_user -d ecommerce_dw -h localhost -f sql/marts/03_load_dim_customer.sql
psql -U etl_user -d ecommerce_dw -h localhost -f sql/marts/04_load_dim_date.sql
psql -U etl_user -d ecommerce_dw -h localhost -f sql/marts/05_load_fact_sales.sql
psql -U etl_user -d ecommerce_dw -h localhost -f sql/marts/06_update_customer_rfm.sql
psql -U etl_user -d ecommerce_dw -h localhost -f sql/marts/07_load_fact_customer_retention.sql
```

> **Note**: `dim_date` is populated only with dates that appear in the transaction data. The RFM (Recency, Frequency, Monetary) analysis snapshot date (`2011-12-10`) does not itself appear in any transaction, so it must be inserted manually before running `07_load_fact_customer_retention.sql`:
> ```sql
> INSERT INTO dim_date (date_key, full_date, day, month, month_name, quarter, year, day_of_week, is_weekend)
> VALUES (20111210, '2011-12-10', 10, 12, 'December', 4, 2011, 'Saturday', true)
> ON CONFLICT (date_key) DO NOTHING;
> ```

### Option B — Run via Airflow (recommended, automated)

```
python3 -m venv venv_airflow
source venv_airflow/bin/activate
pip install apache-airflow==3.3.1 pandas psycopg2-binary python-dotenv openpyxl

export AIRFLOW_HOME=$(pwd)/airflow_home
airflow db migrate
airflow standalone
```

Then open `http://localhost:8080`, log in with the generated admin credentials (printed to console, and saved at `airflow_home/simple_auth_manager_passwords.json.generated`), enable the `ecommerce_sales_etl_pipeline` DAG, and click **Trigger**.

> **Important**: `airflow_home/airflow.cfg`'s `dags_folder` setting must point at this project's `dags/` directory. If you've ever run Airflow from a different path for this project before, `airflow.cfg` may still reference the old location — check with `grep dags_folder airflow_home/airflow.cfg` and correct it if needed. Also always `export AIRFLOW_HOME=$(pwd)/airflow_home` fresh in each new shell session before running `airflow standalone` — it does not persist automatically.

## Running the Dashboard

```
source venv/bin/activate
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in a browser.

## Data Quality & Validation Rules

See `docs/data_dictionary.md` for full column-level rules. Summary of cleaning applied:

- Removed cancelled orders (InvoiceNo starting with 'C')
- Removed rows with non-positive quantity or unit price
- Removed exact duplicate rows
- Missing `CustomerID` retained and mapped to a `GUEST` (`-1`) surrogate key (guest checkouts)
- Missing `Description` filled with `'UNKNOWN'`
- All rejected rows logged to `data/rejected/rejected_records.csv` with a reason code

## Pipeline Execution Evidence

A full end-to-end Airflow DAG (Directed Acyclic Graph) run completed successfully (all 6 tasks: split_monthly → ingest_raw → clean_to_staging → load_staging_db → load_dimensions → load_facts), loading 519,602 fact rows and 4,338 customer records into PostgreSQL. See `docs/screenshots/` for execution evidence.

## Known Issues / Follow-ups

- `scripts/ingestion/ingest_raw.py`'s error handler can itself raise (if `etl_control_log` logging fails), which stops the script after the first file instead of continuing through all 13 monthly files. This only matters on a genuinely fresh environment with no pre-existing `data/raw/` files — worth hardening for Part 2.
- `sql/schema/01_star_schema.sql` was previously out of sync with `sql/marts/*.sql` (missing `dim_geography` and `fact_customer_retention`, incorrect column types on `dim_customer`/`dim_product`/`dim_date`/`fact_sales`, and no `staging_orders` DDL anywhere). This has been fixed as of this commit.

## Author

Individual project — Data Engineering and MLOps course — Dikshyant Lamsal, USN: 1CR23AI031