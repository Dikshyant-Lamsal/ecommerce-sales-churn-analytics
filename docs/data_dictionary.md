# Data Dictionary — E-Commerce Sales Analytics & Customer Churn Project

## 1. Source Dataset (Raw)

**File**: `Online Retail.xlsx` (UCI Online Retail Dataset)

| Column | Type | Description | Notes |
|---|---|---|---|
| InvoiceNo | string | Unique invoice/transaction number | Prefix 'C' indicates a cancelled order |
| StockCode | string | Unique product/item code | |
| Description | string | Product name/description | ~1,454 rows have missing values |
| Quantity | integer | Number of units sold per line item | Negative values indicate returns/cancellations |
| InvoiceDate | datetime | Date and time the transaction occurred | Range: 2010-12-01 to 2011-12-09 |
| UnitPrice | decimal | Price per unit (GBP) | Some rows have 0 or negative values (invalid) |
| CustomerID | integer (nullable) | Unique customer identifier | ~135,080 rows missing (guest checkouts) |
| Country | string | Customer's country | 38 distinct countries |

---

## 2. Staging Table — `staging_orders`

Cleaned, standardized version of the raw data, loaded into PostgreSQL.

| Column | Type | Description | Validation Rule Applied |
|---|---|---|---|
| invoice_no | varchar(20) | Invoice number | Cancelled orders (prefix 'C') removed |
| stock_code | varchar(20) | Product code | — |
| description | text | Product description | Nulls filled with 'UNKNOWN' |
| quantity | integer | Units sold | Rows with quantity <= 0 removed |
| invoice_date | timestamp | Transaction timestamp | Standardized to ISO datetime |
| unit_price | numeric(10,2) | Price per unit | Rows with unit_price <= 0 removed |
| customer_id | integer (nullable) | Customer identifier | Nulls retained (guest orders) |
| country | varchar(100) | Country name | Standardized to title case, trimmed |
| line_revenue | numeric(12,2) | Computed: quantity × unit_price | Derived column |
| source_file | varchar(100) | Originating file name | For traceability/audit |

**Row-level rules applied during cleaning** (see `scripts/quality_checks/clean_to_staging.py`):
1. Remove cancelled orders (InvoiceNo starts with 'C')
2. Remove rows with quantity <= 0
3. Remove rows with unit_price <= 0
4. Remove exact duplicate rows
5. Fill missing Description with 'UNKNOWN'
6. Rejected rows logged to `data/rejected/rejected_records.csv` with a `rejection_reason` column

---

## 3. Dimension Tables

### `dim_customer`
| Column | Type | Description |
|---|---|---|
| customer_key | integer (PK, serial) | Surrogate key |
| customer_id | integer (unique) | Original CustomerID; `-1` = guest/unknown |
| country | varchar(100) | Customer's country |
| first_purchase_date | date | Date of customer's first order |
| last_purchase_date | date | Date of customer's most recent order |
| recency_days | integer | Days since last purchase (relative to analysis date 2011-12-10) |
| frequency | integer | Count of distinct invoices |
| monetary | numeric(12,2) | Total revenue from customer |
| customer_segment | varchar(20) | RFM-based segment: Champion / Loyal / Active / At Risk / Churned |

### `dim_product`
| Column | Type | Description |
|---|---|---|
| product_key | integer (PK, serial) | Surrogate key |
| stock_code | varchar(20) (unique) | Original product code |
| description | text | Product description (most common non-null value) |
| unit_price_avg | numeric(10,2) | Average selling price across all transactions |

### `dim_date`
| Column | Type | Description |
|---|---|---|
| date_key | integer (PK) | Format YYYYMMDD |
| full_date | date (unique) | Calendar date |
| day | smallint | Day of month |
| month | smallint | Month number (1-12) |
| month_name | varchar(10) | Month name |
| quarter | smallint | Quarter (1-4) |
| year | smallint | Year |
| day_of_week | varchar(10) | Day name (e.g., Monday) |
| is_weekend | boolean | True if Saturday/Sunday |

### `dim_geography`
| Column | Type | Description |
|---|---|---|
| geography_key | integer (PK, serial) | Surrogate key |
| country | varchar(100) (unique) | Country name |
| region | varchar(50) | Region grouping (currently unpopulated / optional enrichment) |

---

## 4. Fact Tables

### `fact_sales`
| Column | Type | Description |
|---|---|---|
| sales_key | bigint (PK, serial) | Surrogate key |
| invoice_no | varchar(20) | Invoice number |
| date_key | integer (FK → dim_date) | Transaction date |
| customer_key | integer (FK → dim_customer) | Customer |
| product_key | integer (FK → dim_product) | Product |
| geography_key | integer (FK → dim_geography) | Country |
| quantity | integer | Units sold |
| unit_price | numeric(10,2) | Price per unit |
| line_revenue | numeric(12,2) | quantity × unit_price |

**Grain**: One row per invoice line item (product sold within an invoice).
**Unique constraint**: (invoice_no, product_key, date_key, customer_key)

### `fact_customer_retention`
| Column | Type | Description |
|---|---|---|
| retention_key | bigint (PK, serial) | Surrogate key |
| customer_key | integer (FK → dim_customer) | Customer |
| analysis_date_key | integer (FK → dim_date) | Snapshot date (2011-12-10) |
| recency_days | integer | Days since last purchase at analysis date |
| frequency | integer | Number of distinct invoices |
| monetary | numeric(12,2) | Total lifetime revenue |
| is_repeat_customer | boolean | True if frequency > 1 |
| is_churned | boolean | True if recency_days > 90 (inactivity threshold) |

**Grain**: One row per customer, as of the analysis snapshot date.

---

## 5. Business Rules & Definitions

- **Cancelled order**: Any InvoiceNo beginning with 'C' — excluded from all sales/retention analysis.
- **Guest customer**: Any transaction with a missing CustomerID — mapped to surrogate customer_id = -1 in `dim_customer`, retained in `fact_sales` for accurate revenue totals but excluded from customer-level RFM/segmentation.
- **Churn definition (Part 1 baseline)**: A customer is considered churned if `recency_days > 90` relative to the analysis date (2011-12-10, the day after the last transaction in the dataset). This threshold will be revisited/refined in Part 2 based on further analysis of purchase-cycle length.
- **Customer segments (RFM-based)**:
  - **Champion**: recency ≤ 30 days AND frequency ≥ 10 orders
  - **Loyal**: recency ≤ 90 days AND frequency ≥ 5 orders
  - **Active**: recency ≤ 90 days
  - **At Risk**: recency ≤ 180 days
  - **Churned**: recency > 180 days

## 6. Data Quality Summary (as of last pipeline run)

| Metric | Value |
|---|---|
| Total raw rows | 541,909 |
| Rows rejected (cancelled orders) | 9,288 |
| Rows rejected (non-positive quantity) | 1,336 |
| Rows rejected (non-positive price) | 1,181 |
| Rows rejected (duplicates) | 5,226 |
| Total rows after cleaning (staging) | 524,878 |
| Rows loaded into fact_sales | 519,602 |
| Unique customers | 4,338 (+ 1 guest placeholder) |
| Unique products | 3,922 |
| Unique countries | 38 |
| Unique transaction dates | 305 |