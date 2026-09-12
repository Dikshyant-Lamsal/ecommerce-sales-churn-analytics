-- Dimension: Customer
CREATE TABLE IF NOT EXISTS dim_customer (
    customer_key SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    country VARCHAR(100),
    first_purchase_date DATE,
    last_purchase_date DATE,
    recency_days INTEGER,
    frequency INTEGER,
    monetary NUMERIC(12,2),
    customer_segment VARCHAR(20),
    UNIQUE (customer_id)
);

-- Dimension: Product
CREATE TABLE IF NOT EXISTS dim_product (
    product_key SERIAL PRIMARY KEY,
    stock_code VARCHAR(50) NOT NULL,
    description TEXT,
    unit_price_avg NUMERIC(10,2),
    UNIQUE (stock_code)
);

-- Dimension: Date
CREATE TABLE IF NOT EXISTS dim_date (
    date_key INTEGER PRIMARY KEY,
    full_date DATE NOT NULL,
    day SMALLINT NOT NULL,
    month SMALLINT NOT NULL,
    month_name VARCHAR(15) NOT NULL,
    quarter SMALLINT NOT NULL,
    year SMALLINT NOT NULL,
    day_of_week VARCHAR(10) NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    UNIQUE (full_date)
);

-- Dimension: Geography
CREATE TABLE IF NOT EXISTS dim_geography (
    geography_key SERIAL PRIMARY KEY,
    country VARCHAR(100) NOT NULL,
    region VARCHAR(50),
    UNIQUE (country)
);

-- Fact: Sales
CREATE TABLE IF NOT EXISTS fact_sales (
    sales_key BIGSERIAL PRIMARY KEY,
    invoice_no VARCHAR(20) NOT NULL,
    date_key INTEGER NOT NULL REFERENCES dim_date(date_key),
    customer_key INTEGER NOT NULL REFERENCES dim_customer(customer_key),
    product_key INTEGER NOT NULL REFERENCES dim_product(product_key),
    geography_key INTEGER NOT NULL REFERENCES dim_geography(geography_key),
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(10,2) NOT NULL,
    line_revenue NUMERIC(12,2) NOT NULL,
    UNIQUE (invoice_no, product_key, date_key, customer_key)
);

-- Fact: Customer Retention
CREATE TABLE IF NOT EXISTS fact_customer_retention (
    retention_key BIGSERIAL PRIMARY KEY,
    customer_key INTEGER NOT NULL REFERENCES dim_customer(customer_key),
    analysis_date_key INTEGER NOT NULL REFERENCES dim_date(date_key),
    recency_days INTEGER,
    frequency INTEGER,
    monetary NUMERIC(12,2),
    is_repeat_customer BOOLEAN,
    is_churned BOOLEAN
);
