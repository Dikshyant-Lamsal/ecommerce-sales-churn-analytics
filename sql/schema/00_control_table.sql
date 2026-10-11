CREATE TABLE IF NOT EXISTS etl_control_log (
    control_id SERIAL PRIMARY KEY,
    source_file VARCHAR(255) NOT NULL,
    extraction_date TIMESTAMP NOT NULL DEFAULT NOW(),
    row_count INTEGER,
    status VARCHAR(20) NOT NULL,
    error_message TEXT,
    rejected_row_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS staging_orders (
    invoice_no    varchar(20),
    stock_code    varchar(20),
    description   text,
    quantity      integer,
    invoice_date  timestamp,
    unit_price    numeric(10,2),
    customer_id   integer,
    country       varchar(100),
    line_revenue  numeric(12,2),
    source_file   varchar(100)
);