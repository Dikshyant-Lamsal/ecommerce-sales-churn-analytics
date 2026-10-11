INSERT INTO dim_date (date_key, full_date, day, month, month_name, quarter, year, day_of_week, is_weekend)
SELECT DISTINCT
    TO_CHAR(invoice_date::date, 'YYYYMMDD')::INT AS date_key,
    invoice_date::date AS full_date,
    EXTRACT(DAY FROM invoice_date)::SMALLINT AS day,
    EXTRACT(MONTH FROM invoice_date)::SMALLINT AS month,
    TO_CHAR(invoice_date, 'Month') AS month_name,
    EXTRACT(QUARTER FROM invoice_date)::SMALLINT AS quarter,
    EXTRACT(YEAR FROM invoice_date)::SMALLINT AS year,
    TO_CHAR(invoice_date, 'Day') AS day_of_week,
    (EXTRACT(ISODOW FROM invoice_date) IN (6, 7)) AS is_weekend
FROM staging_orders
WHERE invoice_date IS NOT NULL
ON CONFLICT (date_key) DO NOTHING;
