INSERT INTO fact_sales (invoice_no, date_key, customer_key, product_key, geography_key, quantity, unit_price, line_revenue)
SELECT
    s.invoice_no,
    d.date_key,
    c.customer_key,
    p.product_key,
    g.geography_key,
    s.quantity,
    s.unit_price,
    s.line_revenue
FROM staging_orders s
JOIN dim_date d
    ON d.full_date = s.invoice_date::date
JOIN dim_customer c
    ON c.customer_id = COALESCE(s.customer_id, -1)
JOIN dim_product p
    ON p.stock_code = s.stock_code
JOIN dim_geography g
    ON g.country = s.country
ON CONFLICT (invoice_no, product_key, date_key, customer_key) DO NOTHING;
