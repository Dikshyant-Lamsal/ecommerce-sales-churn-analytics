-- Guest/unknown customer placeholder
INSERT INTO dim_customer (customer_id, country)
VALUES (-1, 'UNKNOWN')
ON CONFLICT (customer_id) DO NOTHING;

-- Real customers, with first/last purchase dates precomputed
INSERT INTO dim_customer (customer_id, country, first_purchase_date, last_purchase_date)
SELECT
    customer_id,
    MAX(country) AS country,
    MIN(invoice_date)::date AS first_purchase_date,
    MAX(invoice_date)::date AS last_purchase_date
FROM staging_orders
WHERE customer_id IS NOT NULL
GROUP BY customer_id
ON CONFLICT (customer_id) DO NOTHING;
