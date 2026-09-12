INSERT INTO dim_product (stock_code, description, unit_price_avg)
SELECT
    stock_code,
    MAX(description) AS description,
    ROUND(AVG(unit_price), 2) AS unit_price_avg
FROM staging_orders
WHERE stock_code IS NOT NULL
GROUP BY stock_code
ON CONFLICT (stock_code) DO NOTHING;
