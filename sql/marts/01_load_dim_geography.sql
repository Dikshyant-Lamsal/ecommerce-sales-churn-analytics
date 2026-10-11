INSERT INTO dim_geography (country, region)
SELECT DISTINCT country, NULL
FROM staging_orders
WHERE country IS NOT NULL
ON CONFLICT (country) DO NOTHING;
