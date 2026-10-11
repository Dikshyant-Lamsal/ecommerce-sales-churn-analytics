WITH rfm AS (
    SELECT
        c.customer_key,
        (DATE '2011-12-10' - MAX(d.full_date)) AS recency_days,
        COUNT(DISTINCT f.invoice_no) AS frequency,
        SUM(f.line_revenue) AS monetary
    FROM fact_sales f
    JOIN dim_customer c ON f.customer_key = c.customer_key
    JOIN dim_date d ON f.date_key = d.date_key
    WHERE c.customer_id != -1
    GROUP BY c.customer_key
)
UPDATE dim_customer dc
SET
    recency_days = rfm.recency_days,
    frequency = rfm.frequency,
    monetary = rfm.monetary,
    customer_segment = CASE
        WHEN rfm.recency_days <= 30 AND rfm.frequency >= 10 THEN 'Champion'
        WHEN rfm.recency_days <= 90 AND rfm.frequency >= 5 THEN 'Loyal'
        WHEN rfm.recency_days <= 90 THEN 'Active'
        WHEN rfm.recency_days <= 180 THEN 'At Risk'
        ELSE 'Churned'
    END
FROM rfm
WHERE dc.customer_key = rfm.customer_key;
