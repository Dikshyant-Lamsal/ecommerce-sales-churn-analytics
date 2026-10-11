INSERT INTO fact_customer_retention (customer_key, analysis_date_key, recency_days, frequency, monetary, is_repeat_customer, is_churned)
SELECT
    customer_key,
    TO_CHAR(DATE '2011-12-10', 'YYYYMMDD')::INT AS analysis_date_key,
    recency_days,
    frequency,
    monetary,
    (frequency > 1) AS is_repeat_customer,
    (recency_days > 90) AS is_churned
FROM dim_customer
WHERE customer_id != -1
  AND recency_days IS NOT NULL;
