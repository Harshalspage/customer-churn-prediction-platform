SELECT
    COUNT(*) AS total_customers,
    SUM(churn_value) AS churned_customers,
    ROUND(AVG(churn_value::numeric) * 100, 2) AS churn_rate_pct
FROM public.telco_churn_raw;

SELECT
    contract,
    COUNT(*) AS customers,
    SUM(churn_value) AS churned,
    ROUND(AVG(churn_value::numeric) * 100, 2) AS churn_rate_pct
FROM public.telco_churn_raw
GROUP BY contract
ORDER BY churn_rate_pct DESC;

SELECT
    CASE
        WHEN tenure_months <= 12 THEN '0-12 months'
        WHEN tenure_months <= 24 THEN '13-24 months'
        WHEN tenure_months <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END AS tenure_group,
    COUNT(*) AS customers,
    ROUND(AVG(churn_value::numeric) * 100, 2) AS churn_rate_pct
FROM public.telco_churn_raw
GROUP BY 1
ORDER BY
    CASE tenure_group
        WHEN '0-12 months' THEN 1
        WHEN '13-24 months' THEN 2
        WHEN '25-48 months' THEN 3
        ELSE 4
    END;

SELECT
    churn_label,
    COUNT(*) AS customers,
    ROUND(AVG(monthly_charges::numeric), 2) AS avg_monthly_charges,
    ROUND(AVG(tenure_months::numeric), 1) AS avg_tenure_months
FROM public.telco_churn_raw
GROUP BY churn_label;

