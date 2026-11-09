-- ============================================================
-- Sales Demand Forecasting & Anomaly Detection
-- Phase 7: SQL Analysis Queries
-- Database: data/processed/sales.db  (created by 07_sql_analysis.ipynb)
--
-- These queries can be run directly against the SQLite database, e.g.:
--   sqlite3 ../data/processed/sales.db
--   .read analysis.sql
--
-- Tables:  sales (1,017,209 store-day rows)
--          stores (1,115 store metadata rows)
--          anomalies (942 network-level daily anomaly rows)
--          view sales_with_store = sales JOIN stores
-- ============================================================

-- ------------------------------------------------------------
-- 1. Sales by store type (revenue share)
-- ------------------------------------------------------------
SELECT
    StoreType,
    COUNT(DISTINCT Store) as NumStores,
    SUM(Sales) as TotalSales,
    ROUND(SUM(Sales) * 100.0 / (SELECT SUM(Sales) FROM sales_with_store WHERE Open = 1), 2) as SalesPct,
    ROUND(AVG(Sales), 0) as AvgDailySales,
    ROUND(AVG(Customers), 0) as AvgDailyCustomers
FROM sales_with_store
WHERE Open = 1
GROUP BY StoreType
ORDER BY TotalSales DESC;

-- ------------------------------------------------------------
-- 2. Per-store efficiency by type (sales per customer)
-- ------------------------------------------------------------
SELECT
    StoreType,
    ROUND(AVG(Sales), 0) as AvgSalesPerStoreDay,
    ROUND(AVG(Customers), 0) as AvgCustomersPerStoreDay,
    ROUND(AVG(Sales) / AVG(Customers), 2) as AvgSalesPerCustomer,
    COUNT(DISTINCT Store) as NumStores
FROM sales_with_store
WHERE Open = 1
GROUP BY StoreType
ORDER BY AvgSalesPerStoreDay DESC;

-- ------------------------------------------------------------
-- 3. Promotional lift by store type (with vs without promo)
-- ------------------------------------------------------------
SELECT
    StoreType,
    ROUND(AVG(CASE WHEN Promo = 1 THEN Sales END) / AVG(CASE WHEN Promo = 0 THEN Sales END) * 100 - 100, 2) as PromoLiftPct
FROM sales_with_store
WHERE Open = 1
GROUP BY StoreType
ORDER BY PromoLiftPct DESC;

-- ------------------------------------------------------------
-- 4. Top 10 stores by lifetime sales
-- ------------------------------------------------------------
SELECT
    s.Store,
    st.StoreType,
    st.Assortment,
    COUNT(*) as DaysOpen,
    SUM(s.Sales) as TotalSales,
    ROUND(AVG(s.Sales), 0) as AvgDailySales,
    ROUND(AVG(s.Customers), 0) as AvgDailyCustomers,
    st.CompetitionDistance
FROM sales s
JOIN stores st ON s.Store = st.Store
WHERE s.Open = 1
GROUP BY s.Store
ORDER BY TotalSales DESC
LIMIT 10;

-- ------------------------------------------------------------
-- 5. Bottom 10 stores by lifetime sales
-- ------------------------------------------------------------
SELECT
    s.Store,
    st.StoreType,
    st.Assortment,
    COUNT(*) as DaysOpen,
    SUM(s.Sales) as TotalSales,
    ROUND(AVG(s.Sales), 0) as AvgDailySales,
    st.CompetitionDistance
FROM sales s
JOIN stores st ON s.Store = st.Store
WHERE s.Open = 1
GROUP BY s.Store
ORDER BY TotalSales ASC
LIMIT 10;

-- ------------------------------------------------------------
-- 6. State holiday effect on sales (when open)
-- ------------------------------------------------------------
SELECT
    StateHoliday,
    COUNT(*) as Days,
    SUM(CASE WHEN Open = 1 THEN 1 ELSE 0 END) as DaysOpen,
    ROUND(AVG(CASE WHEN Open = 1 THEN Sales END), 0) as AvgSalesWhenOpen,
    ROUND(AVG(Sales), 0) as AvgSalesAllDays
FROM sales_with_store
GROUP BY StateHoliday
ORDER BY AvgSalesWhenOpen DESC;

-- ------------------------------------------------------------
-- 7. Competition distance bands vs performance
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN st.CompetitionDistance < 1000 THEN 'Very Close (<1km)'
        WHEN st.CompetitionDistance < 5000 THEN 'Close (1-5km)'
        WHEN st.CompetitionDistance < 10000 THEN 'Medium (5-10km)'
        ELSE 'Far (>10km)'
    END as DistanceGroup,
    COUNT(DISTINCT s.Store) as NumStores,
    ROUND(AVG(s.Sales), 0) as AvgDailySales,
    ROUND(AVG(s.Customers), 0) as AvgDailyCustomers
FROM sales s
JOIN stores st ON s.Store = st.Store
WHERE s.Open = 1 AND st.CompetitionDistance > 0
GROUP BY DistanceGroup
ORDER BY AvgDailySales DESC;

-- ------------------------------------------------------------
-- 8. Monthly sales trend (2013-01 .. 2015-07)
-- ------------------------------------------------------------
SELECT
    substr(Date, 1, 7) as Month,
    SUM(Sales) as TotalSales,
    ROUND(AVG(Sales), 0) as AvgDailySales,
    SUM(Customers) as TotalCustomers,
    COUNT(DISTINCT Store) as ActiveStores
FROM sales_with_store
WHERE Open = 1
GROUP BY Month
ORDER BY Month;

-- ------------------------------------------------------------
-- 9. Year-over-year network sales
-- ------------------------------------------------------------
SELECT
    substr(Date, 1, 4) as Year,
    SUM(Sales) as TotalSales,
    ROUND(AVG(Sales), 0) as AvgDailySales,
    SUM(Customers) as TotalCustomers
FROM sales_with_store
WHERE Open = 1
GROUP BY Year
ORDER BY Year;

-- ------------------------------------------------------------
-- 10. Store performance deciles (NTILE window function)
-- ------------------------------------------------------------
WITH store_stats AS (
    SELECT
        s.Store,
        st.StoreType,
        st.Assortment,
        SUM(s.Sales) as TotalSales,
        ROUND(AVG(s.Sales), 0) as AvgDailySales,
        COUNT(*) as DaysOpen
    FROM sales s
    JOIN stores st ON s.Store = st.Store
    WHERE s.Open = 1
    GROUP BY s.Store
),
ranked AS (
    SELECT
        *,
        NTILE(10) OVER (ORDER BY TotalSales DESC) as Decile
    FROM store_stats
)
SELECT
    Decile,
    COUNT(*) as NumStores,
    ROUND(AVG(TotalSales), 0) as AvgTotalSales,
    ROUND(AVG(AvgDailySales), 0) as AvgDailySales,
    MIN(TotalSales) as MinTotalSales,
    MAX(TotalSales) as MaxTotalSales
FROM ranked
GROUP BY Decile
ORDER BY Decile;

-- ------------------------------------------------------------
-- 11. Day-of-week sales pattern
-- ------------------------------------------------------------
SELECT
    DayOfWeek,
    CASE DayOfWeek
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
        WHEN 7 THEN 'Sunday'
    END as DayName,
    COUNT(*) as Days,
    ROUND(AVG(Sales), 0) as AvgSales,
    ROUND(AVG(Customers), 0) as AvgCustomers
FROM sales_with_store
WHERE Open = 1
GROUP BY DayOfWeek
ORDER BY DayOfWeek;

-- ------------------------------------------------------------
-- 12. Anomaly summary (from Phase 6 output)
-- ------------------------------------------------------------
SELECT
    COUNT(*) as TotalDays,
    SUM(CASE WHEN anomaly_z2 = 1 THEN 1 ELSE 0 END) as AnomaliesZ2,
    SUM(CASE WHEN anomaly_z3 = 1 THEN 1 ELSE 0 END) as AnomaliesZ3,
    ROUND(SUM(CASE WHEN anomaly_z2 = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as AnomalyPctZ2,
    ROUND(AVG(actual), 0) as AvgActualSales
FROM anomalies;

-- ------------------------------------------------------------
-- 13. Executive summary dashboard numbers
-- ------------------------------------------------------------
SELECT 'Total Sales (open days)' as Metric, printf('$%,.0f', SUM(Sales)) as Value
FROM sales_with_store WHERE Open = 1
UNION ALL
SELECT 'Average Daily Sales', printf('$%,.0f', AVG(Sales))
FROM sales_with_store WHERE Open = 1
UNION ALL
SELECT 'Total Store-Days Open', printf('%,d', COUNT(*))
FROM sales_with_store WHERE Open = 1
UNION ALL
SELECT 'Number of Stores', printf('%d', COUNT(DISTINCT Store))
FROM sales_with_store
UNION ALL
SELECT 'Promo Lift', printf('+%.1f%%', AVG(CASE WHEN Promo = 1 THEN Sales END) / AVG(CASE WHEN Promo = 0 THEN Sales END) * 100 - 100)
FROM sales_with_store WHERE Open = 1
UNION ALL
SELECT 'Anomalous Days (Z>2)', printf('%d', COUNT(*))
FROM anomalies WHERE anomaly_z2 = 1;