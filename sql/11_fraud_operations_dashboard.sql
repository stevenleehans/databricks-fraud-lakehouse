-- =====================================================================
-- 11 · Databricks SQL — operational replay dashboard (time series)
-- =====================================================================
-- The third and last dashboard. The other two are cross-sections; this one is
-- the only page where the x-axis is TIME.
--
--   07  drift watch      "is the model still healthy?"    x = monitoring window
--   09  business impact  "which threshold should we run?" x = threshold
--   11  operations       "we shipped it — how is it going?" x = CLOCK TIME
--
-- Every tile below reads `gold_fraud_operations_timeseries` (one row per hour),
-- `gold_fraud_operations_hourly_profile` (24 rows) or
-- `gold_fraud_operations_summary` (1 row) — all written by
-- `src/10_fraud_operations_timeseries.py`. A few hundred rows in total, so the
-- whole page is well inside a Free Edition warehouse's budget.
--
-- Substitute your catalog/schema if you changed them in
-- src/fraud_common/config.py — search-and-replace `workspace.fraud_poc`.
--
-- ---------------------------------------------------------------------
-- READ THIS BEFORE PRESENTING ANY NUMBER ON THIS PAGE
-- ---------------------------------------------------------------------
-- 1. THE DATES ARE SYNTHETIC. PaySim ships no wall-clock timestamps, only an
--    hour counter (`step`). Silver anchors it to 2024-01-01 purely so a chart
--    can have a real time axis. No claim is made that anything happened on
--    these dates.
--
-- 2. THIS IS A REPLAY, NOT A PRODUCTION RECORD. The model is applied at one
--    frozen threshold to transactions it did not influence. In reality,
--    blocking fraud changes fraudster behaviour and reviewed alerts change what
--    the next model learns. Neither is modelled.
--
-- 3. The savings are SIMULATED ESTIMATES based on historical transaction
--    labels, model predictions, and configurable business assumptions. Without
--    production validation they must not be read as guaranteed real savings.
--
-- Operational cost on this page = review cost + false-positive friction cost.
-- Exactly the two components dashboard 09 subtracts — one definition of cost
-- across both pages, never a second one invented here.
--
-- HOW TO BUILD THE DASHBOARD
--   Normally you do not: `python dashboards/build_lvdash.py` generates it and
--   the bundle deploys it. To build by hand, paste one block per query, use the
--   visualisation named in its header, and follow SUGGESTED DASHBOARD LAYOUT.
-- =====================================================================


-- =====================================================================
-- SECTION T · SINCE GO-LIVE HEADLINES
-- =====================================================================

-- ---------------------------------------------------------------------
-- T1 · Cumulative outcome of the whole replayed period    [Table / Counters]
-- Visualisation: Table, full width, pinned at the top.
-- Purpose: the operational bottom line in one row — what was caught, what it
-- saved, what it cost to run, and whether the desk could keep up. Everything
-- below is this row decomposed over time.
-- ---------------------------------------------------------------------
SELECT
    periods_observed,
    period_grain,
    first_period,
    last_period,
    operating_threshold,
    total_transactions,
    total_alerts,
    total_caught_fraud_count,
    total_missed_fraud_count,
    ROUND(total_detected_fraud_amount, 0) AS fraud_value_caught,
    ROUND(total_missed_fraud_amount, 0)   AS fraud_value_missed,
    ROUND(total_gross_savings, 0)         AS gross_savings,
    ROUND(total_operational_cost, 0)      AS operational_cost,
    ROUND(total_net_savings, 0)           AS net_savings,
    ROUND(savings_to_cost_ratio, 1)       AS savings_per_unit_cost,
    periods_over_capacity,
    ROUND(peak_backlog_count, 0)          AS peak_backlog,
    currency,
    operating_note,
    result_disclaimer
FROM workspace.fraud_poc.gold_fraud_operations_summary;


-- ---------------------------------------------------------------------
-- T2 · Return on every unit of operational spend          [Counter]
-- Visualisation: Counter — value `savings_per_unit_cost`, 1 decimal,
--                suffix "×". Title: "Gross savings per 1 unit spent reviewing".
-- Purpose: the single most quotable operational number. "Every dollar we spend
-- working the queue returns N in prevented fraud." Unitless, so it survives a
-- change of currency and is the fairest way to compare two operating points.
-- ---------------------------------------------------------------------
SELECT
    ROUND(savings_to_cost_ratio, 1)  AS savings_per_unit_cost,
    ROUND(total_gross_savings, 0)    AS gross_savings,
    ROUND(total_operational_cost, 0) AS operational_cost,
    currency
FROM workspace.fraud_poc.gold_fraud_operations_summary;


-- =====================================================================
-- SECTION U · MONEY OVER TIME — SAVINGS VS OPERATIONAL COST
-- =====================================================================

-- ---------------------------------------------------------------------
-- U1 · Savings and operational cost, hour by hour       [Line, time axis]
-- Visualisation: Line — X = `period_start` (temporal), Y =
--                `gross_savings`, `operational_cost`, `net_savings`.
-- Purpose: the core tile of this dashboard. Two series that a business cares
-- about on the same clock: what the model prevented, and what preventing it
-- cost to operate. The gap between them is the benefit, and its stability over
-- time matters more than its size in any one hour.
-- Note: gross savings is spiky by nature — one large fraudulent transfer moves
-- it far more than a busy hour of small ones does. Operational cost is smooth,
-- because it tracks alert VOLUME, not value. That contrast is the point.
-- ---------------------------------------------------------------------
SELECT
    period_start,
    ROUND(gross_estimated_savings, 0) AS gross_savings,
    ROUND(operational_cost, 0)        AS operational_cost,
    ROUND(estimated_net_savings, 0)   AS net_savings,
    currency
FROM workspace.fraud_poc.gold_fraud_operations_timeseries
ORDER BY period_start;


-- ---------------------------------------------------------------------
-- U2 · Cumulative net savings since go-live             [Line, time axis]
-- Visualisation: Line — X = `period_start` (temporal), Y =
--                `cumulative_net_savings`, `cumulative_operational_cost`.
-- Purpose: the payback curve, and the tile an executive actually remembers. A
-- rising line means the deployment is compounding; a flat stretch means the
-- model found nothing while the desk kept being paid. If the cumulative cost
-- line ever crosses above cumulative savings, the deployment is under water.
-- ---------------------------------------------------------------------
SELECT
    period_start,
    period_index,
    ROUND(cumulative_net_savings, 0)         AS cumulative_net_savings,
    ROUND(cumulative_gross_savings, 0)       AS cumulative_gross_savings,
    ROUND(cumulative_operational_cost, 0)    AS cumulative_operational_cost,
    ROUND(cumulative_savings_to_cost_ratio, 2) AS cumulative_savings_per_unit_cost,
    currency
FROM workspace.fraud_poc.gold_fraud_operations_timeseries
ORDER BY period_start;


-- ---------------------------------------------------------------------
-- U3 · Daily roll-up — savings vs cost                  [Combo, daily]
-- Visualisation: Combo — bar `operational_cost` and bar `net_savings` by
--                `event_date`; or Line for both.
-- Purpose: the hourly series is noisy enough that a stakeholder argues with the
-- noise instead of the trend. The daily roll-up is the version that goes on a
-- slide. Same data, aggregated — not a different calculation.
-- ---------------------------------------------------------------------
SELECT
    event_date,
    SUM(total_transactions)                  AS transactions,
    SUM(flagged_transaction_count)           AS alerts,
    SUM(caught_fraud_count)                  AS fraud_caught,
    SUM(missed_fraud_count)                  AS fraud_missed,
    ROUND(SUM(detected_fraud_amount), 0)     AS fraud_value_caught,
    ROUND(SUM(gross_estimated_savings), 0)   AS gross_savings,
    ROUND(SUM(operational_cost), 0)          AS operational_cost,
    ROUND(SUM(estimated_net_savings), 0)     AS net_savings,
    ROUND(SUM(gross_estimated_savings) / NULLIF(SUM(operational_cost), 0), 1)
                                             AS savings_per_unit_cost,
    MIN(currency)                            AS currency
FROM workspace.fraud_poc.gold_fraud_operations_timeseries
GROUP BY event_date
ORDER BY event_date;


-- =====================================================================
-- SECTION V · WHAT WE CAUGHT, OVER TIME
-- =====================================================================

-- ---------------------------------------------------------------------
-- V1 · Fraud caught vs missed, per period               [Stacked bar, time]
-- Visualisation: Bar (stacked) — X = `period_start`, Y = `fraud_caught` +
--                `fraud_missed`. Counts, not money.
-- Purpose: "how many did we catch" as a running record rather than a single
-- recall figure. A recall of 99 % is a number; a bar that turns red in one
-- specific hour is an incident someone can investigate.
-- ---------------------------------------------------------------------
SELECT
    period_start,
    caught_fraud_count  AS fraud_caught,
    missed_fraud_count  AS fraud_missed,
    actual_fraud_count  AS fraud_total,
    ROUND(recall * 100, 1) AS recall_pct
FROM workspace.fraud_poc.gold_fraud_operations_timeseries
ORDER BY period_start;


-- ---------------------------------------------------------------------
-- V2 · Fraud VALUE caught vs missed, per period         [Stacked bar, time]
-- Visualisation: Bar (stacked) — X = `period_start`, Y =
--                `fraud_value_caught` + `fraud_value_missed`.
-- Purpose: the same picture weighted by money. Read it against V1: a period
-- where the counts look fine and this one does not is a period where the model
-- missed the expensive ones, which is the failure that actually costs.
-- ---------------------------------------------------------------------
SELECT
    period_start,
    ROUND(detected_fraud_amount, 0) AS fraud_value_caught,
    ROUND(missed_fraud_amount, 0)   AS fraud_value_missed,
    currency
FROM workspace.fraud_poc.gold_fraud_operations_timeseries
ORDER BY period_start;


-- =====================================================================
-- SECTION W · CAN THE DESK KEEP UP?
-- =====================================================================

-- ---------------------------------------------------------------------
-- W1 · Alert volume against staffed review capacity     [Combo, time axis]
-- Visualisation: Combo — bar `alerts` (left axis), line `review_capacity`
--                (same axis, flat reference), line `backlog`.
-- Purpose: the operational constraint, plotted. Alerts above the capacity line
-- do not disappear — they become a backlog, and a backlog is how a fraud desk
-- fails even when the model is right. This tile decides headcount.
-- ---------------------------------------------------------------------
SELECT
    period_start,
    flagged_transaction_count       AS alerts,
    ROUND(review_capacity, 1)       AS review_capacity,
    ROUND(queue_backlog_count, 0)   AS backlog,
    ROUND(capacity_utilisation * 100, 1) AS capacity_utilisation_pct,
    capacity_exceeded,
    analyst_headcount
FROM workspace.fraud_poc.gold_fraud_operations_timeseries
ORDER BY period_start;


-- ---------------------------------------------------------------------
-- W2 · Alert load by hour of day                        [Bar]
-- Visualisation: Bar — X = `event_hour_of_day`, Y = `avg_alerts`, with
--                `review_capacity` as a reference line.
-- Purpose: shift planning. Fraud is not uniform across the clock, so neither is
-- the queue. This is the tile that answers "do we need a night shift?", which
-- no amount of model accuracy can answer.
-- ---------------------------------------------------------------------
SELECT
    event_hour_of_day,
    ROUND(avg_transactions, 0)            AS avg_transactions,
    ROUND(avg_alerts, 1)                  AS avg_alerts,
    peak_alerts,
    ROUND(review_capacity, 1)             AS review_capacity,
    ROUND(avg_capacity_utilisation * 100, 1) AS avg_capacity_utilisation_pct,
    periods_over_capacity,
    ROUND(avg_fraud_caught, 2)            AS avg_fraud_caught,
    ROUND(avg_net_savings, 0)             AS avg_net_savings
FROM workspace.fraud_poc.gold_fraud_operations_hourly_profile
ORDER BY event_hour_of_day;


-- =====================================================================
-- SECTION X · DETECTION QUALITY ON THE CLOCK
-- =====================================================================

-- ---------------------------------------------------------------------
-- X1 · Precision and recall per period                  [Line, time axis]
-- Visualisation: Line — X = `period_start`, Y = `precision_pct`, `recall_pct`
--                (both 0–100).
-- Purpose: the same two metrics the drift dashboard reports per window, here at
-- operating cadence. A precision dip is a false-positive surge the desk feels
-- immediately; a recall dip is fraud getting through, which nobody feels until
-- the chargebacks land. Expect these to be noisy on low-volume hours — a period
-- with two frauds has a recall of 0 %, 50 % or 100 % and nothing in between.
-- ---------------------------------------------------------------------
SELECT
    period_start,
    ROUND(`precision` * 100, 1)          AS precision_pct,
    ROUND(recall * 100, 1)               AS recall_pct,
    true_positive_count,
    false_positive_count,
    false_negative_count,
    actual_fraud_count
FROM workspace.fraud_poc.gold_fraud_operations_timeseries
WHERE actual_fraud_count > 0
ORDER BY period_start;


-- ---------------------------------------------------------------------
-- X2 · Worst operating periods                          [Table]
-- Visualisation: Table, sorted as written. Colour `net_savings` red below 0.
-- Purpose: the exceptions view. Any period that lost money, missed fraud, or
-- overflowed the queue — the shortlist an operations lead would actually work
-- through on a Monday morning. An empty table is the healthy state.
-- ---------------------------------------------------------------------
SELECT
    period_start,
    total_transactions,
    flagged_transaction_count      AS alerts,
    caught_fraud_count             AS fraud_caught,
    missed_fraud_count             AS fraud_missed,
    ROUND(missed_fraud_amount, 0)  AS fraud_value_missed,
    ROUND(operational_cost, 0)     AS operational_cost,
    ROUND(estimated_net_savings, 0) AS net_savings,
    ROUND(queue_backlog_count, 0)  AS backlog,
    CASE
        WHEN NOT net_savings_is_positive AND capacity_exceeded
            THEN 'LOSS-MAKING + OVER CAPACITY'
        WHEN NOT net_savings_is_positive THEN 'LOSS-MAKING'
        WHEN capacity_exceeded           THEN 'OVER CAPACITY'
        ELSE 'FRAUD MISSED'
    END                            AS issue
FROM workspace.fraud_poc.gold_fraud_operations_timeseries
WHERE NOT net_savings_is_positive
   OR capacity_exceeded
   OR missed_fraud_count > 0
ORDER BY estimated_net_savings ASC, missed_fraud_amount DESC
LIMIT 50;


-- =====================================================================
-- SUGGESTED DASHBOARD LAYOUT
-- =====================================================================
--   Row 1   T2  Savings per unit cost  | T1  Since go-live roll-up
--   Row 2   U2  Cumulative net savings (full width — the payback curve)
--   Row 3   U1  Savings vs operational cost, hourly (full width)
--   Row 4   U3  Daily roll-up          | W1  Alerts vs capacity
--   Row 5   V1  Fraud caught vs missed | V2  Fraud VALUE caught vs missed
--   Row 6   W2  Alert load by hour     | X1  Precision / recall over time
--   Row 7   X2  Worst operating periods (full width, last)
--
-- Demo narrative, in order:
--   1. T2 opens with the ratio, not the raw number — it needs no currency and
--      no context to be understood.
--   2. U2 is the payback curve: is the deployment compounding, or flat?
--   3. U1 shows the two forces underneath it. Point out that cost is smooth and
--      savings are spiky, and say why: cost tracks alert VOLUME, savings track
--      transaction VALUE.
--   4. W1/W2 change the conversation from accuracy to staffing. This is the
--      part a fraud operations lead cares about and a model metric cannot answer.
--   5. X2 closes on the exceptions — the periods that would have needed a human.
--   6. Say the caveat out loud: synthetic dates, a replay, simulated savings.
--
-- Free Edition notes:
--   * A few hundred rows across all three tables; the page scans kilobytes.
--   * Re-running notebook 10 overwrites all three; refresh manually afterwards.
-- =====================================================================
