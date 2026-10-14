-- =====================================================================
-- 09 · Databricks SQL — fraud loss prevention simulator dashboard
-- =====================================================================
-- Companion to `07_drift_dashboard.sql`. That dashboard answers "is the model
-- still healthy?"; this one answers "what is it worth?".
--
-- Both queries below read only the two tiny Gold tables written by
-- `src/08_fraud_impact_simulation.py` (19 rows and 1 row respectively), never
-- the 6.3 M-row Gold feature table. Every tile returns in well under a second
-- on a Free Edition warehouse.
--
-- Substitute your catalog/schema if you changed them in
-- src/fraud_common/config.py — search-and-replace `workspace.fraud_poc`.
--
-- ---------------------------------------------------------------------
-- READ THIS BEFORE PRESENTING ANY NUMBER ON THIS PAGE
-- ---------------------------------------------------------------------
-- The savings shown are SIMULATED ESTIMATES based on historical transaction
-- labels, model predictions, and configurable business assumptions. They must
-- not be read as guaranteed real-world savings without production validation.
--
-- Concretely, three of the inputs are assumptions, not measurements:
--   prevention_rate         how much of a caught fraud is actually recovered
--   cost_per_review         what one manual review costs, fully loaded
--   customer_friction_cost  what wrongly stopping a good customer costs
-- Each is carried on every row as `assumption_*` so a tile can never be read
-- without them. Change them in `config.py`, re-run notebook 08, and the whole
-- dashboard moves — which is the honest way to present a business case.
--
-- HOW TO BUILD THE DASHBOARD
--   1. SQL > Queries > Create query, paste one block, Run, Save under the
--      given name.
--   2. On the result, choose the stated visualisation type.
--   3. SQL > Dashboards > Create dashboard > Add > Visualization, pick each
--      saved query, and arrange as in SUGGESTED DASHBOARD LAYOUT below.
-- =====================================================================


-- =====================================================================
-- SECTION A · EXECUTIVE KPI TILES
-- =====================================================================

-- ---------------------------------------------------------------------
-- A1 · Estimated net savings                     [Counter]
-- Visualisation: Counter — value `estimated_net_savings`, prefix from
--                `currency`, 0 decimals. Title: "Estimated NET savings
--                (simulated)".
-- Purpose: the headline. Gross fraud prevented MINUS what it costs to review
-- the queue and MINUS the cost of stopping good customers. It is the only
-- figure on the page that a finance function would recognise as a benefit.
-- ---------------------------------------------------------------------
SELECT
    ROUND(estimated_net_savings, 0)   AS estimated_net_savings,
    currency,
    recommended_threshold,
    'SIMULATED ESTIMATE — not a confirmed saving' AS caveat
FROM workspace.fraud_poc.gold_fraud_business_impact_summary;


-- ---------------------------------------------------------------------
-- A2 · The six executive tiles, as one long-form result   [Counter ×6, or Table]
-- Visualisation: Table for a single tile, or save six Counter tiles each
--                filtered to one `metric` (Counter reads the first row).
-- Purpose: the whole executive story in one query — money prevented, money
-- lost, workload created, and the operating point all three depend on.
-- Order is deliberate: benefit, then gross, then what was caught, then what
-- was NOT caught. A page that stops after the first two tiles is a sales
-- pitch, not an analysis.
-- ---------------------------------------------------------------------
SELECT
    t.display_order,
    t.metric,
    t.value,
    t.unit,
    s.currency,
    s.recommended_threshold
FROM workspace.fraud_poc.gold_fraud_business_impact_summary AS s
LATERAL VIEW EXPLODE(ARRAY(
    STRUCT(1 AS display_order, 'Estimated net savings'          AS metric, ROUND(s.estimated_net_savings, 0)          AS value, 'money' AS unit),
    STRUCT(2 AS display_order, 'Gross estimated fraud prevented' AS metric, ROUND(s.gross_estimated_savings, 0)        AS value, 'money' AS unit),
    STRUCT(3 AS display_order, 'Fraud amount detected'           AS metric, ROUND(s.detected_fraud_amount, 0)          AS value, 'money' AS unit),
    STRUCT(4 AS display_order, 'Fraud amount missed'             AS metric, ROUND(s.missed_fraud_amount, 0)            AS value, 'money' AS unit),
    STRUCT(5 AS display_order, 'Transactions flagged for review' AS metric, CAST(s.flagged_transaction_count AS DOUBLE) AS value, 'count' AS unit),
    STRUCT(6 AS display_order, 'Recommended threshold'           AS metric, s.recommended_threshold                    AS value, 'score' AS unit)
)) AS t
ORDER BY t.display_order;


-- ---------------------------------------------------------------------
-- A3 · Assumptions in force                      [Table]
-- Visualisation: Table. Pin it directly under the KPI tiles.
-- Purpose: makes the page self-refuting if the assumptions are unreasonable.
-- A reviewer who disagrees with `prevention_rate = 0.80` can see immediately
-- which number to argue with, rather than doubting the whole dashboard.
-- ---------------------------------------------------------------------
SELECT
    'Prevention rate (share of caught fraud actually recovered)' AS assumption,
    CAST(assumption_prevention_rate AS STRING)                   AS value
FROM workspace.fraud_poc.gold_fraud_business_impact_summary
UNION ALL
SELECT 'Cost per manual review',
       CONCAT(currency, ' ', CAST(assumption_cost_per_review AS STRING))
FROM workspace.fraud_poc.gold_fraud_business_impact_summary
UNION ALL
SELECT 'Customer friction cost per false positive',
       CONCAT(currency, ' ', CAST(assumption_customer_friction_cost AS STRING))
FROM workspace.fraud_poc.gold_fraud_business_impact_summary
UNION ALL
SELECT 'Analyst daily review capacity',
       CAST(assumption_analyst_daily_capacity AS STRING)
FROM workspace.fraud_poc.gold_fraud_business_impact_summary
UNION ALL
SELECT 'Simulated on',
       CONCAT(evaluation_split, ' split of ', source_table,
              ' · model version ', model_version)
FROM workspace.fraud_poc.gold_fraud_business_impact_summary;


-- =====================================================================
-- SECTION B · THRESHOLD TRADE-OFF CHART
-- =====================================================================

-- ---------------------------------------------------------------------
-- B1 · Savings and costs across the threshold sweep     [Combo / multi-line]
-- Visualisation: Line — X = `threshold`; Y = `estimated_net_savings`,
--                `gross_estimated_savings`, `review_cost`,
--                `false_positive_friction_cost`. Mark the recommended
--                threshold with a reference line.
-- Purpose: the single most useful chart on the page. Gross savings fall
-- monotonically as the threshold rises (fewer frauds caught) while both costs
-- fall faster at the low end — so net savings peak somewhere in the middle.
-- That peak is the recommendation, and its flatness tells you how much the
-- recommendation is worth arguing about.
-- ---------------------------------------------------------------------
SELECT
    sim.threshold,
    ROUND(sim.gross_estimated_savings, 0)      AS gross_estimated_savings,
    ROUND(sim.review_cost, 0)                  AS review_cost,
    ROUND(sim.false_positive_friction_cost, 0) AS false_positive_friction_cost,
    ROUND(sim.estimated_net_savings, 0)        AS estimated_net_savings,
    sim.currency,
    CASE WHEN sim.threshold = s.recommended_threshold
         THEN 'RECOMMENDED' ELSE '' END        AS marker
FROM workspace.fraud_poc.gold_fraud_impact_simulation        AS sim
CROSS JOIN workspace.fraud_poc.gold_fraud_business_impact_summary AS s
ORDER BY sim.threshold;


-- =====================================================================
-- SECTION C · FRAUD CAPTURED VS MISSED
-- =====================================================================

-- ---------------------------------------------------------------------
-- C1 · Fraud value caught vs let through          [Stacked bar]
-- Visualisation: Bar (stacked) — X = `threshold`,
--                Y = `detected_fraud_amount` + `missed_fraud_amount`.
--                The two always sum to `actual_fraud_amount`.
-- Purpose: reframes recall in money. 80 % recall sounds strong until the
-- missing 20 % turns out to be the large-value transactions — this chart makes
-- that visible where a recall number hides it.
-- ---------------------------------------------------------------------
SELECT
    threshold,
    ROUND(detected_fraud_amount, 0)         AS detected_fraud_amount,
    ROUND(missed_fraud_amount, 0)           AS missed_fraud_amount,
    ROUND(actual_fraud_amount, 0)           AS actual_fraud_amount,
    ROUND(fraud_detection_rate * 100, 2)    AS fraud_value_caught_pct,
    actual_fraud_count,
    currency
FROM workspace.fraud_poc.gold_fraud_impact_simulation
ORDER BY threshold;


-- =====================================================================
-- SECTION D · REVIEW WORKLOAD SIMULATION
-- =====================================================================

-- ---------------------------------------------------------------------
-- D1 · Alert volume and the staffing it implies    [Combo]
-- Visualisation: Combo — bar `flagged_transaction_count` (left axis),
--                line `estimated_review_days_required` (right axis),
--                X = `threshold`.
-- Purpose: the constraint that usually decides the operating point in
-- practice. A threshold that maximises modelled savings but needs forty
-- analyst-days per batch is not an option; `analyst_days_at_current_headcount`
-- turns the trade-off into a hiring question.
-- ---------------------------------------------------------------------
SELECT
    threshold,
    flagged_transaction_count,
    ROUND(flagged_transaction_amount, 0)        AS flagged_transaction_amount,
    ROUND(estimated_review_days_required, 2)    AS estimated_review_days_required,
    assumption_analyst_daily_capacity           AS analyst_daily_capacity,
    ROUND(alerts_per_1k_transactions, 2)        AS alerts_per_1k_transactions,
    ROUND(average_saving_per_review, 2)         AS average_saving_per_review,
    currency
FROM workspace.fraud_poc.gold_fraud_impact_simulation
ORDER BY threshold;


-- =====================================================================
-- SECTION E · MODEL QUALITY VS BUSINESS VALUE
-- =====================================================================

-- ---------------------------------------------------------------------
-- E1 · Precision, recall and FPR against net savings   [Combo]
-- Visualisation: Combo — lines `precision_pct`, `recall_pct`,
--                `false_positive_rate_pct` (left axis, 0–100),
--                line `estimated_net_savings` (right axis), X = `threshold`.
-- Purpose: the bridge between the two dashboards. The threshold that maximises
-- money is rarely the one that maximises F1 or precision, because the cost of a
-- false positive and the value of a true positive are not equal. This chart is
-- where a data scientist and a business owner stop talking past each other.
-- ---------------------------------------------------------------------
SELECT
    threshold,
    -- `precision` is an ANSI keyword (DOUBLE PRECISION); backtick the column
    -- or the parser reports a baffling syntax error at the next token.
    ROUND(`precision` * 100, 2)             AS precision_pct,
    ROUND(recall * 100, 2)                AS recall_pct,
    ROUND(false_positive_rate * 100, 4)   AS false_positive_rate_pct,
    ROUND(estimated_net_savings, 0)       AS estimated_net_savings,
    true_positive_count,
    false_positive_count,
    false_negative_count,
    currency
FROM workspace.fraud_poc.gold_fraud_impact_simulation
ORDER BY threshold;


-- =====================================================================
-- SECTION F · FINAL BUSINESS RECOMMENDATION
-- =====================================================================

-- ---------------------------------------------------------------------
-- F1 · The recommendation, in words                [Table, or Counter + Text]
-- Visualisation: Table with wide columns, full page width, placed last.
-- Purpose: states the conclusion, the basis for it, the caveat, and the
-- disclaimer in one place, so the page cannot be screenshotted into a slide
-- without them.
-- ---------------------------------------------------------------------
SELECT
    recommended_threshold,
    recommendation_note,
    selection_basis,
    business_override_note,
    result_disclaimer,
    ROUND(estimated_net_savings, 0)           AS estimated_net_savings,
    ROUND(estimated_review_days_required, 2)  AS estimated_review_days_required,
    ROUND(`precision` * 100, 2)                 AS precision_pct,
    ROUND(recall * 100, 2)                    AS recall_pct,
    currency,
    simulated_at
FROM workspace.fraud_poc.gold_fraud_business_impact_summary;


-- ---------------------------------------------------------------------
-- F2 · Neighbouring thresholds — how much does the choice matter?  [Table]
-- Visualisation: Table, with `is_recommended` colour-ruled.
-- Purpose: guards against false precision. If 0.40 and 0.50 are worth almost
-- the same, "the optimum is 0.45" is noise, and the operating point should be
-- chosen on review capacity instead. `pct_of_best_net_savings` says exactly how
-- much money a different choice costs.
-- ---------------------------------------------------------------------
SELECT
    sim.threshold,
    CASE WHEN sim.threshold = s.recommended_threshold THEN true ELSE false END
                                                       AS is_recommended,
    ROUND(sim.estimated_net_savings, 0)                AS estimated_net_savings,
    ROUND(100.0 * sim.estimated_net_savings
          / NULLIF(s.estimated_net_savings, 0), 1)     AS pct_of_best_net_savings,
    sim.flagged_transaction_count,
    ROUND(sim.estimated_review_days_required, 2)       AS estimated_review_days_required,
    ROUND(sim.`precision` * 100, 2)                      AS precision_pct,
    ROUND(sim.recall * 100, 2)                         AS recall_pct
FROM workspace.fraud_poc.gold_fraud_impact_simulation             AS sim
CROSS JOIN workspace.fraud_poc.gold_fraud_business_impact_summary AS s
WHERE ABS(sim.threshold - s.recommended_threshold) <= 0.15
ORDER BY sim.threshold;


-- =====================================================================
-- SUGGESTED DASHBOARD LAYOUT
-- =====================================================================
--   Row 1   A1  Estimated net savings (large counter, full width)
--   Row 2   A2  The six executive tiles
--   Row 3   A3  Assumptions in force   | C1  Fraud caught vs missed
--   Row 4   B1  Threshold trade-off    (full width — the analytical centre)
--   Row 5   D1  Review workload        | E1  Model quality vs value
--   Row 6   F1  Recommendation         (full width, last)
--   Row 7   F2  Neighbouring thresholds
--
-- Demo narrative, in order:
--   1. A1 gives the headline, immediately followed by A3 so the assumptions
--      arrive before anyone starts believing the number.
--   2. B1 shows WHY the recommended threshold is where it is: gross savings
--      and costs move in opposite directions and net savings peaks between.
--   3. C1 converts recall into money — the fraud value still getting through.
--   4. D1 is the reality check: can the review team actually work this queue?
--   5. E1 shows that the best model threshold and the best business threshold
--      are not the same point.
--   6. F1 closes with the recommendation AND the validation caveat.
--
-- Free Edition notes:
--   * Both tables are under 30 rows; the whole page scans a few kilobytes.
--   * A serverless SQL warehouse auto-stops after ~10 minutes idle, so the
--     first query after a pause pays a ~30 second cold start.
--   * Re-running notebook 08 overwrites both tables; refresh the dashboard
--     manually afterwards.
-- =====================================================================
