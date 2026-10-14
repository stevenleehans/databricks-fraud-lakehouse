-- =====================================================================
-- 07 · Databricks SQL — fraud monitoring dashboard
-- =====================================================================
-- Every query is scoped to one monitoring run and reads only the small
-- aggregate tables written by notebook 06 (tens to hundreds of rows), never
-- the 6.3 M-row Gold table. That keeps a Free Edition SQL warehouse well
-- inside its limits — each query below should return in seconds.
--
-- Substitute your catalog/schema if you changed them in src/fraud_common/config.py.
-- Search-and-replace `workspace.fraud_poc` and you are done.
--
-- HOW TO BUILD THE DASHBOARD
--   1. SQL > Queries > Create query, paste one block, Run, Save with the given
--      name.
--   2. On the result, choose the stated visualisation type.
--   3. SQL > Dashboards > Create dashboard > Add > Visualization, pick each
--      saved query, and arrange as suggested at the bottom of this file.
--
-- READING THE TILES — the distinction the dashboard is built around:
--   Q3/Q6 feature drift    the inputs moved. Does NOT prove the model is worse.
--   Q2/Q4  prediction drift the scores moved. An immediate early warning, still
--                           not proof.
--   Q5     performance      the model actually is worse. Requires labels, which
--                           arrive late — expect the recent windows to be blank.
-- =====================================================================


-- ---------------------------------------------------------------------
-- Q1 · Transaction volume over time
-- Visualisation: Combo — bar `transaction_volume` (left axis),
--                line `total_amount` (right axis), X = `window_id`.
-- Purpose: the denominator for every other tile. A volume collapse explains
-- an alert-rate change far more often than model decay does.
-- ---------------------------------------------------------------------
SELECT
    w.window_id,
    w.window_index,
    w.start_step,
    w.end_step,
    p.transaction_volume,
    p.total_amount,
    ROUND(p.total_amount / NULLIF(p.transaction_volume, 0), 2) AS avg_transaction_amount,
    w.label_available
FROM workspace.fraud_poc.monitoring_windows          AS w
LEFT JOIN workspace.fraud_poc.monitoring_prediction_drift AS p
       ON p.window_id = w.window_id
ORDER BY w.window_index;


-- ---------------------------------------------------------------------
-- Q2 · Fraud-score distribution over time
-- Visualisation: Heatmap — X = `window_id`, Y = `score_band`,
--                Colour = `row_share`. (A stacked bar of `row_share` works too.)
-- Purpose: shows the *shape* of the score distribution moving, not just its
-- mean. A mean can hold steady while mass migrates from the middle to both
-- tails — which changes alert volume without changing the average.
-- ---------------------------------------------------------------------
SELECT
    h.window_id,
    h.window_index,
    h.bucket_index,
    CONCAT(
        LPAD(CAST(CAST(h.bucket_lower * 100 AS INT) AS STRING), 3, '0'), '-',
        LPAD(CAST(CAST(h.bucket_upper * 100 AS INT) AS STRING), 3, '0'), '%'
    )                                    AS score_band,
    h.row_count,
    h.row_share
FROM workspace.fraud_poc.monitoring_score_histogram AS h
WHERE h.row_count > 0
ORDER BY h.window_index, h.bucket_index;


-- ---------------------------------------------------------------------
-- Q3 · Feature drift status by feature and window (PSI)
-- Visualisation: Table, with a colour rule on `status`
--                (RED #d13212, AMBER #e8a33d, GREEN #1d8102);
--                or Heatmap X = `window_id`, Y = `feature`, colour = `psi`.
-- Purpose: locate *which* input moved and *when*.
-- PSI convention: < 0.10 stable · 0.10–0.25 moderate · > 0.25 significant.
-- ---------------------------------------------------------------------
SELECT
    f.feature,
    f.feature_kind,
    f.window_id,
    f.window_index,
    ROUND(f.value, 4)  AS psi,
    f.threshold_amber,
    f.threshold_red,
    f.status
FROM workspace.fraud_poc.monitoring_feature_drift AS f
WHERE f.statistic = 'psi'
ORDER BY
    CASE f.status WHEN 'RED' THEN 0 WHEN 'AMBER' THEN 1 ELSE 2 END,
    f.value DESC,
    f.window_index;


-- ---------------------------------------------------------------------
-- Q3b · Missing-value rate change
-- Visualisation: Table (filter to status <> 'GREEN' for an exceptions view).
-- Purpose: nulls appearing in a served feature are an upstream plumbing
-- failure. They move independently of PSI — a feed that starts returning NULL
-- for `old_balance_dest` can barely shift the value distribution while
-- silently degrading every prediction.
-- ---------------------------------------------------------------------
SELECT
    f.feature,
    f.window_id,
    f.window_index,
    ROUND(f.value, 6) AS missing_rate_delta_vs_reference,
    f.threshold_amber,
    f.threshold_red,
    f.status
FROM workspace.fraud_poc.monitoring_feature_drift AS f
WHERE f.statistic = 'missing_rate_delta'
  AND f.value <> 0
ORDER BY ABS(f.value) DESC, f.window_index;


-- ---------------------------------------------------------------------
-- Q4 · Predicted-positive rate and average score
-- Visualisation: Combo — line `predicted_positive_rate_pct` and reference line
--                `reference_positive_rate_pct`; X = `window_id`.
-- Purpose: the alert-volume tile the fraud operations team actually watches.
-- `alert_rate_ratio` > 1 means more alerts than the training period produced,
-- which is a staffing question before it is a modelling question.
-- ---------------------------------------------------------------------
SELECT
    p.window_id,
    p.window_index,
    ROUND(p.predicted_positive_rate * 100, 4)           AS predicted_positive_rate_pct,
    ROUND(p.reference_predicted_positive_rate * 100, 4) AS reference_positive_rate_pct,
    p.predicted_positive_count,
    p.transaction_volume,
    ROUND(p.avg_score, 6)                               AS avg_score,
    ROUND(p.reference_avg_score, 6)                     AS reference_avg_score,
    ROUND(p.p99_score, 6)                               AS p99_score,
    ROUND(p.score_psi, 4)                               AS prediction_score_psi,
    p.status                                            AS prediction_drift_status,
    ROUND(
        p.predicted_positive_rate
        / NULLIF(p.reference_predicted_positive_rate, 0), 3
    )                                                   AS alert_rate_ratio
FROM workspace.fraud_poc.monitoring_prediction_drift AS p
ORDER BY p.window_index;


-- ---------------------------------------------------------------------
-- Q5 · Precision and recall where labels are available
-- Visualisation: Combo — lines `precision_at_threshold`, `recall_at_threshold`,
--                `pr_auc`; X = `window_id`. Add `label_status` as a table column.
-- Purpose: the only tile that can demonstrate real degradation.
-- The LEFT JOIN is deliberate: recent windows appear with NULL metrics and
-- `label_status = 'AWAITING LABELS'` rather than being dropped. A missing
-- window would read as "fine"; a NULL reads as "not yet knowable".
-- ---------------------------------------------------------------------
SELECT
    w.window_id,
    w.window_index,
    CASE WHEN w.label_available THEN 'LABELLED'
         ELSE 'AWAITING LABELS' END          AS label_status,
    m.n_labeled,
    m.n_positives,
    ROUND(m.prevalence * 100, 4)             AS fraud_prevalence_pct,
    ROUND(m.pr_auc, 4)                       AS pr_auc,
    ROUND(m.reference_pr_auc, 4)             AS reference_pr_auc,
    ROUND(m.pr_auc_delta, 4)                 AS pr_auc_delta,
    ROUND(m.roc_auc, 4)                      AS roc_auc,
    -- `precision` is an ANSI keyword (DOUBLE PRECISION); backtick the column and
    -- give the alias an unambiguous name so this parses under any SQL dialect
    -- setting.
    ROUND(m.`precision`, 4)                  AS precision_at_threshold,
    ROUND(m.recall, 4)                       AS recall_at_threshold,
    ROUND(m.f1, 4)                           AS f1,
    ROUND(m.false_positive_rate, 6)          AS false_positive_rate,
    m.true_positives,
    m.false_positives,
    m.false_negatives,
    ROUND(m.fraud_amount_capture_rate * 100, 2) AS fraud_value_captured_pct,
    m.status                                 AS performance_status
FROM workspace.fraud_poc.monitoring_windows AS w
LEFT JOIN workspace.fraud_poc.monitoring_performance AS m
       ON m.window_id = w.window_id
ORDER BY w.window_index;


-- ---------------------------------------------------------------------
-- Q6 · Red / amber / green drift indicator
-- Visualisation: Table, colour-ruled on `overall_status`. Pin it top-left.
-- Purpose: one row per window, rolling the three families up (worst wins)
-- while keeping each one's own verdict visible in its own column, so the
-- headline can never be mistaken for a claim about model performance.
-- ---------------------------------------------------------------------
WITH feature_status AS (
    SELECT
        window_id,
        MAX(CASE WHEN status = 'RED'   THEN 1 ELSE 0 END) AS any_red,
        MAX(CASE WHEN status = 'AMBER' THEN 1 ELSE 0 END) AS any_amber,
        SUM(CASE WHEN status = 'RED'   THEN 1 ELSE 0 END) AS red_feature_count,
        MAX(CASE WHEN statistic = 'psi' THEN value END)   AS worst_psi
    FROM workspace.fraud_poc.monitoring_feature_drift
    GROUP BY window_id
),
rolled AS (
    SELECT
        w.window_id,
        w.window_index,
        w.start_step,
        w.end_step,
        w.row_count,
        CASE WHEN fs.any_red = 1 THEN 'RED'
             WHEN fs.any_amber = 1 THEN 'AMBER'
             ELSE 'GREEN' END        AS feature_drift_status,
        fs.red_feature_count,
        ROUND(fs.worst_psi, 4)       AS worst_feature_psi,
        COALESCE(p.status, 'GREEN')  AS prediction_drift_status,
        ROUND(p.score_psi, 4)        AS prediction_score_psi,
        CASE WHEN w.label_available THEN COALESCE(perf.status, 'AMBER')
             ELSE 'UNKNOWN' END      AS performance_status,
        ROUND(perf.pr_auc, 4)        AS pr_auc,
        ROUND(perf.pr_auc_delta, 4)  AS pr_auc_delta
    FROM workspace.fraud_poc.monitoring_windows AS w
    LEFT JOIN feature_status                                   AS fs   ON fs.window_id = w.window_id
    LEFT JOIN workspace.fraud_poc.monitoring_prediction_drift  AS p    ON p.window_id  = w.window_id
    LEFT JOIN workspace.fraud_poc.monitoring_performance       AS perf ON perf.window_id = w.window_id
)
SELECT
    *,
    CASE
        WHEN performance_status = 'RED'
          OR feature_drift_status = 'RED'
          OR prediction_drift_status = 'RED' THEN 'RED'
        WHEN performance_status = 'AMBER'
          OR feature_drift_status = 'AMBER'
          OR prediction_drift_status = 'AMBER' THEN 'AMBER'
        ELSE 'GREEN'
    END AS overall_status,
    CASE
        WHEN performance_status IN ('RED', 'AMBER')
            THEN 'Measured performance drop — labels confirm degradation.'
        WHEN performance_status = 'UNKNOWN'
             AND (feature_drift_status <> 'GREEN' OR prediction_drift_status <> 'GREEN')
            THEN 'Inputs and/or scores have shifted, but labels for this window '
                 || 'have not arrived. Degradation is NOT established — investigate, '
                 || 'do not retrain on this signal alone.'
        WHEN feature_drift_status <> 'GREEN' OR prediction_drift_status <> 'GREEN'
            THEN 'Distribution shift with performance still within tolerance. '
                 || 'Monitor; no action required yet.'
        ELSE 'Stable.'
    END AS interpretation
FROM rolled
ORDER BY window_index;


-- ---------------------------------------------------------------------
-- Q7 · Data-quality summary (supporting tile)
-- Visualisation: Table, sorted by `failed_row_count` descending.
-- Purpose: proves the Bronze→Silver contract held for this run and shows where
-- rejected rows went. Reviewers ask for this before they trust any metric.
-- ---------------------------------------------------------------------
SELECT
    rule_name,
    severity,
    failed_row_count,
    evaluated_row_count,
    ROUND(failure_rate * 100, 6) AS failure_rate_pct,
    description
FROM workspace.fraud_poc.silver_quality_results
ORDER BY failed_row_count DESC, rule_name;


-- ---------------------------------------------------------------------
-- Q8 · Model and feature contract (supporting tile)
-- Visualisation: Table. Filter `served_to_endpoint = true` for the online
-- contract; leave it unfiltered to show the excluded columns and *why*.
-- Purpose: makes the leakage decisions auditable from the dashboard itself.
-- ---------------------------------------------------------------------
SELECT
    feature_name,
    dtype,
    availability,
    served_to_endpoint,
    signature_position,
    description,
    derivation
FROM workspace.fraud_poc.gold_feature_contract
ORDER BY served_to_endpoint DESC, signature_position, feature_name;


-- =====================================================================
-- SUGGESTED DASHBOARD LAYOUT
-- =====================================================================
--   Row 1   Q6  RAG indicator                      (full width — the headline)
--   Row 2   Q1  Transaction volume    | Q4  Predicted-positive rate
--   Row 3   Q2  Score distribution    | Q5  Precision / recall (labelled only)
--   Row 4   Q3  Feature PSI           | Q3b Missing-rate change
--   Row 5   Q7  Data quality          | Q8  Feature contract
--
-- Demo narrative, in order:
--   1. Q6 is the headline: recent windows read UNKNOWN for performance because
--      their labels have not arrived. That is the honest state, not a bug.
--   2. Q1/Q4 show what *is* observable today — volume and alert rate.
--   3. Q3 names the feature that moved.
--   4. Q5 is where a real degradation claim would have to be proven, and it
--      only covers windows old enough to have labels.
--
-- Free Edition notes:
--   * A Serverless SQL warehouse auto-stops after ~10 minutes idle; the first
--     query after a pause pays a cold start of roughly 30 seconds.
--   * Scheduled dashboard refreshes are limited. Refresh manually for a demo.
--   * All eight queries together scan well under a megabyte.
-- =====================================================================
