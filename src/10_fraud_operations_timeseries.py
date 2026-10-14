# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · Operational replay — the model as if it had shipped
# MAGIC
# MAGIC Notebook 08 asks **"which threshold should we run?"** — one population,
# MAGIC nineteen thresholds. This notebook asks the opposite question:
# MAGIC
# MAGIC > **We shipped it. What did each hour of running it actually look like?**
# MAGIC
# MAGIC One frozen threshold, many time periods. How many frauds were caught per
# MAGIC hour, how much money that saved, what the alert queue cost to work — and
# MAGIC whether the team staffed to work it could actually keep up.
# MAGIC
# MAGIC ## The time axis is synthetic
# MAGIC
# MAGIC PaySim ships no wall-clock timestamps, only `step` — an hour counter. Silver
# MAGIC anchors it to `SIMULATION_START_TS` (2024-01-01) purely so a dashboard can
# MAGIC plot a real time axis. **These are not real dates**, and the derivation here
# MAGIC is the same expression notebook 02 uses (`schema.step_to_timestamp_sql`), so
# MAGIC the two time axes cannot drift apart.
# MAGIC
# MAGIC ## What "as if it had shipped" does and does not mean
# MAGIC
# MAGIC This is a **replay**, not a production record. The model is applied to
# MAGIC transactions it did not influence. In reality:
# MAGIC
# MAGIC - blocking fraud changes fraudster behaviour, so the fraud in later periods
# MAGIC   would not be the fraud that is in this data;
# MAGIC - reviewed alerts generate labels that change what the next model learns;
# MAGIC - a real desk works a queue with a delay, so an alert raised at 02:00 is not
# MAGIC   necessarily actioned at 02:00.
# MAGIC
# MAGIC None of that is modelled. The replay is still worth doing — it turns a
# MAGIC single PR-AUC into a staffing plan and a cost line — but it is a simulation
# MAGIC under stated assumptions, exactly like notebook 08.
# MAGIC
# MAGIC > **The savings shown are simulated estimates based on historical
# MAGIC > transaction labels, model predictions, and configurable business
# MAGIC > assumptions. They should not be interpreted as
# MAGIC > guaranteed real-world savings without production validation.**
# MAGIC
# MAGIC ## Outputs
# MAGIC
# MAGIC | Table | Grain | Feeds |
# MAGIC |---|---|---|
# MAGIC | `gold_fraud_operations_timeseries` | one row per hour (or day) | every time-series tile |
# MAGIC | `gold_fraud_operations_hourly_profile` | one row per hour of day | shift planning |
# MAGIC | `gold_fraud_operations_summary` | one row | the "since go-live" KPI tiles |

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json

import pandas as pd
from pyspark.sql import functions as F

from fraud_common import impact

spark_session = get_spark()
REPLAYED_AT = datetime.now(timezone.utc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Resolve the input and the frozen threshold
# MAGIC
# MAGIC The threshold is **not** re-optimised here. Re-choosing it per period would
# MAGIC be hindsight: a production desk picks an operating point once and lives with
# MAGIC it while the world moves. Preference order:
# MAGIC
# MAGIC 1. `ops_threshold_override`, when set
# MAGIC 2. `recommended_threshold` from notebook 08's summary table
# MAGIC 3. `CFG.operational_threshold` — the value notebook 04 chose on validation

# COMMAND ----------

if not table_exists(CFG.scored_predictions_table, spark_session):
    raise RuntimeError(
        f"{CFG.scored_predictions_table} does not exist. Run notebook 04 first "
        "— this notebook replays its batch scores over time."
    )

scored_sdf = spark_session.table(CFG.scored_predictions_table)
AVAILABLE = scored_sdf.columns

SCORE_COLUMN = impact.resolve_column(AVAILABLE, impact.SCORE_COLUMN_CANDIDATES, "score")
LABEL_COLUMN = impact.resolve_column(AVAILABLE, impact.LABEL_COLUMN_CANDIDATES, "label")
AMOUNT_COLUMN = impact.resolve_column(AVAILABLE, impact.AMOUNT_COLUMN_CANDIDATES, "amount")


def _resolve_threshold() -> tuple:
    """(threshold, where it came from). Never silently defaults."""
    if CFG.ops_threshold_override is not None:
        return float(CFG.ops_threshold_override), "ops_threshold_override"
    if table_exists(CFG.fraud_impact_summary_table, spark_session):
        rows = (spark_session.table(CFG.fraud_impact_summary_table)
                .select("recommended_threshold").limit(1).collect())
        if rows and rows[0][0] is not None:
            return float(rows[0][0]), f"recommended by {CFG.fraud_impact_summary_table}"
    return float(CFG.operational_threshold), "CFG.operational_threshold"


THRESHOLD, THRESHOLD_SOURCE = _resolve_threshold()
log("ops", f"frozen operating threshold: {THRESHOLD:.4f} ({THRESHOLD_SOURCE})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Build the time axis
# MAGIC
# MAGIC `step` becomes a timestamp using the expression shared with notebook 02, so
# MAGIC this dashboard's x-axis and Silver's `event_timestamp` are the same clock.

# COMMAND ----------

SPLIT_SELECTION = {
    "test": [splits.TEST],
    "valid": [splits.VALID],
    "holdout": [splits.VALID, splits.TEST],
    "all": None,
}[CFG.ops_evaluation_split]

replay_sdf = scored_sdf
if SPLIT_SELECTION is not None:
    replay_sdf = replay_sdf.where(F.col("split_label").isin(SPLIT_SELECTION))
    log("ops", f"replaying split_label IN {SPLIT_SELECTION} "
               f"(ops_evaluation_split={CFG.ops_evaluation_split!r})")

# `train` rows in scored_predictions are a *sample* at natural prevalence, not
# the full period, so including them would understate volume for those hours.
# Guard rather than silently plot a dip that is an artefact of sampling.
if SPLIT_SELECTION is None:
    log("ops", "WARNING: ops_evaluation_split='all' includes the train-period "
               "SAMPLE, so early periods will show artificially low volume.")

PERIOD_FRACTION_OF_DAY = 1.0 / 24.0 if CFG.ops_time_grain == "hour" else 1.0
TRUNC_UNIT = "hour" if CFG.ops_time_grain == "hour" else "day"

replay_sdf = (
    replay_sdf
    .where(F.col(LABEL_COLUMN).isNotNull() & F.col(AMOUNT_COLUMN).isNotNull())
    .withColumn("event_timestamp", F.expr(schema.step_to_timestamp_sql()))
    .withColumn("period_start", F.date_trunc(TRUNC_UNIT, F.col("event_timestamp")))
    .withColumn("event_date", F.to_date("event_timestamp"))
    .withColumn("event_hour_of_day", F.hour("event_timestamp"))
    .withColumn("event_day_of_week", F.date_format("event_timestamp", "EEE"))
    .select(
        "period_start", "event_date", "event_hour_of_day", "event_day_of_week",
        F.col(SCORE_COLUMN).cast("double").alias("fraud_score"),
        F.col(LABEL_COLUMN).cast("int").alias("label_true"),
        F.col(AMOUNT_COLUMN).cast("double").alias("amount"),
    )
)

REPLAY_ROWS = replay_sdf.count()
if REPLAY_ROWS == 0:
    raise RuntimeError(
        f"no rows to replay for ops_evaluation_split={CFG.ops_evaluation_split!r}."
    )
log("ops", f"replaying {REPLAY_ROWS:,} scored transactions at {CFG.ops_time_grain} grain")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Aggregate every period in one pass
# MAGIC
# MAGIC The threshold is fixed, so unlike notebook 08 this is a plain `groupBy` —
# MAGIC one conditional-sum set, grouped by period, rather than one set per
# MAGIC threshold.

# COMMAND ----------

flagged = F.col("fraud_score") >= F.lit(THRESHOLD)
is_fraud = F.col("label_true") == F.lit(1)
zero = F.lit(0.0)

period_sdf = (
    replay_sdf.groupBy("period_start")
    .agg(
        # population for the period
        F.count(F.lit(1)).alias("total_transactions"),
        F.sum("amount").alias("total_transaction_amount"),
        F.sum(F.col("label_true")).alias("actual_fraud_count"),
        F.sum(F.when(is_fraud, F.col("amount")).otherwise(zero))
            .alias("actual_fraud_amount"),
        # confusion at the frozen threshold
        F.sum(F.when(flagged, 1).otherwise(0)).alias("flagged_transaction_count"),
        F.sum(F.when(flagged, F.col("amount")).otherwise(zero))
            .alias("flagged_transaction_amount"),
        F.sum(F.when(flagged & is_fraud, 1).otherwise(0)).alias("true_positive_count"),
        F.sum(F.when(flagged & ~is_fraud, 1).otherwise(0)).alias("false_positive_count"),
        F.sum(F.when(~flagged & is_fraud, 1).otherwise(0)).alias("false_negative_count"),
        F.sum(F.when(~flagged & ~is_fraud, 1).otherwise(0)).alias("true_negative_count"),
        F.sum(F.when(flagged & is_fraud, F.col("amount")).otherwise(zero))
            .alias("detected_fraud_amount"),
        F.sum(F.when(~flagged & is_fraud, F.col("amount")).otherwise(zero))
            .alias("missed_fraud_amount"),
        F.min("event_date").alias("event_date"),
        F.min("event_hour_of_day").alias("event_hour_of_day"),
        F.min("event_day_of_week").alias("event_day_of_week"),
    )
    .orderBy("period_start")
)

period_pdf = period_sdf.toPandas()
log("ops", f"{len(period_pdf)} operating periods "
           f"[{period_pdf['period_start'].min()} … {period_pdf['period_start'].max()}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Apply the financial and operational model
# MAGIC
# MAGIC Money comes from `impact.derive_impact_row`, exactly as in notebook 08 — one
# MAGIC definition of a saving across both dashboards. `derive_operations_row` adds
# MAGIC the operational overlay: total cost of running the queue, savings per unit
# MAGIC of that cost, and whether the staffed team could clear it.

# COMMAND ----------

ASSUMPTIONS = impact.ImpactAssumptions(
    prevention_rate=CFG.impact_prevention_rate,
    cost_per_review=CFG.impact_cost_per_review,
    customer_friction_cost=CFG.impact_customer_friction_cost,
    analyst_daily_capacity=CFG.impact_analyst_daily_capacity,
    currency=CFG.impact_currency,
).validate()

print(f"Frozen threshold      = {THRESHOLD:.4f}  ({THRESHOLD_SOURCE})")
print(f"Analysts staffed      = {CFG.ops_analyst_headcount}")
print(f"Capacity per {CFG.ops_time_grain:<10}= "
      f"{CFG.ops_analyst_headcount * CFG.impact_analyst_daily_capacity * PERIOD_FRACTION_OF_DAY:,.1f} alerts")
for key, value in ASSUMPTIONS.to_dict().items():
    print(f"  {key:<24} = {value!r}")

CONTEXT = {
    "source_table": CFG.scored_predictions_table,
    "evaluation_split": CFG.ops_evaluation_split,
    "period_grain": CFG.ops_time_grain,
    "operating_threshold": float(THRESHOLD),
    "threshold_source": THRESHOLD_SOURCE,
    "currency": CFG.impact_currency,
    "replayed_at": REPLAYED_AT.isoformat(timespec="seconds"),
    "result_disclaimer": impact.DISCLAIMER,
}

operations_rows = []
for record in period_pdf.to_dict("records"):
    counts = {field: float(record[field]) for field in impact.COUNT_FIELDS}
    population = {field: float(record[field]) for field in impact.POPULATION_FIELDS}
    row = impact.derive_operations_row(
        period_key=str(record["period_start"]),
        threshold=THRESHOLD,
        counts=counts,
        population=population,
        assumptions=ASSUMPTIONS,
        analyst_headcount=CFG.ops_analyst_headcount,
        period_fraction_of_day=PERIOD_FRACTION_OF_DAY,
    )
    row["period_start"] = record["period_start"]
    row["event_date"] = record["event_date"]
    row["event_hour_of_day"] = int(record["event_hour_of_day"])
    row["event_day_of_week"] = str(record["event_day_of_week"])
    row.update(CONTEXT)
    operations_rows.append(row)

# Sorted by construction (the groupBy was ordered) -- add_cumulative refuses to
# sort for you, because a cumulative series built on unsorted rows still looks
# like a plausible curve.
operations_rows = impact.add_cumulative(operations_rows)
log("ops", f"{len(operations_rows)} periods costed and accumulated")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Write `gold_fraud_operations_timeseries`

# COMMAND ----------

operations_pdf = pd.DataFrame(operations_rows).sort_values("period_start")
(
    spark_session.createDataFrame(operations_pdf)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(CFG.fraud_operations_timeseries_table)
)
log("ops", f"{CFG.fraud_operations_timeseries_table}: {len(operations_pdf)} rows written")

spark_session.table(CFG.fraud_operations_timeseries_table).select(
    "period_start", "total_transactions", "flagged_transaction_count",
    "caught_fraud_count", "missed_fraud_count", "detected_fraud_amount",
    "operational_cost", "estimated_net_savings", "cumulative_net_savings",
    "capacity_utilisation",
).orderBy("period_start").show(24, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Hour-of-day profile — the shift-planning view
# MAGIC
# MAGIC Averaged across every replayed day. Fraud is not uniform across the clock,
# MAGIC so neither is the alert queue: this is the table that says whether a night
# MAGIC shift is needed, and it is a different question from "how much did we save".

# COMMAND ----------

profile_sdf = (
    spark_session.table(CFG.fraud_operations_timeseries_table)
    .groupBy("event_hour_of_day")
    .agg(
        F.count(F.lit(1)).alias("periods_observed"),
        F.avg("total_transactions").alias("avg_transactions"),
        F.avg("flagged_transaction_count").alias("avg_alerts"),
        F.max("flagged_transaction_count").alias("peak_alerts"),
        F.avg("caught_fraud_count").alias("avg_fraud_caught"),
        F.avg("detected_fraud_amount").alias("avg_fraud_amount_caught"),
        F.avg("operational_cost").alias("avg_operational_cost"),
        F.avg("estimated_net_savings").alias("avg_net_savings"),
        F.avg("capacity_utilisation").alias("avg_capacity_utilisation"),
        F.sum(F.when(F.col("capacity_exceeded"), 1).otherwise(0))
            .alias("periods_over_capacity"),
    )
    .withColumn("review_capacity",
                F.lit(CFG.ops_analyst_headcount * CFG.impact_analyst_daily_capacity
                      * PERIOD_FRACTION_OF_DAY))
    .withColumn("analyst_headcount", F.lit(CFG.ops_analyst_headcount))
    .withColumn("currency", F.lit(CFG.impact_currency))
    .withColumn("result_disclaimer", F.lit(impact.DISCLAIMER))
    .orderBy("event_hour_of_day")
)

(
    profile_sdf.write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(CFG.fraud_operations_profile_table)
)
log("ops", f"{CFG.fraud_operations_profile_table}: 24 hour-of-day rows written")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Since-go-live roll-up

# COMMAND ----------

OPS_SUMMARY = impact.operations_summary(operations_rows, ASSUMPTIONS, CONTEXT)

(
    spark_session.createDataFrame(pd.DataFrame([OPS_SUMMARY]))
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(CFG.fraud_operations_summary_table)
)
log("ops", f"{CFG.fraud_operations_summary_table}: 1 row written")

cur = CFG.impact_currency
print()
print("=" * 78)
print(f"OPERATIONAL REPLAY — {OPS_SUMMARY['periods_observed']} "
      f"{CFG.ops_time_grain}s at threshold {THRESHOLD:.4f}")
print("=" * 78)
print(f"  transactions processed   {OPS_SUMMARY['total_transactions']:>16,}")
print(f"  alerts raised            {OPS_SUMMARY['total_alerts']:>16,}")
print(f"  fraud caught             {OPS_SUMMARY['total_caught_fraud_count']:>16,}")
print(f"  fraud missed             {OPS_SUMMARY['total_missed_fraud_count']:>16,}")
print(f"  fraud value caught       {cur} {OPS_SUMMARY['total_detected_fraud_amount']:>12,.0f}")
print(f"  gross savings            {cur} {OPS_SUMMARY['total_gross_savings']:>12,.0f}")
print(f"  operational cost         {cur} {OPS_SUMMARY['total_operational_cost']:>12,.0f}")
print(f"  NET savings              {cur} {OPS_SUMMARY['total_net_savings']:>12,.0f}")
print(f"  savings per {cur} 1 spent  {OPS_SUMMARY['savings_to_cost_ratio']:>16,.1f}")
print(f"  periods over capacity    {OPS_SUMMARY['periods_over_capacity']:>16,} "
      f"({OPS_SUMMARY['periods_over_capacity_pct']:.1%})")
print(f"  peak backlog             {OPS_SUMMARY['peak_backlog_count']:>16,.0f} alerts")
print()
print(f"!! {OPS_SUMMARY['operating_note']}")
print(f"!! {impact.DISCLAIMER}")
print("=" * 78)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · Evidence

# COMMAND ----------

OPS_EVIDENCE = {
    **CONTEXT,
    "assumptions": ASSUMPTIONS.to_dict(),
    "analyst_headcount": CFG.ops_analyst_headcount,
    "replayed_rows": REPLAY_ROWS,
    "summary": OPS_SUMMARY,
    "timeseries_table": CFG.fraud_operations_timeseries_table,
    "profile_table": CFG.fraud_operations_profile_table,
    "summary_table": CFG.fraud_operations_summary_table,
    "generated_at": utc_now_iso(),
}

try:
    write_json_to_volume("fraud_operations_replay.json", OPS_EVIDENCE)
except Exception as exc:
    log("ops", f"could not write evidence to the volume: {type(exc).__name__}")

print(json.dumps(serving.sanitize_evidence(OPS_EVIDENCE), indent=2, default=str))
