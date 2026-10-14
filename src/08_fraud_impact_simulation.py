# Databricks notebook source
# MAGIC %md
# MAGIC # 08 · Fraud loss prevention simulator
# MAGIC
# MAGIC Notebooks 04 and 06 answer *is the model any good?* — PR-AUC, precision,
# MAGIC recall, PSI. This notebook answers the question the business sponsor asks
# MAGIC instead:
# MAGIC
# MAGIC > **If this model were used to block or review suspicious transactions, how
# MAGIC > much money could the business potentially save?**
# MAGIC
# MAGIC ## What this is, and what it is not
# MAGIC
# MAGIC Every figure produced here is a **counterfactual estimate**, built from three
# MAGIC ingredients:
# MAGIC
# MAGIC | Ingredient | Source | Certainty |
# MAGIC |---|---|---|
# MAGIC | which transactions were fraud | `label_true` in the scored table | historical fact (within the simulation) |
# MAGIC | which transactions the model would flag | `fraud_score` vs a threshold | model output, reproducible |
# MAGIC | what a stopped fraud is worth, what a review costs | `impact_*` assumptions in `config.py` | **assumed** — supplied by the business |
# MAGIC
# MAGIC The third ingredient is a guess. It is therefore written into **every output
# MAGIC row**, so no number can be read without the assumptions that produced it.
# MAGIC
# MAGIC > **The savings shown are simulated estimates based on historical transaction
# MAGIC > labels, model predictions, and configurable business assumptions. They
# MAGIC > should not be interpreted as guaranteed real-world savings without
# MAGIC > production validation.**
# MAGIC
# MAGIC ## Which rows are simulated
# MAGIC
# MAGIC The **test** split only, by default. It is the sole split that is both held
# MAGIC out from training *and* never used to pick the operating threshold, so it is
# MAGIC the only one where the savings figure is not quoting answers the model has
# MAGIC already seen. Change `impact_evaluation_split` to widen it — and say so in
# MAGIC any write-up if you do.
# MAGIC
# MAGIC ## Outputs
# MAGIC
# MAGIC | Table | Grain | Feeds |
# MAGIC |---|---|---|
# MAGIC | `gold_fraud_impact_simulation` | one row per threshold (0.05 … 0.95) | the trade-off, workload and quality charts |
# MAGIC | `gold_fraud_business_impact_summary` | one row | the executive KPI tiles and the recommendation |
# MAGIC
# MAGIC Both are tiny (tens of rows), so `sql/09_fraud_impact_dashboard.sql` can point
# MAGIC a Free Edition warehouse at them without the cost of the 6.3 M-row Gold table.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json

import pandas as pd
from pyspark.sql import functions as F

from fraud_common import impact

spark_session = get_spark()
SIMULATED_AT = datetime.now(timezone.utc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · COLUMN MAPPING — adjust here if you point this at another table
# MAGIC
# MAGIC The scored table this project writes uses `fraud_score` / `label_true` /
# MAGIC `amount`. A scoring job from somewhere else will not, so each role is
# MAGIC resolved against a list of common aliases (case-insensitively), and each
# MAGIC can be pinned outright by setting the matching override below.
# MAGIC
# MAGIC Set an override to a string to force it; leave it `None` to auto-detect.

# COMMAND ----------

# ---------------------------------------------------------------- #
#  COLUMN MAPPING OVERRIDES  — the only cell you should need to edit
#  when connecting this notebook to a different scoring table.
# ---------------------------------------------------------------- #
SCORE_COLUMN_OVERRIDE = None       # e.g. "prediction_probability"
LABEL_COLUMN_OVERRIDE = None       # e.g. "is_fraud"
AMOUNT_COLUMN_OVERRIDE = None      # e.g. "transaction_amount"
PREDICTED_LABEL_COLUMN_OVERRIDE = None   # only used when no probability exists

# Optional: read from a table other than the ones resolved in section 2.
SOURCE_TABLE_OVERRIDE = None       # e.g. "workspace.fraud_poc.my_scores"
# ---------------------------------------------------------------- #

print("auto-detect candidates")
print(f"  score           : {list(impact.SCORE_COLUMN_CANDIDATES)}")
print(f"  label           : {list(impact.LABEL_COLUMN_CANDIDATES)}")
print(f"  amount          : {list(impact.AMOUNT_COLUMN_CANDIDATES)}")
print(f"  predicted label : {list(impact.PREDICTED_LABEL_COLUMN_CANDIDATES)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Resolve the input table
# MAGIC
# MAGIC Preference order, most-specific first:
# MAGIC
# MAGIC 1. `SOURCE_TABLE_OVERRIDE`, when set
# MAGIC 2. `scored_predictions` — written by notebook 04, carries every split
# MAGIC 3. `inference_log` — written by notebook 06, monitoring period only
# MAGIC
# MAGIC If none exists the notebook stops with the **input contract** printed, so the
# MAGIC feature can be wired up later without guessing what it wants.

# COMMAND ----------

EXPECTED_INPUT_CONTRACT = {
    "table": CFG.scored_predictions_table,
    "written_by": "src/04_train_register_model.py",
    "required_columns": {
        "fraud_score": "float in [0, 1] — P(fraud) from the model",
        "label_true": "int 0/1 — the historical fraud outcome",
        "amount": "float — the transaction value, in `impact_currency`",
    },
    "optional_columns": {
        "split_label": "train/valid/test — lets the simulation stay on held-out rows",
        "model_version": "provenance, copied onto every output row",
        "scoring_path": "provenance: how the scores were produced",
    },
    "aliases_accepted": {
        "fraud_score": list(impact.SCORE_COLUMN_CANDIDATES),
        "label_true": list(impact.LABEL_COLUMN_CANDIDATES),
        "amount": list(impact.AMOUNT_COLUMN_CANDIDATES),
    },
}

CANDIDATE_TABLES = [t for t in (
    SOURCE_TABLE_OVERRIDE,
    CFG.scored_predictions_table,
    CFG.inference_log_table,
) if t]

SOURCE_TABLE = next(
    (name for name in CANDIDATE_TABLES if table_exists(name, spark_session)), None
)

if SOURCE_TABLE is None:
    raise RuntimeError(
        "no scored table found — nothing to simulate.\n"
        f"Looked for: {CANDIDATE_TABLES}\n"
        "Run notebook 04 (which writes the batch scores) first, or set "
        "SOURCE_TABLE_OVERRIDE in section 1 to your own scoring table.\n"
        "Expected input contract:\n"
        + json.dumps(EXPECTED_INPUT_CONTRACT, indent=2)
    )

source_sdf = spark_session.table(SOURCE_TABLE)
log("impact", f"input table: {SOURCE_TABLE} ({len(source_sdf.columns)} columns)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Resolve the columns
# MAGIC
# MAGIC A missing probability column is survivable but heavily degraded: without a
# MAGIC score there is no threshold to sweep, so the notebook falls back to a
# MAGIC **single point** built from the stored hard prediction, and says so.

# COMMAND ----------

AVAILABLE = source_sdf.columns

LABEL_COLUMN = impact.resolve_column(
    AVAILABLE, impact.LABEL_COLUMN_CANDIDATES, "label", LABEL_COLUMN_OVERRIDE
)
AMOUNT_COLUMN = impact.resolve_column(
    AVAILABLE, impact.AMOUNT_COLUMN_CANDIDATES, "amount", AMOUNT_COLUMN_OVERRIDE
)

try:
    SCORE_COLUMN = impact.resolve_column(
        AVAILABLE, impact.SCORE_COLUMN_CANDIDATES, "score", SCORE_COLUMN_OVERRIDE
    )
    SCORE_MODE = "probability"
    THRESHOLDS = impact.threshold_grid(
        CFG.impact_threshold_start, CFG.impact_threshold_stop, CFG.impact_threshold_step
    )
except KeyError as exc:
    # No probability anywhere. A stored 0/1 prediction still supports one
    # operating point -- treated as a score of exactly 0 or 1 and swept at 0.5.
    SCORE_COLUMN = impact.resolve_column(
        AVAILABLE, impact.PREDICTED_LABEL_COLUMN_CANDIDATES, "predicted label",
        PREDICTED_LABEL_COLUMN_OVERRIDE,
    )
    SCORE_MODE = "hard_label_only"
    THRESHOLDS = [0.5]
    log("impact", f"NO PROBABILITY COLUMN ({exc}). Falling back to the stored hard "
                  f"prediction {SCORE_COLUMN!r}: the threshold sweep collapses to a "
                  "single point and the trade-off charts will show one row.")

log("impact", f"columns -> score={SCORE_COLUMN!r} ({SCORE_MODE}), "
              f"label={LABEL_COLUMN!r}, amount={AMOUNT_COLUMN!r}")
log("impact", f"threshold grid: {len(THRESHOLDS)} points "
              f"[{THRESHOLDS[0]} … {THRESHOLDS[-1]}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Select the evaluation population
# MAGIC
# MAGIC Rows without a label cannot be scored for impact at all — a transaction whose
# MAGIC outcome is unknown has no fraud value to recover — so they are dropped and
# MAGIC counted rather than silently treated as legitimate. That is the same delayed-
# MAGIC label honesty notebook 06 enforces.

# COMMAND ----------

SPLIT_SELECTION = {
    "test": [splits.TEST],
    "valid": [splits.VALID],
    "holdout": [splits.VALID, splits.TEST],
    "all": None,
}[CFG.impact_evaluation_split]

evaluation_sdf = source_sdf
if SPLIT_SELECTION is not None and "split_label" in AVAILABLE:
    evaluation_sdf = evaluation_sdf.where(F.col("split_label").isin(SPLIT_SELECTION))
    log("impact", f"split filter: split_label IN {SPLIT_SELECTION}")
elif SPLIT_SELECTION is not None:
    log("impact", f"no `split_label` column in {SOURCE_TABLE}; simulating on every "
                  "row. Savings may be optimistic if training rows are included.")

# `label_available` exists on inference_log: unlabelled monitoring windows must
# not be counted as true negatives.
if "label_available" in AVAILABLE:
    evaluation_sdf = evaluation_sdf.where(F.col("label_available"))
    log("impact", "restricted to rows whose labels have arrived (label_available)")

evaluation_sdf = (
    evaluation_sdf
    .where(F.col(LABEL_COLUMN).isNotNull() & F.col(AMOUNT_COLUMN).isNotNull())
    .select(
        F.col(SCORE_COLUMN).cast("double").alias("fraud_score"),
        F.col(LABEL_COLUMN).cast("int").alias("label_true"),
        F.col(AMOUNT_COLUMN).cast("double").alias("amount"),
    )
)

EVALUATION_ROWS = evaluation_sdf.count()
if EVALUATION_ROWS == 0:
    raise RuntimeError(
        f"no labelled rows to simulate in {SOURCE_TABLE} for "
        f"impact_evaluation_split={CFG.impact_evaluation_split!r}. "
        "Check the split filter, or set impact_evaluation_split='all'."
    )
log("impact", f"evaluation population: {EVALUATION_ROWS:,} labelled rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Confusion counts for every threshold, in one pass
# MAGIC
# MAGIC One `agg` with a conditional sum per threshold rather than one query per
# MAGIC threshold: nineteen passes over the scored table would be nineteen times the
# MAGIC cost for identical arithmetic, and Free Edition serverless is not the place
# MAGIC to spend that. The flag rule is `score >= threshold`, matching
# MAGIC `metrics.confusion_at_threshold` and notebook 06's `predicted_label`.

# COMMAND ----------

def _threshold_aggregations(threshold: float):
    """Confusion counts and amount splits for one threshold, as agg expressions."""
    tag = f"t{int(round(threshold * 1000)):04d}"
    flagged = F.col("fraud_score") >= F.lit(float(threshold))
    fraud = F.col("label_true") == F.lit(1)
    zero = F.lit(0.0)
    return [
        F.sum(F.when(flagged, 1).otherwise(0)).alias(f"{tag}__flagged_transaction_count"),
        F.sum(F.when(flagged, F.col("amount")).otherwise(zero))
            .alias(f"{tag}__flagged_transaction_amount"),
        F.sum(F.when(flagged & fraud, 1).otherwise(0)).alias(f"{tag}__true_positive_count"),
        F.sum(F.when(flagged & ~fraud, 1).otherwise(0)).alias(f"{tag}__false_positive_count"),
        F.sum(F.when(~flagged & fraud, 1).otherwise(0)).alias(f"{tag}__false_negative_count"),
        F.sum(F.when(~flagged & ~fraud, 1).otherwise(0)).alias(f"{tag}__true_negative_count"),
        F.sum(F.when(flagged & fraud, F.col("amount")).otherwise(zero))
            .alias(f"{tag}__detected_fraud_amount"),
        F.sum(F.when(~flagged & fraud, F.col("amount")).otherwise(zero))
            .alias(f"{tag}__missed_fraud_amount"),
    ]


population_aggregations = [
    F.count(F.lit(1)).alias("total_transactions"),
    F.sum("amount").alias("total_transaction_amount"),
    F.sum(F.col("label_true")).alias("actual_fraud_count"),
    F.sum(F.when(F.col("label_true") == 1, F.col("amount")).otherwise(F.lit(0.0)))
        .alias("actual_fraud_amount"),
]

aggregated = evaluation_sdf.agg(
    *population_aggregations,
    *[expr for t in THRESHOLDS for expr in _threshold_aggregations(t)],
).collect()[0].asDict()

POPULATION = {
    field: float(aggregated[field] or 0.0) for field in impact.POPULATION_FIELDS
}
log("impact", f"population: {int(POPULATION['total_transactions']):,} transactions, "
              f"{int(POPULATION['actual_fraud_count']):,} fraudulent "
              f"({CFG.impact_currency} {POPULATION['actual_fraud_amount']:,.0f} at risk)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Apply the financial model
# MAGIC
# MAGIC The arithmetic lives in `fraud_common.impact.derive_impact_row`, which is unit
# MAGIC tested — the dashboard and the test suite cannot disagree about how a saving
# MAGIC was computed. See that module for what is deliberately *not* modelled.

# COMMAND ----------

ASSUMPTIONS = impact.ImpactAssumptions(
    prevention_rate=CFG.impact_prevention_rate,
    cost_per_review=CFG.impact_cost_per_review,
    customer_friction_cost=CFG.impact_customer_friction_cost,
    analyst_daily_capacity=CFG.impact_analyst_daily_capacity,
    currency=CFG.impact_currency,
).validate()

print("Business assumptions driving every figure below:")
for key, value in ASSUMPTIONS.to_dict().items():
    print(f"  {key:<24} = {value!r}")

# Provenance stamped onto every output row so a dashboard tile can never be
# mistaken for a measurement of a different model version.
def _first_value(column: str, default: str = "unknown") -> str:
    """First non-null value of a provenance column, or a stated default."""
    if column not in AVAILABLE:
        return default
    rows = source_sdf.select(column).where(F.col(column).isNotNull()).limit(1).collect()
    return str(rows[0][0]) if rows else default


MODEL_VERSION = _first_value("model_version")
SCORING_PATH = _first_value("scoring_path")

CONTEXT = {
    "source_table": SOURCE_TABLE,
    "evaluation_split": CFG.impact_evaluation_split,
    "score_column": SCORE_COLUMN,
    "score_mode": SCORE_MODE,
    "model_version": MODEL_VERSION,
    "scoring_path": SCORING_PATH,
    "currency": CFG.impact_currency,
    "simulated_at": SIMULATED_AT.isoformat(timespec="seconds"),
    "result_disclaimer": impact.DISCLAIMER,
}

simulation_rows = []
for threshold in THRESHOLDS:
    tag = f"t{int(round(threshold * 1000)):04d}"
    counts = {
        field: float(aggregated[f"{tag}__{field}"] or 0.0)
        for field in impact.COUNT_FIELDS
    }
    row = impact.derive_impact_row(threshold, counts, POPULATION, ASSUMPTIONS)
    row.update(CONTEXT)
    simulation_rows.append(row)

log("impact", f"simulated {len(simulation_rows)} thresholds")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Write `gold_fraud_impact_simulation`

# COMMAND ----------

# Via pandas rather than a list of dicts: pandas fixes the column order and the
# dtypes explicitly, where dict inference leaves both to sampling.
simulation_pdf = pd.DataFrame(simulation_rows).sort_values("threshold")
simulation_sdf = spark_session.createDataFrame(simulation_pdf)

(
    simulation_sdf.write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(CFG.fraud_impact_simulation_table)
)
log("impact", f"{CFG.fraud_impact_simulation_table}: {len(simulation_rows)} rows written")

display_columns = [
    "threshold", "flagged_transaction_count", "detected_fraud_amount",
    "missed_fraud_amount", "gross_estimated_savings", "review_cost",
    "false_positive_friction_cost", "estimated_net_savings",
    "precision", "recall", "estimated_review_days_required",
]
spark_session.table(CFG.fraud_impact_simulation_table) \
    .select(*display_columns).orderBy("threshold").show(len(simulation_rows), truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · Recommend an operating threshold
# MAGIC
# MAGIC The recommendation is simply the threshold with the highest
# MAGIC `estimated_net_savings`. That is a **starting point for a conversation**, not
# MAGIC a decision: net savings is one scalar built on assumed costs, and a real
# MAGIC operating point also has to fit the review capacity that actually exists and
# MAGIC the organisation's tolerance for stopping good customers. Both caveats are
# MAGIC written into the summary row as `business_override_note`.

# COMMAND ----------

SUMMARY = impact.build_summary(simulation_rows, ASSUMPTIONS, CONTEXT)

summary_sdf = spark_session.createDataFrame(pd.DataFrame([SUMMARY]))
(
    summary_sdf.write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(CFG.fraud_impact_summary_table)
)
log("impact", f"{CFG.fraud_impact_summary_table}: 1 row written")

print()
print("=" * 78)
print("BUSINESS RECOMMENDATION")
print("=" * 78)
print(SUMMARY["recommendation_note"])
print()
print(SUMMARY["business_override_note"])
print()
print(f"!! {impact.DISCLAIMER}")
print("=" * 78)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 · Evidence
# MAGIC
# MAGIC Same pattern as every other notebook: a sanitised JSON copy in the volume, so
# MAGIC the run can be cited without re-querying the tables.

# COMMAND ----------

IMPACT_SUMMARY = {
    **CONTEXT,
    "assumptions": ASSUMPTIONS.to_dict(),
    "evaluation_rows": EVALUATION_ROWS,
    "population": POPULATION,
    "thresholds_simulated": THRESHOLDS,
    "recommended": {k: v for k, v in SUMMARY.items() if not k.startswith("assumption_")},
    "simulation_table": CFG.fraud_impact_simulation_table,
    "summary_table": CFG.fraud_impact_summary_table,
    "generated_at": utc_now_iso(),
}

try:
    write_json_to_volume("fraud_impact_simulation.json", IMPACT_SUMMARY)
except Exception as exc:
    log("impact", f"could not write evidence to the volume: {type(exc).__name__}")

print(json.dumps(serving.sanitize_evidence(IMPACT_SUMMARY), indent=2, default=str))
