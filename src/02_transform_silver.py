# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver — validation, deduplication, conformed columns
# MAGIC
# MAGIC Bronze is raw. Silver is *trusted*: every row here has passed the quality
# MAGIC rules declared in `fraud_common.schema.QUALITY_RULES`, and every row that
# MAGIC did not is preserved in `silver_rejected_records` with the reason attached.
# MAGIC Nothing is silently discarded.
# MAGIC
# MAGIC Three things happen:
# MAGIC
# MAGIC 1. **Conform** — source columns renamed to snake_case, types asserted, time
# MAGIC    fields derived from `step`.
# MAGIC 2. **Validate** — each rule is a SQL predicate that must be TRUE for a valid
# MAGIC    row. `reject` failures quarantine the row; `warn` failures are counted.
# MAGIC 3. **Deduplicate** — exact duplicates are identified deterministically and
# MAGIC    *flagged*, not deleted. Gold filters on the flag, so the count remains
# MAGIC    auditable.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

spark_session = get_spark()
log("silver", f"reading {CFG.bronze_table}")
bronze = spark_session.table(CFG.bronze_table)
bronze_count = bronze.count()
log("silver", f"bronze rows: {bronze_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Conform
# MAGIC
# MAGIC `step` is an hour counter with no wall-clock anchor. Silver attaches a
# MAGIC synthetic timestamp (`SIMULATION_START_TS`) purely so the dashboard can plot
# MAGIC a real time axis — it is a presentation device, not a claim about real dates.

# COMMAND ----------

rename_exprs = [
    f"`{source}` AS {target}"
    for source, target in features.SOURCE_TO_SILVER.items()
]
lineage_cols = ["ingestion_timestamp", "source_file", "batch_id"]

conformed = bronze.selectExpr(*rename_exprs, *lineage_cols)

conformed = (
    conformed
    .withColumn("event_hour_of_day", F.expr("CAST((step - 1) % 24 AS INT)"))
    .withColumn("event_day_index", F.expr("CAST((step - 1) DIV 24 + 1 AS INT)"))
    .withColumn("event_day_of_week", F.expr("CAST(((step - 1) DIV 24) % 7 AS INT)"))
    .withColumn(
        "event_timestamp",
        F.expr(
            f"TIMESTAMP('{schema.SIMULATION_START_TS}') + MAKE_INTERVAL(0, 0, 0, 0, step - 1, 0, 0)"
        ),
    )
    .withColumn("event_date", F.to_date("event_timestamp"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Validate
# MAGIC
# MAGIC Each rule becomes a boolean column so per-rule failure counts can be
# MAGIC persisted, then the reject rules are combined into a single verdict.

# COMMAND ----------

rule_columns = {}
for rule in schema.QUALITY_RULES:
    column_name = f"rule_{rule.name}"
    rule_columns[rule.name] = column_name
    # COALESCE: a NULL comparison is a failure, not an unknown we can ignore.
    conformed = conformed.withColumn(
        column_name, F.coalesce(F.expr(rule.expression), F.lit(False))
    )

reject_rule_names = [r.name for r in schema.reject_rules()]
warn_rule_names = [r.name for r in schema.warn_rules()]

is_valid = F.lit(True)
for name in reject_rule_names:
    is_valid = is_valid & F.col(rule_columns[name])

# Human-readable reason list for the quarantine table.
reject_reason = F.concat_ws(
    ",",
    *[
        F.when(~F.col(rule_columns[name]), F.lit(name)).otherwise(F.lit(None))
        for name in reject_rule_names
    ],
)
warn_reason = F.concat_ws(
    ",",
    *[
        F.when(~F.col(rule_columns[name]), F.lit(name)).otherwise(F.lit(None))
        for name in warn_rule_names
    ],
)

validated = (
    conformed
    .withColumn("is_valid", is_valid)
    .withColumn("reject_reasons", reject_reason)
    .withColumn("warn_reasons", warn_reason)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Deterministic exact-duplicate detection
# MAGIC
# MAGIC An exact duplicate is a row identical across all 11 source columns. Because
# MAGIC the rows are byte-identical, *which* copy is kept is immaterial — what
# MAGIC matters is that the choice is stable across runs. `row_number()` over the
# MAGIC full natural key gives each copy an `occurrence_index`; index 0 is the
# MAGIC survivor, and the surrogate key folds the index in so ids stay unique.

# COMMAND ----------

natural_key_columns = [features.SOURCE_TO_SILVER[c] for c in schema.SOURCE_COLUMNS]

dedup_window = Window.partitionBy(*natural_key_columns).orderBy(
    F.col("ingestion_timestamp").asc(), F.col("batch_id").asc()
)

deduped = (
    validated
    .withColumn("occurrence_index", F.row_number().over(dedup_window) - 1)
    .withColumn("is_exact_duplicate", F.col("occurrence_index") > 0)
    .withColumn(
        "transaction_id",
        F.sha2(
            F.concat_ws("|", *natural_key_columns, F.col("occurrence_index").cast("string")),
            256,
        ),
    )
)

silver = deduped.select(
    "transaction_id",
    "step",
    "type",
    "amount",
    "name_orig",
    "old_balance_orig",
    "new_balance_orig",
    "name_dest",
    "old_balance_dest",
    "new_balance_dest",
    "is_fraud",
    "is_flagged_fraud",
    "event_timestamp",
    "event_date",
    "event_hour_of_day",
    "event_day_index",
    "event_day_of_week",
    "occurrence_index",
    "is_exact_duplicate",
    "is_valid",
    "reject_reasons",
    "warn_reasons",
    "ingestion_timestamp",
    "source_file",
    "batch_id",
)
# No .cache()/.persist() anywhere in this project: Databricks serverless (the
# only compute Free Edition offers) rejects them with
# [NOT_SUPPORTED_WITH_SERVERLESS]. Counts are read back from the written Delta
# tables instead, which is cheaper than caching anyway.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Silver, rejects and the quality report

# COMMAND ----------

valid_rows = silver.where("is_valid")
rejected_rows = silver.where("NOT is_valid")

(
    valid_rows.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CFG.silver_table)
)
(
    rejected_rows.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CFG.silver_rejects_table)
)

# Count from the persisted tables rather than re-evaluating the lazy plan.
silver_table = spark_session.table(CFG.silver_table)
rejects_table = spark_session.table(CFG.silver_rejects_table)

silver_count = silver_table.count()
rejected_count = rejects_table.count()
duplicate_count = (
    silver_table.where("is_exact_duplicate").count()
    + rejects_table.where("is_exact_duplicate").count()
)
log("silver", f"valid={silver_count:,} rejected={rejected_count:,} "
              f"exact_duplicates={duplicate_count:,}")

# COMMAND ----------

# Per-rule failure counts come from `validated`, which still carries the
# individual rule_* boolean columns (`silver` drops them to stay narrow).
failure_counts = validated.select([
    F.sum((~F.col(rule_columns[r.name])).cast("long")).alias(r.name)
    for r in schema.QUALITY_RULES
]).collect()[0].asDict()

quality_rows = [
    {
        "evaluated_at": datetime.now(timezone.utc),
        "rule_name": rule.name,
        "severity": rule.severity,
        "expression": rule.expression,
        "description": rule.description,
        "failed_row_count": int(failure_counts.get(rule.name) or 0),
        "evaluated_row_count": int(bronze_count),
        "failure_rate": float(failure_counts.get(rule.name) or 0) / max(bronze_count, 1),
    }
    for rule in schema.QUALITY_RULES
]

# Reconciliation is itself a quality record: bronze must equal silver + rejects.
quality_rows.append({
    "evaluated_at": datetime.now(timezone.utc),
    "rule_name": "_reconciliation_bronze_equals_silver_plus_rejects",
    "severity": "reject",
    "expression": "bronze_count = silver_count + rejected_count",
    "description": "No row may be lost between Bronze and Silver.",
    "failed_row_count": int(abs(bronze_count - (silver_count + rejected_count))),
    "evaluated_row_count": int(bronze_count),
    "failure_rate": 0.0,
})

(
    spark_session.createDataFrame(quality_rows)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(CFG.silver_dq_table)
)

for row in quality_rows:
    if row["failed_row_count"]:
        log("quality", f"{row['severity']:<7} {row['rule_name']}: "
                       f"{row['failed_row_count']:,} failures "
                       f"({row['failure_rate']:.4%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reconciliation assertions

# COMMAND ----------

assert bronze_count == silver_count + rejected_count, (
    f"row loss between Bronze and Silver: bronze={bronze_count:,} "
    f"silver={silver_count:,} rejected={rejected_count:,}"
)
assert silver_count > 0, "Silver is empty — every row was rejected. Inspect " \
                         f"{CFG.silver_rejects_table}."

# These read the persisted table, not the lazy plan -- re-evaluating
# `valid_rows` would rescan the whole source once per assertion.
distinct_targets = sorted(
    r["is_fraud"] for r in silver_table.select("is_fraud").distinct().collect()
)
assert set(distinct_targets) <= set(schema.ALLOWED_TARGET_VALUES), (
    f"target column contains values outside {schema.ALLOWED_TARGET_VALUES}: "
    f"{distinct_targets}"
)

distinct_ids = silver_table.select("transaction_id").distinct().count()
assert distinct_ids == silver_count, (
    f"transaction_id is not unique: {distinct_ids:,} distinct ids for "
    f"{silver_count:,} rows"
)

fraud_count = silver_table.where("is_fraud = 1").count()
log("silver", f"fraud prevalence: {fraud_count:,} / {silver_count:,} "
              f"= {fraud_count / max(silver_count, 1):.4%}")
log("silver", f"DONE — {CFG.silver_table}")
