# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold — training-ready features and the online contract
# MAGIC
# MAGIC ## The leakage question, decided explicitly
# MAGIC
# MAGIC PaySim is famous for making models look brilliant. `newbalanceOrig` alone
# MAGIC gets you near-perfect separation, because for a fraudulent transfer the
# MAGIC balance goes exactly to zero. That signal only exists **after** the payment
# MAGIC settles. A real-time system must decide *before* it settles.
# MAGIC
# MAGIC This notebook therefore scores under a strict **pre-transaction**
# MAGIC interpretation, and every column is classified by when its value becomes
# MAGIC known (`fraud_common.schema.GOLD_COLUMNS`):
# MAGIC
# MAGIC | Excluded | Class | Reason |
# MAGIC |---|---|---|
# MAGIC | `newbalanceOrig`, `newbalanceDest` | post-transaction | Settled balances. Unknowable at authorisation time. |
# MAGIC | `isFlaggedFraud` | label-adjacent | The output of PaySim's own rules engine — a downstream control's decision, not an input. |
# MAGIC | `isFraud` | label | The target. |
# MAGIC | `nameOrig`, `nameDest` | identifier | Raw customer keys. Hashed for joins; never sent to the endpoint. |
# MAGIC | `step` | time axis | Governs the split and the monitoring windows. Withheld so the model cannot memorise simulator time. |
# MAGIC
# MAGIC `dest_is_merchant` is kept: it is derived from `nameDest`, but it encodes an
# MAGIC *account type* (merchant vs consumer) that any payment processor knows
# MAGIC before authorising, and the raw identifier itself never leaves the lakehouse.
# MAGIC
# MAGIC The consequence is honest: metrics here are much lower than the ~0.99 PR-AUC
# MAGIC that circulates in PaySim notebooks. That gap **is** the leakage.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql import functions as F

spark_session = get_spark()

# Runtime guard: the served feature list must contain no forbidden column.
served = schema.served_features()
schema.assert_no_leakage(served)
features.assert_feature_contract_consistent()
log("gold", f"{len(served)} served features, leakage check passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the feature table
# MAGIC
# MAGIC Only valid, non-duplicate Silver rows are promoted. Every expression comes
# MAGIC from `fraud_common.features.FEATURE_SQL`, the same declaration the pandas
# MAGIC path and the serving payload builder use — training and serving cannot
# MAGIC diverge without a test failing.

# COMMAND ----------

silver = spark_session.table(CFG.silver_table).where(
    "is_valid AND NOT is_exact_duplicate"
)
silver_count = silver.count()
log("gold", f"eligible silver rows: {silver_count:,}")

gold = silver.selectExpr(*features.gold_select_expressions())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chronological split labels
# MAGIC
# MAGIC Assigned here, once, so training, monitoring and the dashboard all agree on
# MAGIC which period a row belongs to. Cut on the `step` **range** rather than row
# MAGIC count: PaySim's hourly volume is very uneven, and a row-count cut would put
# MAGIC a boundary mid-hour, placing the same hour on both sides of it.

# COMMAND ----------

step_bounds = gold.select(
    F.min("step").alias("min_step"), F.max("step").alias("max_step")
).collect()[0]
MIN_STEP, MAX_STEP = int(step_bounds["min_step"]), int(step_bounds["max_step"])

BOUNDARIES = splits.compute_boundaries(
    MIN_STEP, MAX_STEP, CFG.train_step_fraction, CFG.valid_step_fraction
)
log("gold", f"split boundaries: {BOUNDARIES.as_dict()}")

gold = gold.withColumn("split_label", F.expr(BOUNDARIES.sql_expression("step")))

(
    gold.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("split_label")
    .saveAsTable(CFG.gold_features_table)
)
log("gold", f"wrote {CFG.gold_features_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persist the feature contract
# MAGIC
# MAGIC The contract is a table, not just a docstring: the dashboard and the model
# MAGIC card read it, and a reviewer can query *why* a column was excluded.

# COMMAND ----------

contract_rows = [
    {
        "feature_name": spec.name,
        "dtype": spec.dtype,
        "availability": spec.availability,
        "served_to_endpoint": bool(spec.served),
        "description": spec.description,
        "derivation": spec.derivation or "",
        "sql_expression": features.FEATURE_SQL.get(
            spec.name, features.PASSTHROUGH_SQL.get(spec.name, "")
        ),
        "signature_position": (
            features.signature_input_columns().index(spec.name)
            if spec.served else -1
        ),
        "recorded_at": datetime.now(timezone.utc),
    }
    for spec in schema.GOLD_COLUMNS
]

(
    spark_session.createDataFrame(contract_rows)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(CFG.feature_contract_table)
)
log("gold", f"wrote {CFG.feature_contract_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification

# COMMAND ----------

gold_table = spark_session.table(CFG.gold_features_table)
gold_count = gold_table.count()

assert gold_count == silver_count, (
    f"Gold row count {gold_count:,} != eligible Silver {silver_count:,}"
)

gold_columns = set(gold_table.columns)
assert set(served) <= gold_columns, (
    f"served features missing from Gold: {sorted(set(served) - gold_columns)}"
)
schema.assert_no_leakage(served)

# Split integrity: the step ranges must not overlap.
split_profile = (
    gold_table.groupBy("split_label")
    .agg(
        F.count("*").alias("rows"),
        F.min("step").alias("min_step"),
        F.max("step").alias("max_step"),
        F.sum("is_fraud").alias("fraud_rows"),
    )
    .collect()
)
profile_by_split = {r["split_label"]: r.asDict() for r in split_profile}

for name in splits.SPLIT_ORDER:
    assert name in profile_by_split, f"split {name!r} is empty"

ordered = [profile_by_split[n] for n in splits.SPLIT_ORDER]
for earlier, later in zip(ordered, ordered[1:]):
    assert earlier["max_step"] < later["min_step"], (
        "chronological violation between splits: "
        f"{earlier['max_step']} >= {later['min_step']}"
    )

for name in splits.SPLIT_ORDER:
    row = profile_by_split[name]
    prevalence = (row["fraud_rows"] or 0) / max(row["rows"], 1)
    log("gold", f"{name:<6} rows={row['rows']:>10,} "
                f"steps=[{row['min_step']}, {row['max_step']}] "
                f"fraud={row['fraud_rows']:>6,} ({prevalence:.4%})")

log("gold", f"DONE — {CFG.gold_features_table}: {gold_count:,} rows, "
            f"{len(served)} served features")
