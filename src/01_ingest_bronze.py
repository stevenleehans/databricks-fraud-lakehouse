# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze — raw ingestion
# MAGIC
# MAGIC Reads the manually-uploaded PaySim CSV with an **explicit schema** and lands
# MAGIC it as an append-oriented Delta table with ingestion lineage attached.
# MAGIC
# MAGIC Design decisions worth knowing:
# MAGIC
# MAGIC | Decision | Why |
# MAGIC |---|---|
# MAGIC | Explicit schema, never `inferSchema` | Inference costs a full extra pass over 6.3 M rows and can silently retype a column between runs. |
# MAGIC | `PERMISSIVE` mode + `_corrupt_record` | Malformed rows are captured in `bronze_malformed_records` instead of being dropped without trace. |
# MAGIC | File-metadata-derived `batch_id` | Re-running the same file produces the same id, so the append can be skipped — idempotence without a full-table dedupe. |
# MAGIC | Bronze stays raw | No renaming, no filtering, no type coercion beyond the declared schema. All judgement happens in Silver, where it is auditable. |

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import hashlib

from pyspark.sql import functions as F

spark_session = get_spark()
ensure_uc_objects(spark_session)
source_path = require_input_file()
log("bronze", f"source: {source_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch identity
# MAGIC
# MAGIC `batch_id` hashes the file's path, size and modification time. Same file →
# MAGIC same id → the run is a no-op. A re-upload (new mtime or size) produces a new
# MAGIC id and appends as a genuinely new batch.

# COMMAND ----------

def describe_source_file(path: str) -> dict:
    """Path, size and mtime of the input file, via dbutils or os.stat."""
    db = _dbutils()
    if db is not None:
        try:
            entries = db.fs.ls(path)
            entry = entries[0]
            return {
                "path": path,
                "size_bytes": int(entry.size),
                "modified_ms": int(getattr(entry, "modificationTime", 0) or 0),
            }
        except Exception as exc:
            log("bronze", f"dbutils.fs.ls failed ({type(exc).__name__}); using os.stat")
    stat = os.stat(path)
    return {
        "path": path,
        "size_bytes": int(stat.st_size),
        "modified_ms": int(stat.st_mtime * 1000),
    }


def compute_batch_id(file_info: dict) -> str:
    key = f"{file_info['path']}|{file_info['size_bytes']}|{file_info['modified_ms']}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


SOURCE_INFO = describe_source_file(source_path)
BATCH_ID = compute_batch_id(SOURCE_INFO)
log("bronze", f"batch_id={BATCH_ID} from {SOURCE_INFO}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Idempotence gate

# COMMAND ----------

already_ingested = False
if table_exists(CFG.bronze_table, spark_session):
    existing = (
        spark_session.table(CFG.bronze_table)
        .where(F.col("batch_id") == F.lit(BATCH_ID))
        .limit(1)
        .count()
    )
    already_ingested = existing > 0

if already_ingested and not CFG.overwrite_existing_batch:
    log("bronze", f"batch {BATCH_ID} already present — skipping ingest (idempotent).")
    log("bronze", "set overwrite_existing_batch=True in config to force a re-ingest.")
elif already_ingested and CFG.overwrite_existing_batch:
    log("bronze", f"overwrite requested — deleting batch {BATCH_ID} before re-ingest.")
    spark_session.sql(
        f"DELETE FROM {CFG.bronze_table} WHERE batch_id = '{BATCH_ID}'"
    )
    already_ingested = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read with the explicit contract

# COMMAND ----------

CORRUPT_COLUMN = "_corrupt_record"
READ_SCHEMA = f"{schema.SOURCE_SCHEMA_DDL}, {CORRUPT_COLUMN} STRING"

raw = (
    spark_session.read
    .option("header", "true")
    .option("mode", "PERMISSIVE")
    .option("columnNameOfCorruptRecord", CORRUPT_COLUMN)
    .schema(READ_SCHEMA)
    .csv(source_path)
)

header_columns = [c for c in raw.columns if c != CORRUPT_COLUMN]
missing_columns = [c for c in schema.SOURCE_COLUMNS if c not in header_columns]
if missing_columns:
    raise ValueError(
        f"source schema mismatch: {missing_columns} absent. "
        f"Expected exactly {schema.SOURCE_COLUMNS}. "
        "Confirm you uploaded the Kaggle PaySim1 CSV and not a variant."
    )
log("bronze", f"source columns verified: {header_columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Attach lineage and split off malformed rows

# COMMAND ----------

annotated = (
    raw
    .withColumn("ingestion_timestamp", F.current_timestamp())
    # `_metadata.file_path` rather than `input_file_name()`: the latter is
    # unsupported on serverless / Spark Connect, which is the only compute
    # Free Edition offers.
    .withColumn("source_file", F.col("_metadata.file_path"))
    .withColumn("batch_id", F.lit(BATCH_ID))
    .withColumn("source_size_bytes", F.lit(SOURCE_INFO["size_bytes"]))
)

# A row is malformed if the CSV parser could not honour the declared schema.
is_malformed = F.col(CORRUPT_COLUMN).isNotNull()

well_formed = annotated.where(~is_malformed).drop(CORRUPT_COLUMN)
malformed = annotated.where(is_malformed).select(
    CORRUPT_COLUMN, "ingestion_timestamp", "source_file", "batch_id"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Bronze (append) and quarantine malformed rows

# COMMAND ----------

bronze_rows = 0
malformed_rows = 0

if not already_ingested:
    well_formed.write.mode("append").option("mergeSchema", "false") \
        .saveAsTable(CFG.bronze_table)
    malformed.write.mode("append").option("mergeSchema", "false") \
        .saveAsTable(CFG.bronze_malformed_table)
    log("bronze", f"appended batch {BATCH_ID}")

batch_frame = spark_session.table(CFG.bronze_table).where(F.col("batch_id") == BATCH_ID)
bronze_rows = batch_frame.count()
if table_exists(CFG.bronze_malformed_table, spark_session):
    malformed_rows = (
        spark_session.table(CFG.bronze_malformed_table)
        .where(F.col("batch_id") == BATCH_ID)
        .count()
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest audit
# MAGIC
# MAGIC Row counts and null checks are persisted, not just printed — Silver
# MAGIC reconciles against this table and the dashboard reads it.

# COMMAND ----------

null_checks = batch_frame.select([
    F.sum(F.col(c).isNull().cast("int")).alias(f"null_{c}")
    for c in schema.SOURCE_COLUMNS
]).collect()[0].asDict()

profile = batch_frame.select(
    F.min("step").alias("min_step"),
    F.max("step").alias("max_step"),
    F.countDistinct("type").alias("distinct_types"),
    F.sum(F.col("isFraud")).alias("fraud_rows"),
    F.sum(F.col("isFlaggedFraud")).alias("flagged_rows"),
    F.min("amount").alias("min_amount"),
    F.max("amount").alias("max_amount"),
).collect()[0].asDict()

observed_types = sorted(
    r["type"] for r in batch_frame.select("type").distinct().collect()
    if r["type"] is not None
)
unexpected_types = sorted(set(observed_types) - set(schema.ALLOWED_TRANSACTION_TYPES))

audit_row = {
    "batch_id": BATCH_ID,
    "ingested_at": datetime.now(timezone.utc),
    "source_path": SOURCE_INFO["path"],
    "source_size_bytes": SOURCE_INFO["size_bytes"],
    "source_modified_ms": SOURCE_INFO["modified_ms"],
    "bronze_row_count": int(bronze_rows),
    "malformed_row_count": int(malformed_rows),
    "total_null_cells": int(sum(int(v or 0) for v in null_checks.values())),
    "min_step": int(profile["min_step"]) if profile["min_step"] is not None else None,
    "max_step": int(profile["max_step"]) if profile["max_step"] is not None else None,
    "fraud_row_count": int(profile["fraud_rows"] or 0),
    "flagged_row_count": int(profile["flagged_rows"] or 0),
    "min_amount": float(profile["min_amount"] or 0.0),
    "max_amount": float(profile["max_amount"] or 0.0),
    "observed_types": ",".join(observed_types),
    "unexpected_types": ",".join(unexpected_types),
    "expected_row_count": schema.EXPECTED_SOURCE_ROWS,
    "row_count_within_tolerance": abs(bronze_rows - schema.EXPECTED_SOURCE_ROWS)
    <= 0.01 * schema.EXPECTED_SOURCE_ROWS,
    "skipped_as_duplicate_batch": bool(already_ingested),
}

audit_df = spark_session.createDataFrame([audit_row])
audit_df.write.mode("append").saveAsTable(CFG.bronze_audit_table)

for key, value in audit_row.items():
    log("audit", f"{key} = {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assertions
# MAGIC
# MAGIC These fail the notebook rather than letting a bad batch flow to Silver.

# COMMAND ----------

assert bronze_rows > 0, "Bronze batch is empty — check the CSV path and header row."

if malformed_rows:
    log("bronze", f"WARNING: {malformed_rows} malformed rows quarantined in "
                  f"{CFG.bronze_malformed_table}")

if unexpected_types:
    log("bronze", f"WARNING: transaction types outside the contract: {unexpected_types}. "
                  "Silver will quarantine these rows.")

if not audit_row["row_count_within_tolerance"]:
    log("bronze", f"WARNING: {bronze_rows:,} rows differs from the published "
                  f"{schema.EXPECTED_SOURCE_ROWS:,} by more than 1%. This is fine if "
                  "you deliberately used a subset — note it in the results.")

log("bronze", f"DONE — {CFG.bronze_table} batch {BATCH_ID}: {bronze_rows:,} rows")
