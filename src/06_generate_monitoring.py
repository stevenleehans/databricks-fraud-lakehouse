# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Inference records and drift monitoring
# MAGIC
# MAGIC Scores the post-training period in batch, then computes three **separate**
# MAGIC families of statistics. Keeping them separate is the point of this notebook:
# MAGIC
# MAGIC | Family | Question it answers | Needs labels? | What it does *not* prove |
# MAGIC |---|---|---|---|
# MAGIC | **Feature drift** (`monitoring_feature_drift`) | Did the input distribution move? | No | That the model got worse. A marketing campaign shifts `amount` without any change in fraud behaviour. |
# MAGIC | **Prediction drift** (`monitoring_prediction_drift`) | Did the score distribution or alert rate move? | No | That the model got worse. It is a *symptom* worth alerting on because it is observable immediately. |
# MAGIC | **Performance** (`monitoring_performance`) | Did PR-AUC / precision / recall actually fall? | **Yes** | Nothing — but it is only computable once labels arrive. |
# MAGIC
# MAGIC ## Delayed labels
# MAGIC Fraud labels arrive days-to-weeks late (chargebacks, investigations, customer
# MAGIC reports). The last `label_delay_windows` windows are therefore marked
# MAGIC `label_available = false` and get **no** performance row, so the dashboard is
# MAGIC forced to show "performance unknown" for the recent past — which is exactly
# MAGIC the situation a real fraud team operates in. The ground truth exists in the
# MAGIC simulation; withholding it is deliberate.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
# (no Spark UDF here any more -- scores are read from a table)

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
spark_session = get_spark()
COMPUTED_AT = datetime.now(timezone.utc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Load the batch scores written by notebook 04
# MAGIC
# MAGIC This notebook deliberately does **not** load the model.
# MAGIC
# MAGIC On Databricks Free Edition the notebook runtime ships an MLflow **2.x**
# MAGIC client against an MLflow **3** tracking server, and that client cannot read
# MAGIC the model back by any route: `models:/<uc_name>@alias` fails downloading
# MAGIC from Unity Catalog storage (S3 HeadObject 400), `models:/m-<id>` is rejected
# MAGIC as a malformed URI, `runs:/<id>/model` 404s because MLflow 3 writes a
# MAGIC different layout, and the raw DBFS artefact path is misparsed as a run id.
# MAGIC Logging and registering succeed; reading back does not.
# MAGIC
# MAGIC Notebook 04 therefore scores while the pipeline is still in memory and
# MAGIC writes `scored_predictions`. Every score below carries a `scoring_path` of
# MAGIC `training_notebook_in_memory_pipeline`, which is an honest statement that
# MAGIC these numbers do **not** demonstrate the registered Unity Catalog artefact
# MAGIC is loadable or servable.

# COMMAND ----------

client = MlflowClient()

if not spark_session.catalog.tableExists(CFG.scored_predictions_table):
    raise RuntimeError(
        f"{CFG.scored_predictions_table} does not exist. Run notebook 04 first "
        "-- it writes the batch scores this notebook consumes."
    )

scored_all = spark_session.table(CFG.scored_predictions_table)
scored_count = scored_all.count()
if scored_count == 0:
    raise RuntimeError(f"{CFG.scored_predictions_table} is empty.")

_meta = scored_all.select(
    "model_name", "model_version", "scoring_path", "scoring_threshold"
).limit(1).collect()[0]
MODEL_VERSION = str(_meta["model_version"])
SCORING_PATH = str(_meta["scoring_path"])
MODEL_URI = f"{_meta['model_name']} v{MODEL_VERSION}"
SCORING_THRESHOLD = float(_meta["scoring_threshold"])

log("monitor", f"scores: {scored_count:,} rows from {MODEL_URI} "
               f"(path={SCORING_PATH}, threshold={SCORING_THRESHOLD:.6f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Reference period vs monitoring period
# MAGIC
# MAGIC The reference is the **training** period; monitoring covers everything after
# MAGIC it. Reference statistics are computed once and frozen — recomputing bin
# MAGIC edges per window would compare each window against itself and report zero
# MAGIC drift forever.

# COMMAND ----------

SERVED = schema.served_features()
NUMERIC = schema.numeric_features()
CATEGORICAL = schema.categorical_features()

gold = spark_session.table(CFG.gold_features_table)
reference_sdf = gold.where(F.col("split_label") == splits.TRAIN)
monitoring_sdf = gold.where(F.col("split_label") != splits.TRAIN)

mon_bounds = monitoring_sdf.select(
    F.min("step").alias("min_step"), F.max("step").alias("max_step")
).collect()[0]
MON_MIN, MON_MAX = int(mon_bounds["min_step"]), int(mon_bounds["max_step"])

WINDOWS = drift.make_windows(MON_MIN, MON_MAX, CFG.monitoring_window_count)
LABELLED_WINDOW_COUNT = max(len(WINDOWS) - CFG.label_delay_windows, 0)
for w in WINDOWS:
    w["label_available"] = w["window_index"] < LABELLED_WINDOW_COUNT
log("monitor", f"monitoring steps [{MON_MIN}, {MON_MAX}] -> {len(WINDOWS)} windows; "
               f"labels withheld for the last {CFG.label_delay_windows}")

window_case = "CASE " + " ".join(
    f"WHEN step BETWEEN {w['start_step']} AND {w['end_step']} THEN '{w['window_id']}'"
    for w in WINDOWS
) + " ELSE 'unassigned' END"

monitoring_sdf = monitoring_sdf.withColumn("window_id", F.expr(window_case))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Reference statistics
# MAGIC
# MAGIC Bin edges come from a deterministic sample of the training period rather
# MAGIC than the full 4.4 M rows — quantile edges are stable well before that, and
# MAGIC the sample keeps the driver comfortable on Free Edition.

# COMMAND ----------

REFERENCE_SAMPLE_ROWS = 200_000
reference_total = reference_sdf.count()
ref_fraction = min(1.0, REFERENCE_SAMPLE_ROWS / max(reference_total, 1))
ref_predicate = splits.sample_sql_expression(
    "transaction_id", CFG.random_seed + 2, ref_fraction, keep_all_when=None
)
# De-duplicated: `amount` is already a served feature, and selecting it twice
# yields a pandas frame with a repeated column name, where df["amount"] returns
# a DataFrame rather than a Series and every downstream .tolist() breaks.
REFERENCE_COLUMNS = list(dict.fromkeys([*SERVED, "is_fraud"]))
reference_pdf = (
    reference_sdf.where(F.expr(ref_predicate))
    .select(*REFERENCE_COLUMNS)
    .toPandas()
)
assert reference_pdf.columns.is_unique, (
    f"duplicate columns in the reference sample: "
    f"{reference_pdf.columns[reference_pdf.columns.duplicated()].tolist()}"
)
log("monitor", f"reference sample: {len(reference_pdf):,} of {reference_total:,} rows")

BINNINGS = {
    feature: drift.build_numeric_binning(
        feature, reference_pdf[feature], bins=CFG.psi_bins, epsilon=CFG.psi_epsilon
    )
    for feature in NUMERIC
}
REFERENCE_CATEGORICAL = {
    feature: drift.category_proportions(
        reference_pdf[feature].tolist(),
        schema.ALLOWED_TRANSACTION_TYPES,
        epsilon=CFG.psi_epsilon,
    )
    for feature in CATEGORICAL
}
REFERENCE_MISSING = {
    feature: drift.missing_rate(reference_pdf[feature].tolist()) for feature in SERVED
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Assemble the inference log
# MAGIC
# MAGIC Scores come from `scored_predictions` (written by notebook 04). This cell
# MAGIC assigns each scored row to a monitoring window and marks whether its label
# MAGIC would have arrived yet.

# COMMAND ----------

scored_monitoring = (
    scored_all
    .where(F.col("split_label") != F.lit(splits.TRAIN))
    .withColumn("window_id", F.expr(window_case))
)
log("monitor", f"monitoring scores: {scored_monitoring.count():,} rows")

label_available_case = "CASE " + " ".join(
    f"WHEN window_id = '{w['window_id']}' THEN {str(bool(w['label_available'])).lower()}"
    for w in WINDOWS
) + " ELSE false END"

inference_sdf = (
    scored_monitoring
    .withColumn("predicted_label",
                (F.col("fraud_score") >= F.lit(SCORING_THRESHOLD)).cast("int"))
    .withColumn("label_available", F.expr(label_available_case))
    .withColumn("scored_at", F.lit(COMPUTED_AT).cast("timestamp"))
    .select(
        "transaction_id", "step", "window_id", "split_label",
        "type", "amount", "fraud_score", "predicted_label",
        "label_true", "label_available",
        "model_name", "model_version", "scoring_threshold",
        "scoring_path", "scored_at",
    )
)

(
    inference_sdf.write.mode("overwrite").option("overwriteSchema", "true")
    .partitionBy("window_id").saveAsTable(CFG.inference_log_table)
)
# Re-read from Delta rather than caching: serverless rejects .cache()
# ([NOT_SUPPORTED_WITH_SERVERLESS]).
inference_sdf = spark_session.table(CFG.inference_log_table)
inference_count = inference_sdf.count()
log("monitor", f"{CFG.inference_log_table}: {inference_count:,} rows "
               f"(path={SCORING_PATH})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Reference score distribution
# MAGIC
# MAGIC Prediction drift needs a reference too: the score distribution the model
# MAGIC produced on its own training period.

# COMMAND ----------

# Reference scores come from the train-period rows in `scored_predictions`,
# which notebook 04 sampled *uniformly* (natural prevalence) rather than using
# the class-balanced training sample -- a balanced reference would overstate the
# fraud rate and make every later window look like negative drift.
reference_scored_pdf = (
    scored_all.where(F.col("split_label") == F.lit(splits.TRAIN))
    .select("fraud_score", "label_true")
    .toPandas()
)
if reference_scored_pdf.empty:
    raise RuntimeError(
        f"No train-period rows in {CFG.scored_predictions_table}; "
        "cannot establish a reference score distribution."
    )

reference_scores = reference_scored_pdf["fraud_score"].to_numpy(dtype="float64")
SCORE_BINNING = drift.build_numeric_binning(
    "fraud_score", reference_scores, bins=CFG.psi_bins, epsilon=CFG.psi_epsilon
)
REFERENCE_SCORE_STATS = drift.score_distribution(reference_scores, SCORING_THRESHOLD)
REFERENCE_PR_AUC = metrics.pr_auc(
    reference_scored_pdf["label_true"], reference_scores
)
log("monitor", f"reference avg score={REFERENCE_SCORE_STATS['avg_score']:.6f} "
               f"positive rate={REFERENCE_SCORE_STATS['predicted_positive_rate']:.6f} "
               f"PR-AUC={REFERENCE_PR_AUC:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Window dimension table

# COMMAND ----------

window_counts = {
    r["window_id"]: r["n"]
    for r in inference_sdf.groupBy("window_id").agg(F.count("*").alias("n")).collect()
}

window_rows = [
    {
        "window_id": w["window_id"],
        "window_index": w["window_index"],
        "start_step": w["start_step"],
        "end_step": w["end_step"],
        "row_count": int(window_counts.get(w["window_id"], 0)),
        "label_available": bool(w["label_available"]),
        "computed_at": COMPUTED_AT,
        "reference_split": splits.TRAIN,
        "model_version": MODEL_VERSION,
    }
    for w in WINDOWS
]
(
    spark_session.createDataFrame(window_rows)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(CFG.window_dim_table)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Feature drift — PSI and missing-rate change
# MAGIC
# MAGIC PSI is computed in Spark: rows are bucketed with the frozen reference edges
# MAGIC and counted per window, so only a tiny aggregate reaches the driver.

# COMMAND ----------

# Bucketing SQL lives in fraud_common.drift so it is unit-tested -- a binary
# feature yields no interior edge, and the naive version emitted `CASE  ELSE 0
# END`, a SQL parse error.
bucket_case = drift.bucket_case_sql

feature_drift_rows = []

for feature in NUMERIC:
    binning = BINNINGS[feature]
    n_bins = len(binning.reference_proportions)
    # Computed from `monitoring_sdf`, not the inference log: feature drift is a
    # property of the incoming data and stays measurable even if scoring fell
    # back to a sample.
    counts = (
        monitoring_sdf
        .withColumn("bucket", F.expr(bucket_case(feature, binning.edges)))
        .groupBy("window_id", "bucket")
        .agg(F.count("*").alias("n"))
        .collect()
    )
    per_window = {}
    for row in counts:
        per_window.setdefault(row["window_id"], np.zeros(n_bins))
        idx = int(row["bucket"])
        if 0 <= idx < n_bins:
            per_window[row["window_id"]][idx] += row["n"]

    for w in WINDOWS:
        vector = per_window.get(w["window_id"])
        if vector is None or vector.sum() == 0:
            psi_value = float("nan")
        else:
            props = np.maximum(vector / vector.sum(), CFG.psi_epsilon)
            psi_value = drift.psi_from_proportions(binning.reference_proportions, props)
        feature_drift_rows.append({
            "computed_at": COMPUTED_AT,
            "window_id": w["window_id"],
            "window_index": w["window_index"],
            "feature": feature,
            "feature_kind": "numeric",
            "drift_kind": drift.FEATURE_DRIFT,
            "statistic": "psi",
            "value": float(psi_value),
            "threshold_amber": CFG.psi_amber_threshold,
            "threshold_red": CFG.psi_red_threshold,
            "status": drift.psi_status(
                psi_value, CFG.psi_amber_threshold, CFG.psi_red_threshold
            ),
        })
    log("monitor", f"PSI computed for {feature}")

# COMMAND ----------

for feature in CATEGORICAL:
    counts = (
        monitoring_sdf.groupBy("window_id", F.col(feature).alias("category"))
        .agg(F.count("*").alias("n"))
        .collect()
    )
    per_window = {}
    for row in counts:
        per_window.setdefault(row["window_id"], {})[row["category"]] = int(row["n"])

    for w in WINDOWS:
        observed = per_window.get(w["window_id"], {})
        total = sum(observed.values())
        if total == 0:
            psi_value = float("nan")
        else:
            current = {
                c: max(observed.get(c, 0) / total, CFG.psi_epsilon)
                for c in schema.ALLOWED_TRANSACTION_TYPES
            }
            unseen = sum(
                v for k, v in observed.items()
                if k not in schema.ALLOWED_TRANSACTION_TYPES
            )
            current["__unseen__"] = max(unseen / total, CFG.psi_epsilon)
            psi_value = drift.psi_categorical(REFERENCE_CATEGORICAL[feature], current)
        feature_drift_rows.append({
            "computed_at": COMPUTED_AT,
            "window_id": w["window_id"],
            "window_index": w["window_index"],
            "feature": feature,
            "feature_kind": "categorical",
            "drift_kind": drift.FEATURE_DRIFT,
            "statistic": "psi",
            "value": float(psi_value),
            "threshold_amber": CFG.psi_amber_threshold,
            "threshold_red": CFG.psi_red_threshold,
            "status": drift.psi_status(
                psi_value, CFG.psi_amber_threshold, CFG.psi_red_threshold
            ),
        })

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing-value rate change
# MAGIC
# MAGIC Nulls in a served feature are an upstream-plumbing failure, and they move
# MAGIC independently of the value distribution — a feed that starts returning NULL
# MAGIC for `old_balance_dest` may barely move PSI while silently degrading every
# MAGIC prediction.

# COMMAND ----------

missing_agg = (
    monitoring_sdf.groupBy("window_id")
    .agg(
        F.count("*").alias("n"),
        *[
            F.sum(F.col(c).isNull().cast("long")).alias(f"null_{c}")
            for c in SERVED
        ],
    )
    .collect()
)
missing_by_window = {r["window_id"]: r.asDict() for r in missing_agg}

for w in WINDOWS:
    row = missing_by_window.get(w["window_id"])
    for feature in SERVED:
        if row is None or not row["n"]:
            current_rate, delta = float("nan"), float("nan")
        else:
            current_rate = float(row[f"null_{feature}"] or 0) / row["n"]
            delta = current_rate - REFERENCE_MISSING[feature]
        feature_drift_rows.append({
            "computed_at": COMPUTED_AT,
            "window_id": w["window_id"],
            "window_index": w["window_index"],
            "feature": feature,
            "feature_kind": "any",
            "drift_kind": drift.FEATURE_DRIFT,
            "statistic": "missing_rate_delta",
            "value": float(delta),
            "threshold_amber": CFG.missing_rate_amber_threshold,
            "threshold_red": CFG.missing_rate_red_threshold,
            "status": drift.missing_rate_status(
                delta, CFG.missing_rate_amber_threshold, CFG.missing_rate_red_threshold
            ),
        })

feature_drift_pdf = pd.DataFrame(feature_drift_rows)
(
    spark_session.createDataFrame(feature_drift_pdf)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(CFG.feature_drift_table)
)
log("monitor", f"{CFG.feature_drift_table}: {len(feature_drift_pdf):,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · Prediction drift
# MAGIC
# MAGIC Score distribution, average score, predicted-positive rate and transaction
# MAGIC volume per window. Available immediately, with no labels — this is the
# MAGIC early-warning layer.

# COMMAND ----------

score_agg = (
    inference_sdf.groupBy("window_id")
    .agg(
        F.count("*").alias("n_scored"),
        F.avg("fraud_score").alias("avg_score"),
        F.stddev("fraud_score").alias("std_score"),
        F.expr("percentile_approx(fraud_score, 0.5)").alias("median_score"),
        F.expr("percentile_approx(fraud_score, 0.9)").alias("p90_score"),
        F.expr("percentile_approx(fraud_score, 0.99)").alias("p99_score"),
        F.max("fraud_score").alias("max_score"),
        F.sum("predicted_label").alias("predicted_positive_count"),
        F.sum("amount").alias("total_amount"),
    )
    .collect()
)
score_by_window = {r["window_id"]: r.asDict() for r in score_agg}

score_bucket_counts = (
    inference_sdf
    .withColumn("bucket", F.expr(bucket_case("fraud_score", SCORE_BINNING.edges)))
    .groupBy("window_id", "bucket")
    .agg(F.count("*").alias("n"))
    .collect()
)
score_psi_vectors = {}
for row in score_bucket_counts:
    vec = score_psi_vectors.setdefault(
        row["window_id"], np.zeros(len(SCORE_BINNING.reference_proportions))
    )
    idx = int(row["bucket"])
    if 0 <= idx < len(vec):
        vec[idx] += row["n"]

prediction_rows = []
for w in WINDOWS:
    stats = score_by_window.get(w["window_id"])
    vec = score_psi_vectors.get(w["window_id"])
    if vec is None or vec.sum() == 0:
        score_psi = float("nan")
    else:
        props = np.maximum(vec / vec.sum(), CFG.psi_epsilon)
        score_psi = drift.psi_from_proportions(
            SCORE_BINNING.reference_proportions, props
        )
    n = int(stats["n_scored"]) if stats else 0
    positive_rate = (
        float(stats["predicted_positive_count"] or 0) / n if stats and n else float("nan")
    )
    prediction_rows.append({
        "computed_at": COMPUTED_AT,
        "window_id": w["window_id"],
        "window_index": w["window_index"],
        "drift_kind": drift.PREDICTION_DRIFT,
        "n_scored": n,
        "transaction_volume": n,
        "total_amount": float(stats["total_amount"] or 0.0) if stats else 0.0,
        "avg_score": float(stats["avg_score"]) if stats and stats["avg_score"] is not None else float("nan"),
        "std_score": float(stats["std_score"]) if stats and stats["std_score"] is not None else float("nan"),
        "median_score": float(stats["median_score"]) if stats and stats["median_score"] is not None else float("nan"),
        "p90_score": float(stats["p90_score"]) if stats and stats["p90_score"] is not None else float("nan"),
        "p99_score": float(stats["p99_score"]) if stats and stats["p99_score"] is not None else float("nan"),
        "max_score": float(stats["max_score"]) if stats and stats["max_score"] is not None else float("nan"),
        "predicted_positive_count": int(stats["predicted_positive_count"] or 0) if stats else 0,
        "predicted_positive_rate": positive_rate,
        "reference_avg_score": REFERENCE_SCORE_STATS["avg_score"],
        "reference_predicted_positive_rate": REFERENCE_SCORE_STATS["predicted_positive_rate"],
        "score_psi": float(score_psi),
        "threshold_amber": CFG.psi_amber_threshold,
        "threshold_red": CFG.psi_red_threshold,
        "status": drift.psi_status(
            score_psi, CFG.psi_amber_threshold, CFG.psi_red_threshold
        ),
        "scoring_threshold": SCORING_THRESHOLD,
    })

(
    spark_session.createDataFrame(pd.DataFrame(prediction_rows))
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(CFG.prediction_drift_table)
)
log("monitor", f"{CFG.prediction_drift_table}: {len(prediction_rows)} rows")

# COMMAND ----------

histogram_rows = []
hist_counts = (
    inference_sdf
    .withColumn(
        "bucket",
        F.least(F.floor(F.col("fraud_score") * F.lit(20)), F.lit(19)).cast("int"),
    )
    .groupBy("window_id", "bucket")
    .agg(F.count("*").alias("n"))
    .collect()
)
hist_by_window = {}
for row in hist_counts:
    hist_by_window.setdefault(row["window_id"], {})[int(row["bucket"])] = int(row["n"])

for w in WINDOWS:
    buckets = hist_by_window.get(w["window_id"], {})
    total = sum(buckets.values()) or 1
    for i in range(20):
        histogram_rows.append({
            "computed_at": COMPUTED_AT,
            "window_id": w["window_id"],
            "window_index": w["window_index"],
            "bucket_index": i,
            "bucket_lower": i / 20.0,
            "bucket_upper": (i + 1) / 20.0,
            "row_count": int(buckets.get(i, 0)),
            "row_share": float(buckets.get(i, 0)) / total,
        })

(
    spark_session.createDataFrame(pd.DataFrame(histogram_rows))
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(CFG.score_histogram_table)
)
log("monitor", f"{CFG.score_histogram_table}: {len(histogram_rows)} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 · Performance — only where labels exist
# MAGIC
# MAGIC No row is written for an unlabelled window. An absent row is the honest
# MAGIC representation of "we cannot know yet"; a zero or a carried-forward value
# MAGIC would be a lie the dashboard would happily plot.

# COMMAND ----------

labelled_pdf = (
    inference_sdf.where("label_available")
    .select("window_id", "label_true", "fraud_score", "amount")
    .toPandas()
)

performance_rows = []
for w in WINDOWS:
    if not w["label_available"]:
        log("monitor", f"{w['window_id']}: labels not yet available — no "
                       "performance row written (delayed-label state)")
        continue
    subset = labelled_pdf[labelled_pdf["window_id"] == w["window_id"]]
    if subset.empty or subset["label_true"].sum() == 0:
        log("monitor", f"{w['window_id']}: no positives in window — PR-AUC undefined")
        continue

    window_metrics = metrics.evaluate(
        subset["label_true"], subset["fraud_score"],
        SCORING_THRESHOLD, amounts=subset["amount"],
    )
    pr_auc_delta = window_metrics["pr_auc"] - REFERENCE_PR_AUC
    # A material *drop* is what matters; an improvement is not a problem.
    if pr_auc_delta <= -0.10:
        status = drift.RED
    elif pr_auc_delta <= -0.05:
        status = drift.AMBER
    else:
        status = drift.GREEN

    performance_rows.append({
        "computed_at": COMPUTED_AT,
        "window_id": w["window_id"],
        "window_index": w["window_index"],
        "drift_kind": drift.PERFORMANCE,
        "label_available": True,
        "n_labeled": int(window_metrics["n_rows"]),
        "n_positives": int(window_metrics["n_positives"]),
        "prevalence": window_metrics["prevalence"],
        "pr_auc": window_metrics["pr_auc"],
        "roc_auc": window_metrics["roc_auc"],
        "precision": window_metrics["precision"],
        "recall": window_metrics["recall"],
        "f1": window_metrics["f1"],
        "false_positive_rate": window_metrics["false_positive_rate"],
        "true_positives": int(window_metrics["true_positives"]),
        "false_positives": int(window_metrics["false_positives"]),
        "true_negatives": int(window_metrics["true_negatives"]),
        "false_negatives": int(window_metrics["false_negatives"]),
        "fraud_amount_total": window_metrics.get("fraud_amount_total", float("nan")),
        "fraud_amount_captured": window_metrics.get("fraud_amount_captured", float("nan")),
        "fraud_amount_capture_rate": window_metrics.get("fraud_amount_capture_rate", float("nan")),
        "reference_pr_auc": REFERENCE_PR_AUC,
        "pr_auc_delta": pr_auc_delta,
        "status": status,
        "threshold": SCORING_THRESHOLD,
    })
    log("monitor", f"{w['window_id']}: PR-AUC={window_metrics['pr_auc']:.4f} "
                   f"precision={window_metrics['precision']:.4f} "
                   f"recall={window_metrics['recall']:.4f} -> {status}")

if performance_rows:
    (
        spark_session.createDataFrame(pd.DataFrame(performance_rows))
        .write.mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(CFG.performance_table)
    )
    log("monitor", f"{CFG.performance_table}: {len(performance_rows)} rows")
else:
    log("monitor", "WARNING: no labelled window produced a performance row.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10 · Persist the reference statistics

# COMMAND ----------

reference_rows = [
    {
        "computed_at": COMPUTED_AT,
        "feature": feature,
        "feature_kind": "numeric",
        "reference_split": splits.TRAIN,
        "reference_rows": len(reference_pdf),
        "edges_json": json.dumps(BINNINGS[feature].edges),
        "proportions_json": json.dumps(BINNINGS[feature].reference_proportions),
        "mean_value": float(np.nanmean(reference_pdf[feature].astype(float))),
        "std_value": float(np.nanstd(reference_pdf[feature].astype(float))),
        "missing_rate": REFERENCE_MISSING[feature],
        "model_version": MODEL_VERSION,
    }
    for feature in NUMERIC
] + [
    {
        "computed_at": COMPUTED_AT,
        "feature": feature,
        "feature_kind": "categorical",
        "reference_split": splits.TRAIN,
        "reference_rows": len(reference_pdf),
        "edges_json": "",
        "proportions_json": json.dumps(REFERENCE_CATEGORICAL[feature]),
        "mean_value": float("nan"),
        "std_value": float("nan"),
        "missing_rate": REFERENCE_MISSING[feature],
        "model_version": MODEL_VERSION,
    }
    for feature in CATEGORICAL
] + [
    {
        "computed_at": COMPUTED_AT,
        "feature": "fraud_score",
        "feature_kind": "prediction",
        "reference_split": splits.TRAIN,
        "reference_rows": len(reference_pdf),
        "edges_json": json.dumps(SCORE_BINNING.edges),
        "proportions_json": json.dumps(SCORE_BINNING.reference_proportions),
        "mean_value": REFERENCE_SCORE_STATS["avg_score"],
        "std_value": REFERENCE_SCORE_STATS["std_score"],
        "missing_rate": 0.0,
        "model_version": MODEL_VERSION,
    }
]

(
    spark_session.createDataFrame(pd.DataFrame(reference_rows))
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(CFG.reference_stats_table)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

overall = drift.overall_status(
    [r["status"] for r in feature_drift_rows]
    + [r["status"] for r in prediction_rows]
    + [r["status"] for r in performance_rows]
)

SUMMARY = {
    "generated_at": utc_now_iso(),
    "model_uri": MODEL_URI,
    "model_version": MODEL_VERSION,
    "scoring_path": SCORING_PATH,
    "inference_rows": int(inference_count),
    "windows": len(WINDOWS),
    "labelled_windows": LABELLED_WINDOW_COUNT,
    "unlabelled_windows_delayed": len(WINDOWS) - LABELLED_WINDOW_COUNT,
    "reference_pr_auc": REFERENCE_PR_AUC,
    "overall_status": overall,
    "tables": {
        "inference_log": CFG.inference_log_table,
        "windows": CFG.window_dim_table,
        "feature_drift": CFG.feature_drift_table,
        "prediction_drift": CFG.prediction_drift_table,
        "score_histogram": CFG.score_histogram_table,
        "performance": CFG.performance_table,
        "reference_stats": CFG.reference_stats_table,
    },
    "interpretation_note": (
        "Feature drift and prediction drift are observable without labels but "
        "do not by themselves demonstrate model degradation. Only the "
        "performance table, which requires labels, can do that -- and it is "
        "deliberately blank for the most recent windows."
    ),
}

try:
    write_json_to_volume("monitoring_summary.json", SUMMARY)
except Exception as exc:
    log("monitor", f"could not persist summary: {type(exc).__name__}")

print(json.dumps(SUMMARY, indent=2, default=str))
log("monitor", f"DONE — overall drift status: {overall}")
