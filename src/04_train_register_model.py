# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Train, track and register
# MAGIC
# MAGIC ## Evaluation methodology
# MAGIC
# MAGIC * **Chronological split**, never random. Train on the earliest 70 % of
# MAGIC   `step`, validate on the next 15 %, test on the final 15 %. A random split
# MAGIC   lets the model see the future and inflates every metric.
# MAGIC * **Down-sample only the training majority class.** Every fraud row in the
# MAGIC   training period is kept; legitimate rows are thinned by a SHA-256 hash of
# MAGIC   the transaction id, so the same rows are chosen on every rerun and on any
# MAGIC   cluster size (`.sample(seed=)` is not stable across repartitioning).
# MAGIC * **Validation and test keep their natural prevalence** (~0.13 %). Balancing
# MAGIC   them would make precision a fiction.
# MAGIC * **PR-AUC selects the winner.** At this prevalence, accuracy is 99.87 % for
# MAGIC   a model that never predicts fraud, and ROC-AUC flatters any ranker because
# MAGIC   the true-negative pool is enormous.
# MAGIC * **The operating threshold is chosen on validation only**, as the
# MAGIC   highest-recall point inside a false-positive budget. Tuning it on test
# MAGIC   would leak the test set into the reported numbers.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json
import time

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

spark_session = get_spark()
np.random.seed(CFG.random_seed)

EXPERIMENT_NAME = resolved_experiment_name(spark_session)
mlflow.set_registry_uri("databricks-uc")   # register into Unity Catalog
mlflow.set_experiment(EXPERIMENT_NAME)
log("train", f"experiment: {EXPERIMENT_NAME}")
log("train", f"registry target: {CFG.uc_model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Gold and rebuild the split boundaries

# COMMAND ----------

gold = spark_session.table(CFG.gold_features_table)

bounds = gold.select(
    F.min("step").alias("min_step"), F.max("step").alias("max_step")
).collect()[0]
BOUNDARIES = splits.compute_boundaries(
    int(bounds["min_step"]), int(bounds["max_step"]),
    CFG.train_step_fraction, CFG.valid_step_fraction,
)
log("train", f"boundaries: {BOUNDARIES.as_dict()}")

SERVED = schema.served_features()
schema.assert_no_leakage(SERVED)
KEEP_COLUMNS = SERVED + ["step", "is_fraud", "transaction_id"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the training sample
# MAGIC
# MAGIC The sampling predicate runs in Spark so 6.3 M rows are never collected to
# MAGIC the driver — only the reduced sample is.

# COMMAND ----------

sample_predicate = splits.sample_sql_expression(
    key_col="transaction_id",
    seed=CFG.random_seed,
    fraction=CFG.train_negative_sample_fraction,
    keep_all_when="is_fraud = 1",
)

train_sdf = (
    gold.where(F.col("split_label") == splits.TRAIN)
    .where(F.expr(sample_predicate))
    .select(*KEEP_COLUMNS)
)

train_period_total = gold.where(F.col("split_label") == splits.TRAIN).count()
train_pdf = train_sdf.toPandas()
log("train", f"training period rows: {train_period_total:,} -> sampled "
             f"{len(train_pdf):,} ({len(train_pdf)/max(train_period_total,1):.2%})")
log("train", f"training fraud rows retained: {int(train_pdf['is_fraud'].sum()):,} "
             f"({train_pdf['is_fraud'].mean():.4%} of the sample)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load validation and test at natural prevalence

# COMMAND ----------

def load_split(split_name: str) -> pd.DataFrame:
    """Collect an evaluation split, optionally capped but never rebalanced.

    If `eval_max_rows` is set, rows are thinned uniformly across *both* classes
    with the same deterministic hash, which preserves the fraud prevalence.
    Class-conditional sampling here would silently rewrite precision.
    """
    sdf = gold.where(F.col("split_label") == split_name).select(*KEEP_COLUMNS)
    total = sdf.count()
    if CFG.eval_max_rows and total > CFG.eval_max_rows:
        fraction = CFG.eval_max_rows / total
        predicate = splits.sample_sql_expression(
            "transaction_id", CFG.random_seed + 1, fraction, keep_all_when=None
        )
        sdf = sdf.where(F.expr(predicate))
        log("train", f"{split_name}: capped {total:,} -> ~{CFG.eval_max_rows:,} "
                     "(uniform, prevalence preserved)")
    pdf = sdf.toPandas()
    log("train", f"{split_name}: {len(pdf):,} rows, "
                 f"{int(pdf['is_fraud'].sum()):,} fraud "
                 f"({pdf['is_fraud'].mean():.4%})")
    return pdf


valid_pdf = load_split(splits.VALID)
test_pdf = load_split(splits.TEST)

splits.assert_chronological(
    [("train", train_pdf), ("valid", valid_pdf), ("test", test_pdf)]
)
log("train", "chronological ordering verified")

# COMMAND ----------

X_train, y_train = train_pdf[SERVED], train_pdf["is_fraud"].astype(int)
X_valid, y_valid = valid_pdf[SERVED], valid_pdf["is_fraud"].astype(int)
X_test, y_test = test_pdf[SERVED], test_pdf["is_fraud"].astype(int)

assert list(X_train.columns) == features.signature_input_columns(), \
    "feature order must match the model signature"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model definitions
# MAGIC
# MAGIC A logistic-regression baseline plus one gradient-boosted tree ensemble.
# MAGIC `HistGradientBoostingClassifier` ships with scikit-learn, so no extra
# MAGIC library is required on Free Edition. No hyperparameter search — this is a
# MAGIC one-day PoC and tuning would not change the architecture it demonstrates.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

NUMERIC_FEATURES = schema.numeric_features()
CATEGORICAL_FEATURES = schema.categorical_features()


def make_one_hot() -> OneHotEncoder:
    """Dense one-hot, tolerant of sklearn's sparse/sparse_output rename."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:                       # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("categorical", make_one_hot(), CATEGORICAL_FEATURES),
            ("numeric", numeric_step, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


MODEL_SPECS = {
    "logistic_regression": {
        "pipeline": lambda: Pipeline([
            ("preprocess", make_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=CFG.random_seed,
            )),
        ]),
        "params": {
            "algorithm": "logistic_regression",
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "lbfgs",
            "scaled_numeric": True,
        },
        "use_sample_weight": False,
        "role": "baseline",
    },
    "hist_gradient_boosting": {
        "pipeline": lambda: Pipeline([
            ("preprocess", make_preprocessor(scale_numeric=False)),
            ("model", HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.1,
                max_leaf_nodes=31,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=CFG.random_seed,
            )),
        ]),
        "params": {
            "algorithm": "hist_gradient_boosting",
            "max_iter": 200,
            "learning_rate": 0.1,
            "max_leaf_nodes": 31,
            "early_stopping": True,
            "class_weight": "balanced_sample_weight",
        },
        "use_sample_weight": True,
        "role": "challenger",
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow logging helpers
# MAGIC
# MAGIC Written defensively so the notebook runs on both MLflow 2.x and 3.x — the
# MAGIC `artifact_path` → `name` rename in `log_model` is the usual breakage.

# COMMAND ----------

def log_sklearn_model(pipeline, signature, input_example, artifact_name="model"):
    """Log an sklearn pipeline whose pyfunc flavour returns probabilities.

    `pyfunc_predict_fn="predict_proba"` matters: the serving endpoint must
    return a score, not a hard 0/1, otherwise drift monitoring has nothing to
    measure and no threshold can be tuned after deployment.
    """
    kwargs = dict(
        sk_model=pipeline,
        signature=signature,
        input_example=input_example,
        pyfunc_predict_fn="predict_proba",
    )
    try:
        return mlflow.sklearn.log_model(name=artifact_name, **kwargs)
    except TypeError:
        return mlflow.sklearn.log_model(artifact_path=artifact_name, **kwargs)


def log_dataset_inputs(train_df, valid_df, test_df):
    """Record dataset lineage on the run, if this MLflow build supports it."""
    try:
        for frame, context, name in [
            (train_df, "training", "gold_train_sample"),
            (valid_df, "validation", "gold_valid"),
            (test_df, "test", "gold_test"),
        ]:
            dataset = mlflow.data.from_pandas(
                frame,
                source=CFG.gold_features_table,
                name=name,
                targets="is_fraud",
            )
            mlflow.log_input(dataset, context=context)
        return True
    except Exception as exc:
        log("train", f"dataset lineage unavailable: {type(exc).__name__}: {exc}")
        return False


def threshold_sweep(y_true, y_score, thresholds=None) -> pd.DataFrame:
    """Precision/recall/FPR across candidate operating points, for the record."""
    grid = thresholds if thresholds is not None else np.round(np.arange(0.05, 1.0, 0.05), 2)
    rows = []
    for t in grid:
        tp, fp, tn, fn = metrics.confusion_at_threshold(y_true, y_score, t)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append({
            "threshold": float(t),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
            "alerts_per_1k": 1000.0 * (tp + fp) / max(len(y_true), 1),
        })
    return pd.DataFrame(rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train

# COMMAND ----------

INPUT_EXAMPLE = X_train.head(5).copy()
RUN_TAGS = {
    "project": "paysim-fraud-lakehouse-poc",
    "leakage_policy": "pre_transaction_only",
    "excluded_features": ",".join(schema.leakage_columns()),
    "primary_metric": metrics.PRIMARY_METRIC,
    "split_strategy": "chronological_by_step_70_15_15",
    "dataset": "PaySim1 (Kaggle ealaxi/paysim1)",
}

results = {}
child_runs = {}
parent_run_id = None

with mlflow.start_run(run_name=f"paysim_fraud_{int(time.time())}") as parent_run:
    parent_run_id = parent_run.info.run_id
    mlflow.set_tags(RUN_TAGS)
    mlflow.log_params({
        "random_seed": CFG.random_seed,
        "train_negative_sample_fraction": CFG.train_negative_sample_fraction,
        "train_rows": len(train_pdf),
        "valid_rows": len(valid_pdf),
        "test_rows": len(test_pdf),
        "n_served_features": len(SERVED),
        **{f"boundary_{k}": v for k, v in BOUNDARIES.as_dict().items()},
    })
    mlflow.log_dict(
        {
            "served_features": SERVED,
            "excluded_columns": schema.leakage_columns(),
            "feature_contract": [
                {
                    "name": s.name,
                    "availability": s.availability,
                    "served": s.served,
                    "description": s.description,
                    "derivation": s.derivation,
                }
                for s in schema.GOLD_COLUMNS
            ],
        },
        "feature_contract.json",
    )
    log_dataset_inputs(train_pdf, valid_pdf, test_pdf)

    for model_name, spec in MODEL_SPECS.items():
        with mlflow.start_run(run_name=model_name, nested=True) as child:
            log("train", f"fitting {model_name} …")
            started = time.time()
            pipeline = spec["pipeline"]()

            if spec["use_sample_weight"]:
                weights = compute_sample_weight("balanced", y_train)
                pipeline.fit(X_train, y_train, model__sample_weight=weights)
            else:
                pipeline.fit(X_train, y_train)

            fit_seconds = time.time() - started

            valid_scores = pipeline.predict_proba(X_valid)[:, 1]
            valid_metrics = metrics.evaluate(
                y_valid, valid_scores, CFG.operational_threshold,
                amounts=valid_pdf["amount"], prefix="valid_",
            )

            mlflow.set_tags({**RUN_TAGS, "model_role": spec["role"]})
            mlflow.log_params({**spec["params"], "random_seed": CFG.random_seed})
            mlflow.log_metric("fit_seconds", fit_seconds)
            mlflow.log_metrics(valid_metrics)

            signature = infer_signature(X_valid.head(100), pipeline.predict_proba(X_valid.head(100)))
            model_info = log_sklearn_model(pipeline, signature, INPUT_EXAMPLE)

            results[model_name] = {
                "pipeline": pipeline,
                "valid_scores": valid_scores,
                "metrics": {k.replace("valid_", ""): v for k, v in valid_metrics.items()},
                "run_id": child.info.run_id,
                "model_uri": model_info.model_uri,
                "signature": signature,
                "fit_seconds": fit_seconds,
            }
            child_runs[model_name] = child.info.run_id
            log("train", f"{model_name}: valid PR-AUC="
                         f"{valid_metrics['valid_pr_auc']:.4f} "
                         f"ROC-AUC={valid_metrics['valid_roc_auc']:.4f} "
                         f"({fit_seconds:.1f}s)")

    # ---------------------------------------------------------------- #
    # Winner selection — validation PR-AUC only.
    # ---------------------------------------------------------------- #
    winner_name, winner_score, all_scores = metrics.compare_models(
        {k: v["metrics"] for k, v in results.items()}, metrics.PRIMARY_METRIC
    )
    winner = results[winner_name]
    log("train", f"winner: {winner_name} (valid PR-AUC={winner_score:.4f}) "
                 f"from {all_scores}")

    # ---------------------------------------------------------------- #
    # Operating threshold — chosen on VALIDATION, then frozen.
    # ---------------------------------------------------------------- #
    if CFG.target_max_fpr is not None:
        chosen_threshold = metrics.choose_threshold_for_max_fpr(
            y_valid, winner["valid_scores"], CFG.target_max_fpr,
            default=CFG.operational_threshold,
        )
        threshold_rationale = (
            f"highest-recall point with FPR <= {CFG.target_max_fpr} on validation"
        )
    else:
        chosen_threshold = CFG.operational_threshold
        threshold_rationale = "fixed configuration value"
    log("train", f"operating threshold: {chosen_threshold:.6f} ({threshold_rationale})")

    # ---------------------------------------------------------------- #
    # Final, untouched test evaluation.
    # ---------------------------------------------------------------- #
    test_scores = winner["pipeline"].predict_proba(X_test)[:, 1]
    test_metrics = metrics.evaluate(
        y_test, test_scores, chosen_threshold,
        amounts=test_pdf["amount"], prefix="test_",
    )
    valid_at_threshold = metrics.evaluate(
        y_valid, winner["valid_scores"], chosen_threshold,
        amounts=valid_pdf["amount"], prefix="valid_at_threshold_",
    )

    mlflow.log_metrics(test_metrics)
    mlflow.log_metrics(valid_at_threshold)
    mlflow.log_metrics({f"valid_pr_auc_{k}": v for k, v in all_scores.items()})
    mlflow.log_params({
        "winning_model": winner_name,
        "operational_threshold": chosen_threshold,
        "threshold_rationale": threshold_rationale,
    })

    sweep = threshold_sweep(y_test, test_scores)
    mlflow.log_text(sweep.to_csv(index=False), "test_threshold_sweep.csv")
    mlflow.log_dict(
        {
            "confusion_matrix_test": {
                "true_positives": test_metrics["test_true_positives"],
                "false_positives": test_metrics["test_false_positives"],
                "true_negatives": test_metrics["test_true_negatives"],
                "false_negatives": test_metrics["test_false_negatives"],
            },
            "threshold": chosen_threshold,
            "note": "Evaluated at natural fraud prevalence on an unseen future "
                    "time period.",
        },
        "test_confusion_matrix.json",
    )

    log("train", "--- TEST (final, unseen period) ---")
    for key in ["test_pr_auc", "test_roc_auc", "test_precision", "test_recall",
                "test_f1", "test_false_positive_rate", "test_prevalence",
                "test_alerts_per_1k_transactions", "test_fraud_amount_capture_rate"]:
        if key in test_metrics:
            log("train", f"  {key:<38} = {test_metrics[key]:.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch scoring for monitoring
# MAGIC
# MAGIC Scores are written **here**, while the fitted pipeline is still in memory,
# MAGIC rather than in notebook 06.
# MAGIC
# MAGIC **Why:** two independent Free Edition problems, both observed on real runs.
# MAGIC
# MAGIC 1. **The serverless identity lacks S3 rights on the Unity Catalog model
# MAGIC    location.** Registration of a second version failed with an explicit
# MAGIC    `AccessDenied` on `s3:PutObject` against
# MAGIC    `__unitystorage/.../models/...`. The read-side failures
# MAGIC    (`HeadObject` 400) are the same restriction surfacing on HEAD requests,
# MAGIC    which cannot carry an error body.
# MAGIC 2. **The runtime's MLflow client is 2.x against an MLflow 3 tracking
# MAGIC    server.** So `models:/m-<id>` is rejected as a malformed URI,
# MAGIC    `runs:/<id>/model` 404s because MLflow 3 does not write that layout, and
# MAGIC    a raw DBFS artefact path is misparsed as a run id.
# MAGIC
# MAGIC Either alone would block re-loading the model here; together there is no
# MAGIC route at all.
# MAGIC
# MAGIC **Honest consequence:** these scores come from the in-memory pipeline, which
# MAGIC is the same fitted object that was logged — but this does **not**
# MAGIC demonstrate that the registered Unity Catalog artefact can be loaded and
# MAGIC served. `scoring_path` records that distinction in the data.
# MAGIC
# MAGIC Three populations are scored:
# MAGIC
# MAGIC | Population | Why |
# MAGIC |---|---|
# MAGIC | uniform sample of the **train** period | reference score distribution and reference PR-AUC. Uniform, *not* the class-balanced training sample — using that would bias the reference toward fraud. |
# MAGIC | all **validation** rows | monitoring window, natural prevalence |
# MAGIC | all **test** rows | monitoring window, natural prevalence |

# COMMAND ----------

SCORING_PATH_LABEL = "training_notebook_in_memory_pipeline"
REFERENCE_SAMPLE_ROWS = 200_000

ref_fraction = min(1.0, REFERENCE_SAMPLE_ROWS / max(train_period_total, 1))
reference_pdf = (
    gold.where(F.col("split_label") == splits.TRAIN)
    .where(F.expr(splits.sample_sql_expression(
        "transaction_id", CFG.random_seed + 2, ref_fraction, keep_all_when=None
    )))
    .select(*KEEP_COLUMNS)
    .toPandas()
)
log("score", f"reference sample: {len(reference_pdf):,} rows at natural "
             f"prevalence ({reference_pdf['is_fraud'].mean():.4%})")

reference_scores = winner["pipeline"].predict_proba(reference_pdf[SERVED])[:, 1]

scored_pdf = pd.concat(
    [
        pd.DataFrame({
            "transaction_id": frame["transaction_id"].values,
            "step": frame["step"].astype(int).values,
            "split_label": split_name,
            "type": frame["type"].astype(str).values,
            "amount": frame["amount"].astype(float).values,
            "fraud_score": np.asarray(scores).astype(float),
            "label_true": frame["is_fraud"].astype(int).values,
        })
        for split_name, frame, scores in [
            (splits.TRAIN, reference_pdf, reference_scores),
            (splits.VALID, valid_pdf, winner["valid_scores"]),
            (splits.TEST, test_pdf, test_scores),
        ]
    ],
    ignore_index=True,
)
log("score", f"scored {len(scored_pdf):,} rows: "
             f"{scored_pdf.groupby('split_label').size().to_dict()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the winner in Unity Catalog
# MAGIC
# MAGIC Registration and the `Champion` alias are attempted separately: on a
# MAGIC restricted workspace the alias can fail while the version is created
# MAGIC perfectly well, and that distinction has to survive into the report.

# COMMAND ----------

REGISTRATION = {
    "attempted": True,
    "registered": False,
    "alias_set": False,
    "model_name": CFG.uc_model_name,
    "model_version": None,
    "run_id": winner["run_id"],
    "parent_run_id": parent_run_id,
    "experiment_name": EXPERIMENT_NAME,
    "winning_model": winner_name,
    "operational_threshold": chosen_threshold,
    "error": None,
}

try:
    registered = mlflow.register_model(
        model_uri=winner["model_uri"],
        name=CFG.uc_model_name,
        tags={"winning_algorithm": winner_name,
              "primary_metric": metrics.PRIMARY_METRIC},
    )
    REGISTRATION["registered"] = True
    REGISTRATION["model_version"] = str(registered.version)
    log("register", f"registered {CFG.uc_model_name} version {registered.version}")

    client = MlflowClient()
    client.update_model_version(
        name=CFG.uc_model_name,
        version=registered.version,
        description=(
            f"PaySim fraud classifier ({winner_name}). Pre-transaction features "
            f"only; post-transaction balances and isFlaggedFraud excluded. "
            f"Validation PR-AUC {winner_score:.4f}; test PR-AUC "
            f"{test_metrics['test_pr_auc']:.4f} at natural prevalence. "
            f"Operating threshold {chosen_threshold:.6f} ({threshold_rationale}). "
            f"pyfunc output is predict_proba: [P(legit), P(fraud)]."
        ),
    )

    try:
        client.set_registered_model_alias(
            CFG.uc_model_name, CFG.champion_alias, registered.version
        )
        REGISTRATION["alias_set"] = True
        log("register", f"alias '{CFG.champion_alias}' -> version {registered.version}")
    except Exception as alias_exc:
        REGISTRATION["error"] = serving.sanitize_error(alias_exc)
        log("register", f"alias NOT set ({type(alias_exc).__name__}). The version "
                        "exists; notebook 05 will fall back to the latest version.")

except Exception as exc:
    REGISTRATION["error"] = serving.sanitize_error(exc)
    log("register", f"REGISTRATION FAILED: {REGISTRATION['error']}")
    log("register", "Common causes: no CREATE MODEL privilege on the schema, or "
                    "the workspace is not UC-enabled. Fix with: "
                    f"GRANT CREATE MODEL, USE SCHEMA ON SCHEMA {CFG.full_schema} "
                    "TO `<your-principal>`;")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persist the batch scores
# MAGIC
# MAGIC Written after registration so the row carries the model version it came
# MAGIC from. Notebook 06 reads this table instead of re-scoring.

# COMMAND ----------

# If this run could not register (Free Edition S3 AccessDenied is common), fall
# back to the newest version that already exists, so the scores are still
# attributable to a real registered model rather than to nothing.
EFFECTIVE_MODEL_VERSION = REGISTRATION.get("model_version")
if not EFFECTIVE_MODEL_VERSION:
    try:
        _existing = MlflowClient().search_model_versions(
            f"name = '{CFG.uc_model_name}'"
        )
        if _existing:
            EFFECTIVE_MODEL_VERSION = str(
                max(_existing, key=lambda v: int(v.version)).version
            )
            REGISTRATION["reused_existing_version"] = EFFECTIVE_MODEL_VERSION
            log("register", f"registration failed; scores attributed to the "
                            f"existing version {EFFECTIVE_MODEL_VERSION}")
    except Exception as exc:
        log("register", f"could not list existing versions: {type(exc).__name__}")

scored_pdf["model_name"] = CFG.uc_model_name
scored_pdf["model_version"] = str(EFFECTIVE_MODEL_VERSION or "unregistered")
scored_pdf["scoring_threshold"] = float(chosen_threshold)
scored_pdf["scoring_path"] = SCORING_PATH_LABEL
scored_pdf["winning_algorithm"] = winner_name

(
    spark_session.createDataFrame(scored_pdf)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .partitionBy("split_label")
    .saveAsTable(CFG.scored_predictions_table)
)
log("score", f"{CFG.scored_predictions_table}: {len(scored_pdf):,} rows written")

# COMMAND ----------

TRAINING_SUMMARY = {
    **REGISTRATION,
    "scored_predictions_table": CFG.scored_predictions_table,
    "scoring_path": SCORING_PATH_LABEL,
    "child_runs": child_runs,
    "validation_pr_auc_by_model": all_scores,
    "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    "split_boundaries": BOUNDARIES.as_dict(),
    "train_rows": len(train_pdf),
    "valid_rows": len(valid_pdf),
    "test_rows": len(test_pdf),
    "served_features": SERVED,
    "generated_at": utc_now_iso(),
}

try:
    write_json_to_volume("training_summary.json", TRAINING_SUMMARY)
except Exception as exc:
    log("train", f"could not write evidence to the volume: {type(exc).__name__}")

print(json.dumps(serving.sanitize_evidence(TRAINING_SUMMARY), indent=2, default=str))

if not REGISTRATION["registered"]:
    if REGISTRATION.get("reused_existing_version"):
        # ETL, training, evaluation and batch scoring are all still valid; only
        # *this* run's registration failed. Failing the task here would throw
        # that work away and block monitoring for no benefit.
        log("register", "WARNING: this run did not register a new version. "
                        f"Existing version {REGISTRATION['reused_existing_version']} "
                        "remains the servable model. Do NOT describe this run as "
                        "having produced a new registered model.")
    else:
        raise RuntimeError(
            "Model registration did not succeed and no existing version was "
            "found — notebook 05 cannot deploy. See the error above. Do not "
            "report this run as deployed."
        )
log("train", f"DONE — {CFG.uc_model_name} v{REGISTRATION['model_version']}")
