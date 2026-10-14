# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Configuration
# MAGIC
# MAGIC Shared setup for every notebook in the pipeline. Other notebooks start with
# MAGIC `%run ./00_config`, which makes `CFG` and the helper functions below
# MAGIC available in their own namespace.
# MAGIC
# MAGIC **Nothing here contains a workspace URL, username, token or Kaggle
# MAGIC credential.** The MLflow experiment path is resolved at runtime from
# MAGIC `current_user()`; everything else is a plain default that can be overridden
# MAGIC by a widget, an environment variable, or a job parameter.
# MAGIC
# MAGIC Running this notebook on its own just prints the resolved configuration.

# COMMAND ----------

import json
import os
import sys
from datetime import datetime, timezone

# COMMAND ----------

# MAGIC %md
# MAGIC ## Put `fraud_common` on the import path
# MAGIC
# MAGIC In a Databricks Git folder the notebook's own directory is already on
# MAGIC `sys.path`, so `import fraud_common` resolves. The walk below is a
# MAGIC fallback for workspace layouts (or a local shell) where it is not.


# COMMAND ----------

def _bootstrap_import_path() -> str:
    """Locate the directory containing the `fraud_common` package."""
    here = os.getcwd()
    candidates = []
    probe = here
    for _ in range(5):
        candidates += [probe, os.path.join(probe, "src")]
        probe = os.path.dirname(probe) or "/"
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "fraud_common")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return candidate
    raise ImportError(
        "could not locate the 'fraud_common' package. Expected it beside this "
        f"notebook or under ./src. Searched from cwd={here!r}. If you are "
        "running in a Databricks Git folder, confirm the repo was cloned whole."
    )


PACKAGE_ROOT = _bootstrap_import_path()
print(f"fraud_common import root: {PACKAGE_ROOT}")

from fraud_common import drift, features, metrics, schema, serving, splits  # noqa: E402
from fraud_common.config import Config, load_config, resolve_experiment_name  # noqa: E402

# Fails loudly if features.py and schema.py have drifted apart.
features.assert_feature_contract_consistent()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets
# MAGIC
# MAGIC Only the fields worth changing during a demo get a widget. Everything else
# MAGIC is edited in `src/fraud_common/config.py` or set via a `FRAUD_*`
# MAGIC environment variable.

# COMMAND ----------

WIDGET_FIELDS = [
    "catalog",
    "schema",
    "volume",
    "input_path",
    "experiment_name",
    "model_name",
    "model_uri_override",
    "endpoint_name",
    "random_seed",
    "train_negative_sample_fraction",
    "operational_threshold",
    # Business assumptions for notebook 08. Exposed as widgets because they are
    # the numbers a stakeholder will want changed live during a demo -- and
    # every figure the impact dashboard shows moves when they do.
    "impact_prevention_rate",
    "impact_cost_per_review",
    "impact_customer_friction_cost",
    "impact_analyst_daily_capacity",
    "impact_currency",
    "impact_evaluation_split",
    # Operational replay (notebook 10).
    "ops_evaluation_split",
    "ops_time_grain",
    "ops_analyst_headcount",
    "ops_threshold_override",
]


def _dbutils():
    """Return dbutils if we are inside Databricks, else None."""
    try:
        return dbutils  # noqa: F821  (injected by the Databricks runtime)
    except NameError:
        return None


def _collect_widget_overrides() -> dict:
    db = _dbutils()
    if db is None:
        return {}
    defaults = Config()
    overrides = {}
    for field in WIDGET_FIELDS:
        default_value = getattr(defaults, field)
        default_str = "" if default_value is None else str(default_value)
        try:
            db.widgets.text(field, default_str, field)
            overrides[field] = db.widgets.get(field)
        except Exception as exc:  # widget API unavailable (e.g. some job runs)
            print(f"[config] skipping widget {field!r}: {type(exc).__name__}")
    return overrides


CFG: Config = load_config(_collect_widget_overrides())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Runtime helpers

# COMMAND ----------

def get_spark():
    """Return the active SparkSession, or None outside Databricks."""
    try:
        return spark  # noqa: F821
    except NameError:
        try:
            from pyspark.sql import SparkSession
            return SparkSession.getActiveSession()
        except Exception:
            return None


def current_user(spark_session=None) -> str:
    """Workspace identity, read from the session -- never hard-coded."""
    session = spark_session or get_spark()
    if session is None:
        return ""
    try:
        return session.sql("SELECT current_user()").collect()[0][0]
    except Exception as exc:
        print(f"[config] current_user() unavailable: {type(exc).__name__}")
        return ""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(stage: str, message: str) -> None:
    print(f"[{utc_now_iso()}] [{stage}] {message}")


def ensure_uc_objects(spark_session=None) -> dict:
    """Create the schema and the raw volume if they do not exist.

    Deliberately does **not** attempt `CREATE CATALOG`: on Free Edition the
    `workspace` catalog already exists and the account usually lacks the
    privilege, so trying would fail noisily for no benefit.
    """
    session = spark_session or get_spark()
    if session is None:
        raise RuntimeError("no SparkSession -- run this inside Databricks")

    outcome = {"schema": None, "volume": None}
    try:
        session.sql(f"CREATE SCHEMA IF NOT EXISTS {CFG.full_schema}")
        outcome["schema"] = f"ok: {CFG.full_schema}"
    except Exception as exc:
        outcome["schema"] = f"FAILED: {type(exc).__name__}: {exc}"
        raise

    try:
        session.sql(
            f"CREATE VOLUME IF NOT EXISTS {CFG.full_schema}.{CFG.volume}"
        )
        outcome["volume"] = f"ok: {CFG.volume_root}"
    except Exception as exc:
        # A missing volume only blocks the CSV upload and the evidence writes,
        # not the Delta tables, so report rather than abort.
        outcome["volume"] = f"FAILED: {type(exc).__name__}: {exc}"

    for key, value in outcome.items():
        log("setup", f"{key}: {value}")
    return outcome


def input_file_exists(path: str = None) -> bool:
    """True when the PaySim CSV is readable at the configured path."""
    target = path or CFG.input_path
    db = _dbutils()
    if db is not None:
        try:
            db.fs.ls(target)
            return True
        except Exception:
            return False
    return os.path.exists(target)


def require_input_file(path: str = None) -> str:
    target = path or CFG.input_path
    if not input_file_exists(target):
        raise FileNotFoundError(
            f"PaySim CSV not found at {target!r}.\n"
            "Databricks Free Edition blocks outbound network access, so the "
            "file must be uploaded by hand:\n"
            "  1. Download 'PaySim1' from "
            "https://www.kaggle.com/datasets/ealaxi/paysim1\n"
            "  2. Unzip it (PS_20174392719_1491204439457_log.csv, ~470 MB)\n"
            f"  3. Catalog > {CFG.catalog} > {CFG.schema} > Volumes > "
            f"{CFG.volume} > Upload, into a 'paysim' folder\n"
            f"  4. Re-run with input_path = {CFG.input_path!r} (or set the "
            "`input_path` widget to wherever you put it)."
        )
    return target


def resolved_experiment_name(spark_session=None) -> str:
    return resolve_experiment_name(CFG, current_user(spark_session))


def table_exists(name: str, spark_session=None) -> bool:
    session = spark_session or get_spark()
    if session is None:
        return False
    try:
        return session.catalog.tableExists(name)
    except Exception:
        return False


def write_json_to_volume(relative_name: str, payload: dict) -> str:
    """Write sanitised JSON evidence into the volume. Never call with secrets.

    Everything passed through here is run through `serving.sanitize_evidence`
    first, so an accidental token in a nested error message is redacted rather
    than persisted.
    """
    safe = serving.sanitize_evidence(payload)
    target_dir = CFG.evidence_dir
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, relative_name)
    with open(target, "w") as handle:
        json.dump(safe, handle, indent=2, default=str)
    log("evidence", f"wrote {target}")
    return target


# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolved configuration

# COMMAND ----------

print(CFG.describe())
print()
print(f"  {'bronze_table':<32} = {CFG.bronze_table}")
print(f"  {'silver_table':<32} = {CFG.silver_table}")
print(f"  {'gold_features_table':<32} = {CFG.gold_features_table}")
print(f"  {'served features':<32} = {len(schema.served_features())} columns")
print(f"  {'excluded (leakage/identifier)':<32} = {schema.leakage_columns()}")
