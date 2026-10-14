"""Central configuration for the PaySim fraud lakehouse PoC.

Every tunable lives here. Notebooks import ``load_config()`` (via ``00_config``)
and never re-declare names, so a single edit repoints the whole pipeline.

Design notes (one-day MVP on Databricks Free Edition):
  * No workspace URL, username, token or Kaggle credential is stored here.
    ``experiment_name`` deliberately defaults to ``None`` and is resolved at
    runtime from ``current_user()`` so the repo stays portable between accounts.
  * Every field can be overridden by an environment variable (``FRAUD_<FIELD>``)
    or a Databricks notebook widget, so the same code runs in a job, an
    interactive notebook and a local pytest session.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

ENV_PREFIX = "FRAUD_"


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------ #
    # Unity Catalog placement
    # ------------------------------------------------------------------ #
    # Free Edition provisions a catalog literally named `workspace`; change
    # this if your metastore uses a different default catalog.
    catalog: str = "workspace"
    schema: str = "fraud_poc"
    volume: str = "raw"

    # Path of the manually-uploaded Kaggle CSV. Free Edition blocks outbound
    # network access from clusters, so this file MUST be uploaded by hand.
    # Local pytest runs may point this at any readable path.
    input_path: str = (
        "/Volumes/workspace/fraud_poc/raw/paysim/PS_20174392719_1491204439457_log.csv"
    )

    # ------------------------------------------------------------------ #
    # MLflow / model registry
    # ------------------------------------------------------------------ #
    # None -> resolved at runtime to /Users/<current_user>/paysim_fraud_poc
    experiment_name: Optional[str] = None
    model_name: str = "paysim_fraud_clf"          # unqualified; UC name is derived
    champion_alias: str = "Champion"

    # Escape hatch for loading the model when the normal routes fail.
    # Databricks Free Edition serverless cannot read artifacts back from the
    # Unity Catalog model location (S3 HeadObject 400), even though logging and
    # registering succeed. Setting this to an MLflow 3 logged-model URI, e.g.
    # "models:/m-<uuid>", bypasses UC storage and reads from the experiment's
    # DBFS tracking store instead. Tried first when set.
    model_uri_override: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Serving
    # ------------------------------------------------------------------ #
    endpoint_name: str = "paysim-fraud-poc"
    endpoint_workload_size: str = "Small"
    endpoint_scale_to_zero: bool = True
    endpoint_ready_timeout_minutes: int = 30

    # ------------------------------------------------------------------ #
    # Reproducibility / sampling
    # ------------------------------------------------------------------ #
    random_seed: int = 42
    # Fraction of *legitimate* training-period rows kept. All fraud rows are
    # always retained (see fraud_common.splits.sample_training_frame).
    train_negative_sample_fraction: float = 0.10
    # Optional cap on evaluation rows if Free Edition compute is strained.
    # None -> evaluate validation/test at natural prevalence (preferred).
    eval_max_rows: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Chronological split (fractions of the observed `step` range)
    # ------------------------------------------------------------------ #
    train_step_fraction: float = 0.70
    valid_step_fraction: float = 0.15
    # test = remainder (0.15)

    # ------------------------------------------------------------------ #
    # Operating point
    # ------------------------------------------------------------------ #
    # Score threshold used for precision/recall/FPR reporting and for the
    # predicted-positive rate tracked in monitoring. Chosen on VALIDATION only.
    operational_threshold: float = 0.50
    # If set, the threshold is instead chosen as the highest-recall point whose
    # false-positive rate stays under this budget on the validation split.
    target_max_fpr: Optional[float] = 0.01

    # ------------------------------------------------------------------ #
    # Drift thresholds (Population Stability Index convention)
    # ------------------------------------------------------------------ #
    psi_amber_threshold: float = 0.10     # 0.10-0.25 -> investigate
    psi_red_threshold: float = 0.25       # >0.25     -> significant shift
    psi_bins: int = 10
    # Guard against log(0) when a bin empties out in a monitoring window.
    psi_epsilon: float = 1e-6
    # Absolute change in missing-rate that trips amber/red.
    missing_rate_amber_threshold: float = 0.01
    missing_rate_red_threshold: float = 0.05
    # Number of equal-width `step` windows carved out of the monitoring period.
    monitoring_window_count: int = 6
    # Fraud labels arrive late (chargebacks, investigations). The most recent
    # N windows are treated as unlabelled so the dashboard has to show the
    # "performance unknown" state instead of pretending it is measurable.
    label_delay_windows: int = 2

    # ------------------------------------------------------------------ #
    # Business impact simulation (notebook 08)
    #
    # These are ASSUMPTIONS supplied by the business, not properties of the
    # data. They drive the simulated savings figures, so every one of them is
    # written into the output tables alongside the numbers it produced.
    # ------------------------------------------------------------------ #
    # Share of a flagged fraud's value actually recovered. Never 1.0: reviews
    # take time, some funds have already moved, some blocks get overridden.
    impact_prevention_rate: float = 0.80
    # Fully-loaded cost of one manual review. Charged on every flagged row,
    # true and false positives alike -- the queue cannot tell them apart.
    impact_cost_per_review: float = 5.00
    # Cost of wrongly interrupting a genuine customer, charged on false
    # positives IN ADDITION to the review cost.
    impact_customer_friction_cost: float = 10.00
    # Alerts one analyst clears per working day; turns alert volume into a
    # staffing number.
    impact_analyst_daily_capacity: int = 200
    impact_currency: str = "USD"
    # Threshold sweep bounds. 0 flags everything and 1 flags nothing, so
    # neither endpoint is a candidate operating point.
    impact_threshold_start: float = 0.05
    impact_threshold_stop: float = 0.95
    impact_threshold_step: float = 0.05
    # Which scored split the simulation runs on. `test` is the default because
    # it is the only split that is both held out and never used to choose the
    # operating threshold -- simulating on train would quote savings the model
    # has already seen the answers for.
    #   test | valid | holdout (valid+test) | all
    impact_evaluation_split: str = "test"

    # ------------------------------------------------------------------ #
    # Operational replay (notebook 10)
    #
    # Same model, same money, different question: not "which threshold?" but
    # "we shipped it -- what did each hour of running it look like?"
    # ------------------------------------------------------------------ #
    # `holdout` (valid + test) is the whole post-training period, which is the
    # honest analogue of "everything since go-live". `test` alone is only a few
    # days and makes a thin time series.
    ops_evaluation_split: str = "holdout"
    # hour | day. Hourly is the default because PaySim's `step` IS an hour, so
    # an hourly series is the native grain and needs no interpolation.
    ops_time_grain: str = "hour"
    # Analysts staffed on the alert queue. Sets the capacity line that alert
    # volume is measured against -- a queue nobody can clear is an operational
    # failure even when the model is right.
    ops_analyst_headcount: int = 3
    # None -> replay at the threshold the impact simulator recommended, falling
    # back to `operational_threshold`. Set a float to pin it.
    ops_threshold_override: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    # Re-running ingest with the same batch_id is a no-op (idempotence guard).
    overwrite_existing_batch: bool = False

    # ------------------------------------------------------------------ #
    # Derived names -- keep table naming in exactly one place
    # ------------------------------------------------------------------ #
    @property
    def full_schema(self) -> str:
        return f"{self.catalog}.{self.schema}"

    @property
    def volume_root(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"

    @property
    def evidence_dir(self) -> str:
        """Where sanitised (never secret-bearing) run evidence is written."""
        return f"{self.volume_root}/evidence"

    @property
    def uc_model_name(self) -> str:
        """Three-level Unity Catalog model name."""
        return f"{self.catalog}.{self.schema}.{self.model_name}"

    def table(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"

    # Canonical table names ------------------------------------------------
    @property
    def bronze_table(self) -> str:
        return self.table("bronze_transactions")

    @property
    def bronze_audit_table(self) -> str:
        return self.table("bronze_ingest_audit")

    @property
    def bronze_malformed_table(self) -> str:
        return self.table("bronze_malformed_records")

    @property
    def silver_table(self) -> str:
        return self.table("silver_transactions")

    @property
    def silver_rejects_table(self) -> str:
        return self.table("silver_rejected_records")

    @property
    def silver_dq_table(self) -> str:
        return self.table("silver_quality_results")

    @property
    def gold_features_table(self) -> str:
        return self.table("gold_transaction_features")

    @property
    def feature_contract_table(self) -> str:
        return self.table("gold_feature_contract")

    @property
    def scored_predictions_table(self) -> str:
        """Raw batch scores written by 04 while the model is still in memory.

        Exists because the notebook runtime's MLflow 2.x client cannot load a
        model logged by the MLflow 3 tracking server, so notebook 06 cannot
        re-score for itself on Free Edition. See the README.
        """
        return self.table("scored_predictions")

    @property
    def inference_log_table(self) -> str:
        return self.table("inference_log")

    @property
    def reference_stats_table(self) -> str:
        return self.table("monitoring_reference_stats")

    @property
    def feature_drift_table(self) -> str:
        return self.table("monitoring_feature_drift")

    @property
    def prediction_drift_table(self) -> str:
        return self.table("monitoring_prediction_drift")

    @property
    def score_histogram_table(self) -> str:
        return self.table("monitoring_score_histogram")

    @property
    def performance_table(self) -> str:
        return self.table("monitoring_performance")

    @property
    def window_dim_table(self) -> str:
        return self.table("monitoring_windows")

    @property
    def fraud_impact_simulation_table(self) -> str:
        """Gold: one row per simulated decision threshold."""
        return self.table("gold_fraud_impact_simulation")

    @property
    def fraud_impact_summary_table(self) -> str:
        """Gold: single-row executive summary at the recommended threshold."""
        return self.table("gold_fraud_business_impact_summary")

    @property
    def fraud_operations_timeseries_table(self) -> str:
        """Gold: one row per operating period at the frozen threshold."""
        return self.table("gold_fraud_operations_timeseries")

    @property
    def fraud_operations_profile_table(self) -> str:
        """Gold: average alert load by hour of day, for shift planning."""
        return self.table("gold_fraud_operations_hourly_profile")

    @property
    def fraud_operations_summary_table(self) -> str:
        """Gold: single-row roll-up of the whole replayed operating period."""
        return self.table("gold_fraud_operations_summary")

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        lines = ["Config:"]
        for k, v in sorted(self.to_dict().items()):
            lines.append(f"  {k:<32} = {v!r}")
        for k in ("full_schema", "uc_model_name", "volume_root"):
            lines.append(f"  {k:<32} = {getattr(self, k)!r}")
        return "\n".join(lines)


_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def _coerce(raw: str, target_type: Any, field_name: str) -> Any:
    """Turn a widget/env string into the dataclass field's type."""
    text = raw.strip()
    type_str = str(target_type)

    # Optional[...] fields accept an explicit empty/none sentinel.
    optional = "Optional" in type_str or "None" in type_str
    if optional and text.lower() in {"", "none", "null"}:
        return None
    if text == "" and not optional:
        raise ValueError(f"empty override for non-optional field {field_name!r}")

    if "bool" in type_str:
        low = text.lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"cannot parse boolean for {field_name!r}: {raw!r}")
    if "int" in type_str:
        return int(text)
    if "float" in type_str:
        return float(text)
    return text


def load_config(overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Build a Config from defaults <- environment <- explicit overrides.

    Environment variables use the ``FRAUD_`` prefix and the upper-cased field
    name, e.g. ``FRAUD_CATALOG=main`` or ``FRAUD_RANDOM_SEED=7``. Explicit
    ``overrides`` (typically Databricks widget values) win over everything.
    """
    values: Dict[str, Any] = {}
    field_types = {f.name: f.type for f in fields(Config)}

    for name, ftype in field_types.items():
        env_val = os.environ.get(ENV_PREFIX + name.upper())
        if env_val is not None:
            values[name] = _coerce(env_val, ftype, name)

    for name, val in (overrides or {}).items():
        if name not in field_types:
            raise KeyError(f"unknown config field: {name!r}")
        if val is None:
            values[name] = None
        elif isinstance(val, str):
            values[name] = _coerce(val, field_types[name], name)
        else:
            values[name] = val

    cfg = Config(**values)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    """Fail fast on configurations that would silently produce bad science."""
    if not 0.0 < cfg.train_step_fraction < 1.0:
        raise ValueError("train_step_fraction must be in (0, 1)")
    if not 0.0 < cfg.valid_step_fraction < 1.0:
        raise ValueError("valid_step_fraction must be in (0, 1)")
    if cfg.train_step_fraction + cfg.valid_step_fraction >= 1.0:
        raise ValueError(
            "train_step_fraction + valid_step_fraction must leave room for a "
            "test split (< 1.0)"
        )
    if not 0.0 < cfg.train_negative_sample_fraction <= 1.0:
        raise ValueError("train_negative_sample_fraction must be in (0, 1]")
    if not 0.0 <= cfg.operational_threshold <= 1.0:
        raise ValueError("operational_threshold must be in [0, 1]")
    if cfg.psi_amber_threshold >= cfg.psi_red_threshold:
        raise ValueError("psi_amber_threshold must be < psi_red_threshold")
    if cfg.psi_bins < 2:
        raise ValueError("psi_bins must be >= 2")
    if cfg.monitoring_window_count < 1:
        raise ValueError("monitoring_window_count must be >= 1")
    if cfg.target_max_fpr is not None and not 0.0 < cfg.target_max_fpr < 1.0:
        raise ValueError("target_max_fpr must be in (0, 1) when set")
    if not 0.0 <= cfg.impact_prevention_rate <= 1.0:
        raise ValueError("impact_prevention_rate must be in [0, 1]")
    if cfg.impact_cost_per_review < 0:
        raise ValueError("impact_cost_per_review must be >= 0")
    if cfg.impact_customer_friction_cost < 0:
        raise ValueError("impact_customer_friction_cost must be >= 0")
    if cfg.impact_analyst_daily_capacity <= 0:
        raise ValueError("impact_analyst_daily_capacity must be > 0")
    if not 0.0 <= cfg.impact_threshold_start <= cfg.impact_threshold_stop <= 1.0:
        raise ValueError(
            "impact thresholds must satisfy 0 <= start <= stop <= 1"
        )
    if cfg.impact_threshold_step <= 0:
        raise ValueError("impact_threshold_step must be > 0")
    if cfg.impact_evaluation_split not in {"test", "valid", "holdout", "all"}:
        raise ValueError(
            "impact_evaluation_split must be one of: test, valid, holdout, all"
        )
    if cfg.ops_evaluation_split not in {"test", "valid", "holdout", "all"}:
        raise ValueError(
            "ops_evaluation_split must be one of: test, valid, holdout, all"
        )
    if cfg.ops_time_grain not in {"hour", "day"}:
        raise ValueError("ops_time_grain must be 'hour' or 'day'")
    if cfg.ops_analyst_headcount <= 0:
        raise ValueError("ops_analyst_headcount must be > 0")
    if cfg.ops_threshold_override is not None and \
            not 0.0 <= cfg.ops_threshold_override <= 1.0:
        raise ValueError("ops_threshold_override must be in [0, 1] when set")


def resolve_experiment_name(cfg: Config, current_user: Optional[str]) -> str:
    """Resolve the MLflow experiment path without hard-coding a username.

    ``current_user`` comes from ``spark.sql("select current_user()")`` at
    runtime. Falls back to ``/Shared/`` when it is unavailable (e.g. local
    tests), which keeps the value deterministic for assertions.
    """
    if cfg.experiment_name:
        return cfg.experiment_name
    if current_user:
        return f"/Users/{current_user}/paysim_fraud_poc"
    return "/Shared/paysim_fraud_poc"
