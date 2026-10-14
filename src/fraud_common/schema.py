"""PaySim source schema, data-quality rules and the online feature contract.

This module is the single source of truth for *what a column means* and
*when its value becomes known*. The leakage tests in ``tests/`` assert against
these declarations rather than against notebook code, so the contract cannot
drift away from the implementation without a test failing.

PaySim reference (Lopez-Rojas et al., 2016; Kaggle: ealaxi/paysim1):
  ~6.36 M rows, 11 columns, 1 `step` == 1 simulated hour, 744 steps == 30 days.
  Note the source file's inconsistent spelling: `oldbalanceOrg` (no `i`) but
  `newbalanceOrig` (with `i`). Reproduced verbatim below on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# 1. Source contract
# --------------------------------------------------------------------------- #

SOURCE_COLUMNS: List[str] = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

# Spark DDL string -> used for an explicit read schema (never inferSchema:
# inference triggers a full extra pass over 6.3 M rows and can silently retype
# a column between runs).
SOURCE_SCHEMA_DDL: str = (
    "step INT, "
    "type STRING, "
    "amount DOUBLE, "
    "nameOrig STRING, "
    "oldbalanceOrg DOUBLE, "
    "newbalanceOrig DOUBLE, "
    "nameDest STRING, "
    "oldbalanceDest DOUBLE, "
    "newbalanceDest DOUBLE, "
    "isFraud INT, "
    "isFlaggedFraud INT"
)

# Pandas dtypes for the same contract (local tests / small-sample debugging).
SOURCE_PANDAS_DTYPES: Dict[str, str] = {
    "step": "int32",
    "type": "string",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

# The five transaction categories PaySim emits. Anything else is a reject.
ALLOWED_TRANSACTION_TYPES: List[str] = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]

# Fraud in PaySim only ever occurs on these two types. Kept as documentation
# and as a data-quality *observation* -- never used to filter training data,
# which would bake a simulator artefact into the model.
FRAUD_CAPABLE_TYPES: List[str] = ["CASH_OUT", "TRANSFER"]

TARGET_COLUMN: str = "isFraud"
ALLOWED_TARGET_VALUES: List[int] = [0, 1]

TIME_COLUMN: str = "step"
STEPS_PER_DAY: int = 24
EXPECTED_MAX_STEP: int = 744           # 30 days of hourly steps
EXPECTED_SOURCE_ROWS: int = 6_362_620  # published row count; asserted loosely

# PaySim ships no wall-clock timestamps, only the `step` counter. Silver anchors
# steps to this synthetic epoch purely so dashboards can plot a real time axis.
# It is a presentation device -- no claim is made that these are real dates.
# Plain `yyyy-MM-dd HH:mm:ss` -- an ISO string with a trailing `Z` is parsed
# inconsistently across Spark versions and ANSI settings.
SIMULATION_START_TS: str = "2024-01-01 00:00:00"

#: The `step` -> wall-clock derivation, written once. Notebook 02 applies this
#: to Silver; the operational replay re-applies it to the scored table, which
#: carries `step` but not the derived time columns. Both must produce the same
#: instant for the same step or the time axis of one dashboard would silently
#: disagree with the other.
STEP_TO_TIMESTAMP_INTERVAL: str = "MAKE_INTERVAL(0, 0, 0, 0, step - 1, 0, 0)"


def step_to_timestamp_sql(start_ts: str = None) -> str:
    """SQL expression turning the `step` hour counter into a timestamp."""
    return (f"TIMESTAMP('{start_ts or SIMULATION_START_TS}') + "
            f"{STEP_TO_TIMESTAMP_INTERVAL}")


# --------------------------------------------------------------------------- #
# 2. Feature contract
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FeatureSpec:
    """One column of the Gold feature table.

    availability
        ``pre_transaction``  -- knowable before the payment is authorised, so
                                legitimate input to a real-time decision.
        ``post_transaction`` -- only settled after the payment completes.
        ``label``            -- the target or a target-derived artefact.
        ``identifier``       -- account key / lineage; never scored.
    served
        True  -> part of the model signature sent to the serving endpoint.
        False -> retained in Gold for analysis, monitoring or joins only.
    """

    name: str
    dtype: str                 # "double" | "int" | "string"
    availability: str          # pre_transaction | post_transaction | label | identifier
    served: bool
    description: str
    derivation: Optional[str] = None


PRE_TRANSACTION = "pre_transaction"
POST_TRANSACTION = "post_transaction"
LABEL = "label"
IDENTIFIER = "identifier"

GOLD_COLUMNS: List[FeatureSpec] = [
    # ---------------- served model inputs (pre-transaction only) ----------- #
    FeatureSpec(
        "amount", "double", PRE_TRANSACTION, True,
        "Requested transaction amount in local currency.",
        "source column, verbatim",
    ),
    FeatureSpec(
        "type", "string", PRE_TRANSACTION, True,
        "Transaction category; one of ALLOWED_TRANSACTION_TYPES.",
        "source column, verbatim (one-hot encoded inside the pipeline)",
    ),
    FeatureSpec(
        "old_balance_orig", "double", PRE_TRANSACTION, True,
        "Originating account balance BEFORE the transaction.",
        "rename of source `oldbalanceOrg`",
    ),
    FeatureSpec(
        "old_balance_dest", "double", PRE_TRANSACTION, True,
        "Destination account balance BEFORE the transaction.",
        "rename of source `oldbalanceDest`",
    ),
    FeatureSpec(
        "hour_of_day", "int", PRE_TRANSACTION, True,
        "Simulated hour of day, 0-23. Captures the nocturnal fraud pattern.",
        "(step - 1) % 24",
    ),
    FeatureSpec(
        "is_night", "int", PRE_TRANSACTION, True,
        "1 when hour_of_day is in [0, 6); PaySim fraud concentrates overnight.",
        "derived from hour_of_day",
    ),
    FeatureSpec(
        "day_of_week", "int", PRE_TRANSACTION, True,
        "Simulated day of week, 0-6, from the 30-day step sequence.",
        "((step - 1) // 24) % 7",
    ),
    FeatureSpec(
        "amount_to_balance_ratio", "double", PRE_TRANSACTION, True,
        "amount / old_balance_orig; near 1.0 flags account-drain behaviour.",
        "amount / (old_balance_orig + 1.0), clipped for stability",
    ),
    FeatureSpec(
        "drains_origin_account", "int", PRE_TRANSACTION, True,
        "1 when the amount would empty the origin account (within tolerance).",
        "amount >= old_balance_orig - 0.01 AND old_balance_orig > 0",
    ),
    FeatureSpec(
        "exceeds_origin_balance", "int", PRE_TRANSACTION, True,
        "1 when the requested amount is larger than the available balance.",
        "amount > old_balance_orig",
    ),
    FeatureSpec(
        "dest_is_merchant", "int", PRE_TRANSACTION, True,
        "1 when the destination is a merchant account (PaySim 'M' prefix). "
        "An account-type attribute, not the identifier itself.",
        "nameDest LIKE 'M%'",
    ),
    FeatureSpec(
        "orig_balance_is_zero", "int", PRE_TRANSACTION, True,
        "1 when the origin account starts at zero -- a mule-account signal.",
        "old_balance_orig == 0",
    ),
    FeatureSpec(
        "dest_balance_is_zero", "int", PRE_TRANSACTION, True,
        "1 when the destination account starts at zero.",
        "old_balance_dest == 0",
    ),
    FeatureSpec(
        "amount_log", "double", PRE_TRANSACTION, True,
        "log1p(amount); linearises the heavy-tailed amount distribution for "
        "the logistic-regression baseline.",
        "ln(1 + amount)",
    ),

    # ---------------- retained but NOT served ------------------------------ #
    FeatureSpec(
        "step", "int", PRE_TRANSACTION, False,
        "Simulated hour index. Governs the chronological split and the "
        "monitoring windows; withheld from the model so it cannot memorise "
        "simulator time.",
        "source column, verbatim",
    ),
    FeatureSpec(
        "transaction_id", "string", IDENTIFIER, False,
        "Deterministic surrogate key for joining scores back to rows.",
        "sha2 of the natural key, see 02_transform_silver",
    ),
    FeatureSpec(
        "name_orig_hash", "string", IDENTIFIER, False,
        "Salt-free SHA-256 of nameOrig. Analysis/joins only -- raw customer "
        "identifiers never leave the lakehouse.",
        "sha2(nameOrig, 256)",
    ),
    FeatureSpec(
        "name_dest_hash", "string", IDENTIFIER, False,
        "SHA-256 of nameDest. Analysis/joins only.",
        "sha2(nameDest, 256)",
    ),
    FeatureSpec(
        "new_balance_orig", "double", POST_TRANSACTION, False,
        "Origin balance AFTER settlement. LEAKAGE for pre-transaction scoring.",
        "rename of source `newbalanceOrig`",
    ),
    FeatureSpec(
        "new_balance_dest", "double", POST_TRANSACTION, False,
        "Destination balance AFTER settlement. LEAKAGE for pre-transaction "
        "scoring.",
        "rename of source `newbalanceDest`",
    ),
    FeatureSpec(
        "is_flagged_fraud", "int", LABEL, False,
        "Output of PaySim's own >200k transfer rule. A downstream control's "
        "decision, not an input -- excluded to avoid label contamination.",
        "rename of source `isFlaggedFraud`",
    ),
    FeatureSpec(
        "is_fraud", "int", LABEL, False,
        "Binary target.",
        "rename of source `isFraud`",
    ),
]

_BY_NAME: Dict[str, FeatureSpec] = {f.name: f for f in GOLD_COLUMNS}


def get_spec(name: str) -> FeatureSpec:
    return _BY_NAME[name]


def served_features() -> List[str]:
    """Ordered model-input column names. This *is* the online contract."""
    return [f.name for f in GOLD_COLUMNS if f.served]


def numeric_features() -> List[str]:
    return [f.name for f in GOLD_COLUMNS if f.served and f.dtype in ("double", "int")]


def categorical_features() -> List[str]:
    return [f.name for f in GOLD_COLUMNS if f.served and f.dtype == "string"]


def leakage_columns() -> List[str]:
    """Columns that must never appear in the served feature list."""
    return [
        f.name for f in GOLD_COLUMNS
        if f.availability in (POST_TRANSACTION, LABEL, IDENTIFIER)
    ]


def gold_column_names() -> List[str]:
    return [f.name for f in GOLD_COLUMNS]


def served_dtypes() -> Dict[str, str]:
    return {f.name: f.dtype for f in GOLD_COLUMNS if f.served}


def assert_no_leakage(feature_names: Sequence[str]) -> None:
    """Raise if a proposed feature list contains a forbidden column.

    Called by 03_build_gold_features and 04_train_register_model before the
    frame is handed to the estimator -- a runtime guard, not just a test.
    """
    forbidden = set(leakage_columns())
    # Also catch the raw source spellings in case someone bypasses the rename.
    forbidden |= {
        "isFraud", "isFlaggedFraud",
        "newbalanceOrig", "newbalanceDest",
        "nameOrig", "nameDest",
    }
    offenders = sorted(set(feature_names) & forbidden)
    if offenders:
        raise ValueError(
            "target leakage: these columns must not be model inputs: "
            f"{offenders}. See fraud_common.schema.GOLD_COLUMNS for the "
            "availability rationale of each."
        )


# --------------------------------------------------------------------------- #
# 3. Data-quality rules applied at Silver
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class QualityRule:
    name: str
    # SQL boolean expression that is TRUE for a *valid* record. Evaluated
    # against the renamed Silver columns.
    expression: str
    severity: str   # "reject" -> row is quarantined; "warn" -> counted only
    description: str


QUALITY_RULES: List[QualityRule] = [
    QualityRule(
        "step_present_and_positive",
        "step IS NOT NULL AND step >= 1",
        "reject",
        "step is the time axis; a null breaks the chronological split.",
    ),
    QualityRule(
        "type_in_allowed_set",
        "type IN ('CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER')",
        "reject",
        "Unknown transaction category -- schema drift at the source.",
    ),
    QualityRule(
        "amount_present",
        "amount IS NOT NULL",
        "reject",
        "A transaction with no amount cannot be scored.",
    ),
    QualityRule(
        "amount_non_negative",
        "amount >= 0",
        "reject",
        "Negative amounts are not representable in PaySim.",
    ),
    QualityRule(
        "balances_non_negative",
        "old_balance_orig >= 0 AND new_balance_orig >= 0 "
        "AND old_balance_dest >= 0 AND new_balance_dest >= 0",
        "reject",
        "Negative balances indicate a corrupt or misparsed row.",
    ),
    QualityRule(
        "balances_present",
        "old_balance_orig IS NOT NULL AND old_balance_dest IS NOT NULL",
        "reject",
        "Pre-transaction balances are served features; nulls cannot be scored.",
    ),
    QualityRule(
        "target_in_allowed_values",
        "is_fraud IN (0, 1)",
        "reject",
        "Target must be binary; anything else corrupts training.",
    ),
    QualityRule(
        "flag_in_allowed_values",
        "is_flagged_fraud IN (0, 1)",
        "reject",
        "isFlaggedFraud must be binary.",
    ),
    QualityRule(
        "account_names_present",
        "name_orig IS NOT NULL AND name_dest IS NOT NULL",
        "reject",
        "Account identifiers are needed for the surrogate key and hashing.",
    ),
    QualityRule(
        "amount_zero_is_suspicious",
        "amount > 0",
        "warn",
        "PaySim contains a small number of zero-amount rows; kept but counted.",
    ),
    QualityRule(
        "step_within_expected_horizon",
        f"step <= {EXPECTED_MAX_STEP}",
        "warn",
        "Steps beyond 744 would mean the simulation horizon changed.",
    ),
    QualityRule(
        "origin_is_not_merchant",
        "name_orig NOT LIKE 'M%'",
        "warn",
        "PaySim never originates from a merchant; a violation signals a "
        "generator change.",
    ),
]


def reject_rules() -> List[QualityRule]:
    return [r for r in QUALITY_RULES if r.severity == "reject"]


def warn_rules() -> List[QualityRule]:
    return [r for r in QUALITY_RULES if r.severity == "warn"]


def silver_valid_expression() -> str:
    """Single SQL predicate that is TRUE only for fully valid rows."""
    return " AND ".join(f"({r.expression})" for r in reject_rules())


# --------------------------------------------------------------------------- #
# 4. Serving payload contract
# --------------------------------------------------------------------------- #

# A syntactically valid example request, used in the README, in
# 05_deploy_and_test_endpoint and in tests. Values are fabricated, not sampled
# from the dataset.
EXAMPLE_SERVING_RECORD: Dict[str, object] = {
    "amount": 181.0,
    "type": "TRANSFER",
    "old_balance_orig": 181.0,
    "old_balance_dest": 0.0,
    "hour_of_day": 1,
    "is_night": 1,
    "day_of_week": 0,
    "amount_to_balance_ratio": 0.9945054945054945,
    "drains_origin_account": 1,
    "exceeds_origin_balance": 0,
    "dest_is_merchant": 0,
    "orig_balance_is_zero": 0,
    "dest_balance_is_zero": 1,
    "amount_log": 5.204006687076795,
}
