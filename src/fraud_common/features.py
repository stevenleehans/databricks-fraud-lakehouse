"""Feature derivations, declared once and reused by Spark and pandas.

Each Gold feature is defined as a SQL expression over Silver columns. The
Spark notebook feeds these to ``selectExpr``; the pandas implementation below
mirrors them for local tests and for building serving payloads off-cluster.
Keeping one declaration prevents the classic PoC failure where the training
feature and the online feature quietly diverge.
"""

from __future__ import annotations

from typing import Dict, List

from .schema import PRE_TRANSACTION, GOLD_COLUMNS

# --------------------------------------------------------------------------- #
# Bronze (raw source spelling) -> Silver (snake_case) rename map.
# --------------------------------------------------------------------------- #
SOURCE_TO_SILVER: Dict[str, str] = {
    "step": "step",
    "type": "type",
    "amount": "amount",
    "nameOrig": "name_orig",
    "oldbalanceOrg": "old_balance_orig",
    "newbalanceOrig": "new_balance_orig",
    "nameDest": "name_dest",
    "oldbalanceDest": "old_balance_dest",
    "newbalanceDest": "new_balance_dest",
    "isFraud": "is_fraud",
    "isFlaggedFraud": "is_flagged_fraud",
}

# Silver columns a *served* feature is permitted to read. Anything outside this
# set would be post-transaction knowledge or an identifier.
PRE_TRANSACTION_SILVER_INPUTS: List[str] = [
    "step",
    "type",
    "amount",
    "old_balance_orig",
    "old_balance_dest",
    "name_dest",   # only via the LIKE 'M%' merchant test, never emitted raw
]

# Guard against division by zero on empty accounts without dropping the row.
BALANCE_EPSILON = 1.0
# PaySim balances carry cent-level rounding noise.
BALANCE_TOLERANCE = 0.01
# Ratio ceiling: transfers out of a zero balance would otherwise be unbounded.
MAX_AMOUNT_RATIO = 100.0

# --------------------------------------------------------------------------- #
# Served feature expressions (Silver -> Gold). Order matters: it defines the
# model signature column order.
# --------------------------------------------------------------------------- #
FEATURE_SQL: Dict[str, str] = {
    "amount": "amount",
    "type": "type",
    "old_balance_orig": "old_balance_orig",
    "old_balance_dest": "old_balance_dest",
    "hour_of_day": "CAST((step - 1) % 24 AS INT)",
    "is_night": "CAST(CASE WHEN (step - 1) % 24 < 6 THEN 1 ELSE 0 END AS INT)",
    "day_of_week": "CAST(((step - 1) DIV 24) % 7 AS INT)",
    "amount_to_balance_ratio": (
        f"LEAST(amount / (old_balance_orig + {BALANCE_EPSILON}), {MAX_AMOUNT_RATIO})"
    ),
    "drains_origin_account": (
        "CAST(CASE WHEN old_balance_orig > 0 "
        f"AND amount >= old_balance_orig - {BALANCE_TOLERANCE} "
        "THEN 1 ELSE 0 END AS INT)"
    ),
    "exceeds_origin_balance": (
        "CAST(CASE WHEN amount > old_balance_orig THEN 1 ELSE 0 END AS INT)"
    ),
    "dest_is_merchant": (
        "CAST(CASE WHEN name_dest LIKE 'M%' THEN 1 ELSE 0 END AS INT)"
    ),
    "orig_balance_is_zero": (
        "CAST(CASE WHEN old_balance_orig = 0 THEN 1 ELSE 0 END AS INT)"
    ),
    "dest_balance_is_zero": (
        "CAST(CASE WHEN old_balance_dest = 0 THEN 1 ELSE 0 END AS INT)"
    ),
    "amount_log": "LN(1 + amount)",
}

# Non-served Gold columns carried for lineage, splitting and monitoring.
PASSTHROUGH_SQL: Dict[str, str] = {
    "transaction_id": "transaction_id",
    "step": "step",
    "name_orig_hash": "SHA2(name_orig, 256)",
    "name_dest_hash": "SHA2(name_dest, 256)",
    "new_balance_orig": "new_balance_orig",
    "new_balance_dest": "new_balance_dest",
    "is_flagged_fraud": "is_flagged_fraud",
    "is_fraud": "is_fraud",
}


def gold_select_expressions() -> List[str]:
    """``selectExpr`` list building the whole Gold table in one projection."""
    exprs = [f"{sql} AS {name}" for name, sql in FEATURE_SQL.items()]
    exprs += [f"{sql} AS {name}" for name, sql in PASSTHROUGH_SQL.items()
              if name not in FEATURE_SQL]
    return exprs


def derive_features_pandas(df):
    """Pandas mirror of ``FEATURE_SQL``.

    Accepts a frame with Silver column names, returns a new frame containing
    exactly the served features in signature order. Used by the tests and by
    ``05_deploy_and_test_endpoint`` to build request payloads from real rows.
    """
    import numpy as np
    import pandas as pd

    missing = [c for c in PRE_TRANSACTION_SILVER_INPUTS if c not in df.columns]
    if missing:
        raise KeyError(f"missing Silver input columns: {missing}")

    step = df["step"].astype("int64")
    amount = df["amount"].astype("float64")
    old_orig = df["old_balance_orig"].astype("float64")
    old_dest = df["old_balance_dest"].astype("float64")

    hour_of_day = ((step - 1) % 24).astype("int32")

    out = pd.DataFrame(index=df.index)
    out["amount"] = amount
    out["type"] = df["type"].astype("string")
    out["old_balance_orig"] = old_orig
    out["old_balance_dest"] = old_dest
    out["hour_of_day"] = hour_of_day
    out["is_night"] = (hour_of_day < 6).astype("int32")
    out["day_of_week"] = (((step - 1) // 24) % 7).astype("int32")
    out["amount_to_balance_ratio"] = np.minimum(
        amount / (old_orig + BALANCE_EPSILON), MAX_AMOUNT_RATIO
    )
    out["drains_origin_account"] = (
        (old_orig > 0) & (amount >= old_orig - BALANCE_TOLERANCE)
    ).astype("int32")
    out["exceeds_origin_balance"] = (amount > old_orig).astype("int32")
    out["dest_is_merchant"] = (
        df["name_dest"].astype("string").str.startswith("M").fillna(False)
    ).astype("int32")
    out["orig_balance_is_zero"] = (old_orig == 0).astype("int32")
    out["dest_balance_is_zero"] = (old_dest == 0).astype("int32")
    out["amount_log"] = np.log1p(amount)

    return out[list(FEATURE_SQL.keys())]


def signature_input_columns() -> List[str]:
    """Model-signature column order. Must equal ``schema.served_features()``."""
    return list(FEATURE_SQL.keys())


def _served_specs():
    return [f for f in GOLD_COLUMNS if f.served]


def assert_feature_contract_consistent() -> None:
    """Cross-check features.py against schema.py.

    Runs at notebook import time so a half-finished edit fails immediately
    rather than producing a Gold table the model signature cannot consume.
    """
    declared = [f.name for f in _served_specs()]
    implemented = signature_input_columns()
    if declared != implemented:
        raise AssertionError(
            "served feature contract mismatch.\n"
            f"  schema.served_features(): {declared}\n"
            f"  features.FEATURE_SQL keys: {implemented}"
        )
    for spec in _served_specs():
        if spec.availability != PRE_TRANSACTION:
            raise AssertionError(
                f"served feature {spec.name!r} is marked {spec.availability!r}; "
                "only pre_transaction columns may be served."
            )
