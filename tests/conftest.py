"""Shared fixtures. Everything here runs without Spark, MLflow or Databricks."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def project_root() -> str:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def notebook_paths():
    src = os.path.join(PROJECT_ROOT, "src")
    return sorted(
        os.path.join(src, name)
        for name in os.listdir(src)
        if name.endswith(".py") and name[0].isdigit()
    )


@pytest.fixture(scope="session")
def synthetic_silver() -> pd.DataFrame:
    """A small PaySim-shaped frame with Silver column names.

    Deliberately imperfect: it contains zero balances, a drained account, a
    merchant destination and a handful of fraud rows, so the derivations are
    exercised on the edge cases that matter rather than on tidy averages.
    """
    rng = np.random.default_rng(20240101)
    n = 4_000
    steps = rng.integers(1, 745, size=n)
    types = rng.choice(
        ["PAYMENT", "CASH_OUT", "TRANSFER", "CASH_IN", "DEBIT"],
        size=n,
        p=[0.34, 0.35, 0.22, 0.07, 0.02],
    )
    amount = np.round(rng.lognormal(mean=8.0, sigma=1.6, size=n), 2)
    old_orig = np.round(rng.lognormal(mean=8.5, sigma=1.8, size=n), 2)
    old_orig[rng.random(n) < 0.25] = 0.0            # zero-balance origins
    old_dest = np.round(rng.lognormal(mean=8.0, sigma=2.0, size=n), 2)
    old_dest[rng.random(n) < 0.35] = 0.0

    # Fraud: rarer, later-drifting, and correlated with account drain.
    fraud_prob = 0.002 + 0.004 * (steps / 744.0)
    is_fraud = (rng.random(n) < fraud_prob).astype(int)
    drain_rows = is_fraud == 1
    amount[drain_rows] = old_orig[drain_rows]        # fraud empties the account

    name_dest = np.where(
        rng.random(n) < 0.30,
        np.array([f"M{i:09d}" for i in range(n)]),
        np.array([f"C{i:09d}" for i in range(n)]),
    )

    return pd.DataFrame({
        "transaction_id": [f"tx{i:07d}" for i in range(n)],
        "step": steps.astype(int),
        "type": types,
        "amount": amount,
        "name_orig": [f"C{i:09d}" for i in range(n, 2 * n)],
        "old_balance_orig": old_orig,
        "new_balance_orig": np.maximum(old_orig - amount, 0.0),
        "name_dest": name_dest,
        "old_balance_dest": old_dest,
        "new_balance_dest": old_dest + amount,
        "is_fraud": is_fraud,
        "is_flagged_fraud": ((amount > 200_000) & (is_fraud == 1)).astype(int),
    })


@pytest.fixture(scope="session")
def synthetic_gold(synthetic_silver):
    """Served features plus the columns needed for splitting and evaluation."""
    from fraud_common import features

    gold = features.derive_features_pandas(synthetic_silver)
    gold["step"] = synthetic_silver["step"].values
    gold["is_fraud"] = synthetic_silver["is_fraud"].values
    gold["transaction_id"] = synthetic_silver["transaction_id"].values
    return gold


@pytest.fixture(scope="session")
def scored_frame(synthetic_gold):
    """A frame with a plausible, imperfect score column.

    The score correlates with the label but is far from perfect — a perfect
    ranker would make several metric assertions vacuous.
    """
    rng = np.random.default_rng(7)
    n = len(synthetic_gold)
    base = rng.beta(2, 20, size=n)
    lift = synthetic_gold["is_fraud"].values * rng.beta(6, 3, size=n)
    score = np.clip(base + lift, 0.0, 1.0)
    out = synthetic_gold.copy()
    out["fraud_score"] = score
    return out
