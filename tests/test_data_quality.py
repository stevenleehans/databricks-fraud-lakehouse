"""Silver quality-rule semantics and Bronze→Silver reconciliation.

The production rules run as Spark SQL. This module re-implements the *reject*
rules in pandas and checks them against deliberately corrupted rows, so the
semantics ("what counts as invalid") are verified without a cluster. The Spark
execution of the same rules is asserted inside `02_transform_silver`, and
`test_notebooks_static` proves that assertion is present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_common import schema


# --------------------------------------------------------------------------- #
# pandas mirror of the reject rules
# --------------------------------------------------------------------------- #

def evaluate_reject_rules(df: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per reject rule; True == the row satisfies the rule."""
    results = pd.DataFrame(index=df.index)
    results["step_present_and_positive"] = df["step"].notna() & (df["step"] >= 1)
    results["type_in_allowed_set"] = df["type"].isin(schema.ALLOWED_TRANSACTION_TYPES)
    results["amount_present"] = df["amount"].notna()
    results["amount_non_negative"] = df["amount"].fillna(-1) >= 0
    results["balances_non_negative"] = (
        (df["old_balance_orig"].fillna(-1) >= 0)
        & (df["new_balance_orig"].fillna(-1) >= 0)
        & (df["old_balance_dest"].fillna(-1) >= 0)
        & (df["new_balance_dest"].fillna(-1) >= 0)
    )
    results["balances_present"] = (
        df["old_balance_orig"].notna() & df["old_balance_dest"].notna()
    )
    results["target_in_allowed_values"] = df["is_fraud"].isin(schema.ALLOWED_TARGET_VALUES)
    results["flag_in_allowed_values"] = df["is_flagged_fraud"].isin([0, 1])
    results["account_names_present"] = df["name_orig"].notna() & df["name_dest"].notna()
    return results


@pytest.fixture()
def corrupted(synthetic_silver) -> pd.DataFrame:
    """Clean rows plus one row violating each reject rule."""
    clean = synthetic_silver.head(100).copy()
    template = clean.iloc[0].to_dict()

    def bad(**overrides):
        row = dict(template)
        row.update(overrides)
        return row

    broken = [
        bad(transaction_id="bad_step", step=0),
        bad(transaction_id="bad_type", type="CRYPTO_SWAP"),
        bad(transaction_id="null_amount", amount=np.nan),
        bad(transaction_id="negative_amount", amount=-50.0),
        bad(transaction_id="negative_balance", old_balance_orig=-1.0),
        bad(transaction_id="null_balance", old_balance_dest=np.nan),
        bad(transaction_id="bad_target", is_fraud=2),
        bad(transaction_id="bad_flag", is_flagged_fraud=7),
        bad(transaction_id="null_name", name_dest=None),
    ]
    return pd.concat([clean, pd.DataFrame(broken)], ignore_index=True)


# --------------------------------------------------------------------------- #
# Rule semantics
# --------------------------------------------------------------------------- #

def test_every_reject_rule_is_mirrored_here(corrupted):
    mirrored = set(evaluate_reject_rules(corrupted).columns)
    declared = {rule.name for rule in schema.reject_rules()}
    assert mirrored == declared, (
        "the pandas mirror has drifted from schema.QUALITY_RULES: "
        f"{mirrored ^ declared}"
    )


def test_clean_rows_pass_every_rule(synthetic_silver):
    results = evaluate_reject_rules(synthetic_silver)
    assert results.all(axis=None), (
        "the synthetic fixture should be entirely valid; failing rules: "
        f"{[c for c in results.columns if not results[c].all()]}"
    )


@pytest.mark.parametrize("transaction_id,rule", [
    ("bad_step", "step_present_and_positive"),
    ("bad_type", "type_in_allowed_set"),
    ("null_amount", "amount_present"),
    ("negative_amount", "amount_non_negative"),
    ("negative_balance", "balances_non_negative"),
    ("null_balance", "balances_present"),
    ("bad_target", "target_in_allowed_values"),
    ("bad_flag", "flag_in_allowed_values"),
    ("null_name", "account_names_present"),
])
def test_each_corruption_trips_its_rule(corrupted, transaction_id, rule):
    results = evaluate_reject_rules(corrupted)
    index = corrupted.index[corrupted["transaction_id"] == transaction_id][0]
    assert not results.loc[index, rule], f"{transaction_id} should fail {rule}"


def test_an_out_of_range_target_is_always_rejected(corrupted):
    results = evaluate_reject_rules(corrupted)
    valid = corrupted[results.all(axis=1)]
    assert set(valid["is_fraud"].unique()) <= set(schema.ALLOWED_TARGET_VALUES)


def test_only_allowed_transaction_types_survive(corrupted):
    results = evaluate_reject_rules(corrupted)
    valid = corrupted[results.all(axis=1)]
    assert set(valid["type"].unique()) <= set(schema.ALLOWED_TRANSACTION_TYPES)


# --------------------------------------------------------------------------- #
# Reconciliation: nothing may be lost between Bronze and Silver
# --------------------------------------------------------------------------- #

def test_valid_plus_rejected_equals_the_input(corrupted):
    results = evaluate_reject_rules(corrupted)
    is_valid = results.all(axis=1)
    assert int(is_valid.sum()) + int((~is_valid).sum()) == len(corrupted)


def test_every_corrupted_row_is_quarantined_not_dropped(corrupted):
    results = evaluate_reject_rules(corrupted)
    rejected = corrupted[~results.all(axis=1)]
    assert len(rejected) == 9
    # Each rejected row keeps its identity so it can be investigated later.
    assert rejected["transaction_id"].notna().all()


def test_reject_reasons_can_be_reconstructed_for_each_rejected_row(corrupted):
    results = evaluate_reject_rules(corrupted)
    failing = ~results
    for index in corrupted.index[~results.all(axis=1)]:
        reasons = [c for c in results.columns if failing.loc[index, c]]
        assert reasons, "a rejected row must record at least one reason"


def test_a_row_can_fail_several_rules_at_once():
    row = pd.DataFrame([{
        "step": 0, "type": "NOPE", "amount": -1.0, "name_orig": "C1",
        "old_balance_orig": -5.0, "new_balance_orig": 0.0,
        "name_dest": "C2", "old_balance_dest": 0.0, "new_balance_dest": 0.0,
        "is_fraud": 5, "is_flagged_fraud": 0,
    }])
    failures = (~evaluate_reject_rules(row)).sum(axis=1).iloc[0]
    assert failures >= 5


# --------------------------------------------------------------------------- #
# Deterministic exact-duplicate handling
# --------------------------------------------------------------------------- #

def natural_key_columns():
    from fraud_common.features import SOURCE_TO_SILVER
    return [SOURCE_TO_SILVER[c] for c in schema.SOURCE_COLUMNS]


def test_exact_duplicates_are_identified(synthetic_silver):
    keys = natural_key_columns()
    base = synthetic_silver.head(50)
    doubled = pd.concat([base, base], ignore_index=True)
    occurrence = doubled.groupby(keys, dropna=False).cumcount()
    assert int((occurrence > 0).sum()) == 50


def test_deduplication_keeps_exactly_one_copy(synthetic_silver):
    keys = natural_key_columns()
    base = synthetic_silver.head(50)
    doubled = pd.concat([base, base, base], ignore_index=True)
    occurrence = doubled.groupby(keys, dropna=False).cumcount()
    assert len(doubled[occurrence == 0]) == 50


def test_occurrence_index_makes_surrogate_keys_unique(synthetic_silver):
    """transaction_id folds in the occurrence index, so identical source rows
    still receive distinct ids rather than colliding."""
    import hashlib

    keys = natural_key_columns()
    base = synthetic_silver.head(20)
    doubled = pd.concat([base, base], ignore_index=True)
    occurrence = doubled.groupby(keys, dropna=False).cumcount()

    ids = [
        hashlib.sha256(
            "|".join([str(doubled.loc[i, k]) for k in keys] + [str(occurrence.loc[i])])
            .encode()
        ).hexdigest()
        for i in doubled.index
    ]
    assert len(set(ids)) == len(doubled)


def test_duplicate_detection_is_order_independent(synthetic_silver):
    keys = natural_key_columns()
    base = synthetic_silver.head(30)
    doubled = pd.concat([base, base], ignore_index=True)
    forward = int((doubled.groupby(keys, dropna=False).cumcount() > 0).sum())
    shuffled = doubled.sample(frac=1.0, random_state=1).reset_index(drop=True)
    backward = int((shuffled.groupby(keys, dropna=False).cumcount() > 0).sum())
    assert forward == backward


# --------------------------------------------------------------------------- #
# Warn rules
# --------------------------------------------------------------------------- #

def test_warn_rules_do_not_quarantine_rows():
    warn_names = {rule.name for rule in schema.warn_rules()}
    reject_names = {rule.name for rule in schema.reject_rules()}
    assert warn_names.isdisjoint(reject_names)
    assert "amount_zero_is_suspicious" in warn_names


def test_reject_and_warn_rules_partition_all_rules():
    assert len(schema.reject_rules()) + len(schema.warn_rules()) == len(schema.QUALITY_RULES)


def test_zero_amount_rows_are_kept_but_counted():
    """PaySim contains genuine zero-amount rows. Dropping them would be a
    silent, undocumented data decision — so it is a warning, not a reject."""
    rule = next(r for r in schema.QUALITY_RULES if r.name == "amount_zero_is_suspicious")
    assert rule.severity == "warn"
