"""Chronological split ordering and reproducible sampling."""

from __future__ import annotations

import pandas as pd
import pytest

from fraud_common import splits
from fraud_common.config import load_config


# --------------------------------------------------------------------------- #
# Boundaries
# --------------------------------------------------------------------------- #

def test_boundaries_are_strictly_increasing():
    b = splits.compute_boundaries(1, 744, 0.70, 0.15)
    assert b.min_step <= b.train_end_step < b.valid_end_step < b.max_step


def test_boundaries_match_the_seventy_fifteen_fifteen_intent():
    b = splits.compute_boundaries(1, 744, 0.70, 0.15)
    assert b.train_end_step == 520          # 1 + int(744*0.70) - 1
    assert b.valid_end_step == 632          # 1 + int(744*0.85) - 1


def test_every_split_is_non_empty_on_a_degenerate_range():
    b = splits.compute_boundaries(1, 3, 0.70, 0.15)
    assert {b.assign(s) for s in (1, 2, 3)} == {"train", "valid", "test"}


def test_boundaries_reject_an_inverted_range():
    with pytest.raises(ValueError):
        splits.compute_boundaries(100, 10, 0.70, 0.15)


def test_boundaries_reject_a_range_too_short_to_split():
    with pytest.raises(ValueError):
        splits.compute_boundaries(1, 2, 0.70, 0.15)


@pytest.mark.parametrize("step,expected", [
    (1, "train"), (520, "train"), (521, "valid"),
    (632, "valid"), (633, "test"), (744, "test"),
])
def test_assign_places_each_step_in_exactly_one_split(step, expected):
    b = splits.compute_boundaries(1, 744, 0.70, 0.15)
    assert b.assign(step) == expected


def test_sql_expression_agrees_with_python_assignment():
    b = splits.compute_boundaries(1, 744, 0.70, 0.15)
    sql = b.sql_expression("step")
    assert f"step <= {b.train_end_step} THEN 'train'" in sql
    assert f"step <= {b.valid_end_step} THEN 'valid'" in sql
    assert "ELSE 'test'" in sql


def test_boundaries_reject_overlapping_construction():
    with pytest.raises(ValueError):
        splits.SplitBoundaries(min_step=1, train_end_step=100,
                               valid_end_step=50, max_step=744)


# --------------------------------------------------------------------------- #
# Split integrity on data
# --------------------------------------------------------------------------- #

def test_splits_do_not_overlap_in_time(synthetic_gold):
    b = splits.compute_boundaries(
        int(synthetic_gold["step"].min()), int(synthetic_gold["step"].max()), 0.70, 0.15
    )
    train, valid, test = splits.split_frames(synthetic_gold, b)
    assert train["step"].max() < valid["step"].min()
    assert valid["step"].max() < test["step"].min()


def test_splits_partition_the_data_without_loss(synthetic_gold):
    b = splits.compute_boundaries(
        int(synthetic_gold["step"].min()), int(synthetic_gold["step"].max()), 0.70, 0.15
    )
    train, valid, test = splits.split_frames(synthetic_gold, b)
    assert len(train) + len(valid) + len(test) == len(synthetic_gold)
    ids = set(train["transaction_id"]) | set(valid["transaction_id"]) | set(test["transaction_id"])
    assert len(ids) == len(synthetic_gold), "a row appeared in two splits"


def test_assert_chronological_accepts_correct_ordering(synthetic_gold):
    b = splits.compute_boundaries(1, 744, 0.70, 0.15)
    train, valid, test = splits.split_frames(synthetic_gold, b)
    splits.assert_chronological(
        [("train", train), ("valid", valid), ("test", test)]
    )


def test_assert_chronological_catches_a_random_split(synthetic_gold):
    """A random split is exactly what this project forbids — prove it is caught."""
    shuffled = synthetic_gold.sample(frac=1.0, random_state=0).reset_index(drop=True)
    third = len(shuffled) // 3
    with pytest.raises(ValueError, match="chronological violation"):
        splits.assert_chronological([
            ("train", shuffled.iloc[:third]),
            ("valid", shuffled.iloc[third:2 * third]),
            ("test", shuffled.iloc[2 * third:]),
        ])


def test_assert_chronological_rejects_an_empty_split():
    empty = pd.DataFrame({"step": []})
    full = pd.DataFrame({"step": [1, 2, 3]})
    with pytest.raises(ValueError, match="empty"):
        splits.assert_chronological([("train", full), ("valid", empty)])


# --------------------------------------------------------------------------- #
# Deterministic sampling
# --------------------------------------------------------------------------- #

def test_hash_is_in_the_unit_interval():
    values = [splits.hash_unit_interval(f"tx{i}", 42) for i in range(500)]
    assert all(0.0 <= v < 1.0 for v in values)


def test_hash_is_stable_across_calls():
    assert splits.hash_unit_interval("tx0001", 42) == splits.hash_unit_interval("tx0001", 42)


def test_hash_depends_on_the_seed():
    assert splits.hash_unit_interval("tx0001", 42) != splits.hash_unit_interval("tx0001", 43)


def test_hash_is_roughly_uniform():
    values = [splits.hash_unit_interval(f"tx{i}", 42) for i in range(20_000)]
    below_ten_percent = sum(v < 0.10 for v in values) / len(values)
    assert 0.09 < below_ten_percent < 0.11


def test_keep_mask_is_repeatable(synthetic_gold):
    keys = synthetic_gold["transaction_id"].tolist()
    first = splits.deterministic_keep_mask(keys, 0.10, 42)
    second = splits.deterministic_keep_mask(keys, 0.10, 42)
    assert first == second


def test_keep_mask_is_order_independent(synthetic_gold):
    """Repartitioning must not change which rows are selected."""
    keys = synthetic_gold["transaction_id"].tolist()
    forward = dict(zip(keys, splits.deterministic_keep_mask(keys, 0.10, 42)))
    reversed_keys = list(reversed(keys))
    backward = dict(zip(reversed_keys,
                        splits.deterministic_keep_mask(reversed_keys, 0.10, 42)))
    assert forward == backward


def test_keep_mask_respects_the_requested_fraction(synthetic_gold):
    keys = synthetic_gold["transaction_id"].tolist()
    kept = sum(splits.deterministic_keep_mask(keys, 0.25, 42))
    assert 0.22 < kept / len(keys) < 0.28


def test_keep_mask_at_fraction_one_keeps_everything(synthetic_gold):
    keys = synthetic_gold["transaction_id"].tolist()
    assert all(splits.deterministic_keep_mask(keys, 1.0, 42))


def test_keep_mask_rejects_an_out_of_range_fraction():
    with pytest.raises(ValueError):
        splits.deterministic_keep_mask(["a"], 1.5, 42)


# --------------------------------------------------------------------------- #
# Training-frame sampling
# --------------------------------------------------------------------------- #

def test_sampling_retains_every_fraud_row(synthetic_gold):
    sampled = splits.sample_training_frame(
        synthetic_gold, "transaction_id", "is_fraud", 0.10, 42
    )
    assert sampled["is_fraud"].sum() == synthetic_gold["is_fraud"].sum()


def test_sampling_thins_only_the_majority_class(synthetic_gold):
    sampled = splits.sample_training_frame(
        synthetic_gold, "transaction_id", "is_fraud", 0.10, 42
    )
    original_negatives = (synthetic_gold["is_fraud"] == 0).sum()
    sampled_negatives = (sampled["is_fraud"] == 0).sum()
    assert sampled_negatives < original_negatives
    assert 0.07 < sampled_negatives / original_negatives < 0.13


def test_sampling_is_reproducible(synthetic_gold):
    a = splits.sample_training_frame(synthetic_gold, "transaction_id", "is_fraud", 0.10, 42)
    b = splits.sample_training_frame(synthetic_gold, "transaction_id", "is_fraud", 0.10, 42)
    assert a["transaction_id"].tolist() == b["transaction_id"].tolist()


def test_sampling_raises_on_a_missing_column(synthetic_gold):
    with pytest.raises(KeyError):
        splits.sample_training_frame(synthetic_gold, "nope", "is_fraud", 0.10, 42)


# --------------------------------------------------------------------------- #
# Spark SQL / Python parity
# --------------------------------------------------------------------------- #

def test_sample_sql_expression_uses_the_same_constants_as_python():
    """The SQL and Python paths must divide by the identical power of two.

    2**60 is exactly representable in IEEE-754, so both engines produce
    byte-identical ratios and therefore identical samples.
    """
    sql = splits.sample_sql_expression("transaction_id", 42, 0.10)
    assert "SHA2(CONCAT('42', ':', transaction_id), 256)" in sql
    assert f"SUBSTR" in sql and ", 1, 15)" in sql
    assert "1152921504606846976" in sql
    assert splits._HASH_DENOM_INT == 2 ** 60
    assert float(splits._HASH_DENOM_INT) == splits._HASH_MAX


def test_sample_sql_expression_bypasses_sampling_for_the_minority_class():
    sql = splits.sample_sql_expression("transaction_id", 42, 0.10, "is_fraud = 1")
    assert sql.startswith("((is_fraud = 1) OR (")


def test_config_split_fractions_leave_room_for_a_test_split():
    cfg = load_config()
    assert cfg.train_step_fraction + cfg.valid_step_fraction < 1.0
    boundaries = splits.compute_boundaries(
        1, 744, cfg.train_step_fraction, cfg.valid_step_fraction
    )
    assert boundaries.valid_end_step < boundaries.max_step
