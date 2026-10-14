"""PSI, categorical shift, missingness, score distribution and windowing."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_common import drift


# --------------------------------------------------------------------------- #
# PSI — numeric
# --------------------------------------------------------------------------- #

def test_identical_distributions_give_psi_of_zero():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=20_000)
    binning = drift.build_numeric_binning("x", reference, bins=10)
    assert drift.psi_numeric(binning, reference) == pytest.approx(0.0, abs=1e-9)


def test_a_resample_of_the_same_distribution_stays_green():
    rng = np.random.default_rng(2)
    binning = drift.build_numeric_binning("x", rng.normal(size=50_000), bins=10)
    fresh = rng.normal(size=50_000)
    value = drift.psi_numeric(binning, fresh)
    assert value < drift.DEFAULT_AMBER
    assert drift.psi_status(value) == drift.GREEN


def test_a_large_shift_is_flagged_red():
    rng = np.random.default_rng(3)
    binning = drift.build_numeric_binning("x", rng.normal(size=20_000), bins=10)
    shifted = rng.normal(loc=2.0, size=20_000)
    value = drift.psi_numeric(binning, shifted)
    assert value > drift.DEFAULT_RED
    assert drift.psi_status(value) == drift.RED


def test_psi_grows_monotonically_with_the_shift():
    rng = np.random.default_rng(4)
    binning = drift.build_numeric_binning("x", rng.normal(size=40_000), bins=10)
    values = [
        drift.psi_numeric(binning, rng.normal(loc=shift, size=40_000))
        for shift in (0.0, 0.25, 0.5, 1.0, 2.0)
    ]
    assert values == sorted(values)


def test_psi_is_symmetric():
    """PSI's (cur - ref) * ln(cur / ref) form is symmetric by construction."""
    a = [0.2, 0.3, 0.5]
    b = [0.4, 0.3, 0.3]
    assert drift.psi_from_proportions(a, b) == pytest.approx(
        drift.psi_from_proportions(b, a)
    )


def test_psi_is_non_negative():
    rng = np.random.default_rng(5)
    for _ in range(20):
        a = rng.dirichlet(np.ones(8))
        b = rng.dirichlet(np.ones(8))
        assert drift.psi_from_proportions(a, b) >= -1e-12


def test_psi_hand_computed_example():
    reference = [0.5, 0.5]
    current = [0.6, 0.4]
    expected = (0.6 - 0.5) * np.log(0.6 / 0.5) + (0.4 - 0.5) * np.log(0.4 / 0.5)
    assert drift.psi_from_proportions(reference, current) == pytest.approx(expected)


def test_an_emptied_bin_does_not_produce_infinity():
    """A bin going to zero is the signal, so it must be floored, not dropped."""
    binning = drift.build_numeric_binning("x", np.arange(1000.0), bins=10)
    value = drift.psi_numeric(binning, np.full(500, 5.0), epsilon=1e-6)
    assert np.isfinite(value)
    assert value > drift.DEFAULT_RED


def test_mismatched_vector_lengths_are_rejected():
    with pytest.raises(ValueError):
        drift.psi_from_proportions([0.5, 0.5], [0.3, 0.3, 0.4])


def test_quantile_edges_handle_a_heavy_tail():
    """`amount` spans 0 to ~9.2e7; equal-width bins would be useless."""
    rng = np.random.default_rng(6)
    heavy = rng.lognormal(mean=8, sigma=2, size=50_000)
    binning = drift.build_numeric_binning("amount", heavy, bins=10)
    counts, _ = np.histogram(heavy, bins=np.asarray(binning.edges))
    share = counts / counts.sum()
    assert share.max() < 0.20, "quantile binning should spread mass evenly"


# --------------------------------------------------------------------------- #
# Bucketing SQL — regression tests for a real production failure
# --------------------------------------------------------------------------- #

BINARY_FEATURES = [0, 1] * 500


def test_a_binary_feature_gets_a_usable_boundary():
    """`is_night` and friends must not collapse to a single bin.

    Quantiles of a 0/1 column give two unique edges; replacing both with
    +/-inf used to leave no interior boundary at all.
    """
    binning = drift.build_numeric_binning("is_night", BINARY_FEATURES, bins=10)
    interior = [e for e in binning.edges[1:-1] if np.isfinite(e)]
    assert interior, "a binary feature must still have a split point"
    assert len(binning.reference_proportions) >= 2


def test_binary_feature_psi_detects_a_real_shift():
    """The bug hid drift: a single bin makes PSI identically zero."""
    binning = drift.build_numeric_binning("is_night", BINARY_FEATURES, bins=10)
    shifted = [1] * 1000                      # every transaction now at night
    assert drift.psi_numeric(binning, shifted) > drift.DEFAULT_RED


def test_bucket_sql_is_valid_for_a_binary_feature():
    """Regression: this emitted `CASE  ELSE 0 END` -> PARSE_SYNTAX_ERROR."""
    binning = drift.build_numeric_binning("is_night", BINARY_FEATURES, bins=10)
    sql = drift.bucket_case_sql("is_night", binning.edges)
    assert "CASE  ELSE" not in sql
    assert sql.startswith("CASE WHEN")
    assert sql.endswith("END")


def test_bucket_sql_never_emits_an_empty_case():
    """No binning, however degenerate, may produce a CASE with zero clauses."""
    for values in ([0, 1] * 50, [5.0] * 50, list(range(3)) * 50,
                   list(np.random.default_rng(0).normal(size=500))):
        binning = drift.build_numeric_binning("f", values, bins=10)
        sql = drift.bucket_case_sql("f", binning.edges)
        assert "CASE  " not in sql
        if sql != "0":
            assert sql.count("WHEN") >= 1


def test_bucket_sql_falls_back_to_a_single_bin_when_unavoidable():
    assert drift.bucket_case_sql("f", [-np.inf, np.inf]) == "0"


def test_bucket_sql_excludes_infinite_edges_from_clauses():
    sql = drift.bucket_case_sql("f", [-np.inf, 0.5, 1.5, np.inf])
    assert "inf" not in sql.lower()
    assert sql.count("WHEN") == 2


def test_bucket_sql_indices_cover_every_reference_bin():
    binning = drift.build_numeric_binning("f", list(range(100)), bins=10)
    sql = drift.bucket_case_sql("f", binning.edges)
    highest = int(sql.rsplit("ELSE", 1)[1].replace("END", "").strip())
    assert highest == len(binning.reference_proportions) - 1


def test_a_constant_feature_does_not_crash():
    binning = drift.build_numeric_binning("c", np.full(100, 7.0), bins=10)
    assert np.isfinite(drift.psi_numeric(binning, np.full(100, 7.0)))


def test_empty_reference_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        drift.build_numeric_binning("x", [])


def test_edges_are_unbounded_at_both_ends():
    binning = drift.build_numeric_binning("x", np.arange(100.0), bins=5)
    assert binning.edges[0] == -np.inf
    assert binning.edges[-1] == np.inf


def test_unseen_extreme_values_land_in_the_end_bins():
    binning = drift.build_numeric_binning("x", np.arange(100.0), bins=5)
    assert np.isfinite(drift.psi_numeric(binning, [-1e9, 1e9]))


# --------------------------------------------------------------------------- #
# PSI — categorical
# --------------------------------------------------------------------------- #

CATEGORIES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def test_identical_category_mixes_give_psi_of_zero():
    values = ["PAYMENT"] * 60 + ["CASH_OUT"] * 30 + ["TRANSFER"] * 10
    props = drift.category_proportions(values, CATEGORIES)
    assert drift.psi_categorical(props, props) == pytest.approx(0.0, abs=1e-9)


def test_a_category_mix_shift_is_detected():
    reference = drift.category_proportions(
        ["PAYMENT"] * 80 + ["CASH_OUT"] * 20, CATEGORIES
    )
    current = drift.category_proportions(
        ["PAYMENT"] * 20 + ["CASH_OUT"] * 80, CATEGORIES
    )
    assert drift.psi_categorical(reference, current) > drift.DEFAULT_RED


def test_a_vanished_category_still_contributes():
    """Fixing the category list means a disappearing type is visible."""
    reference = drift.category_proportions(["PAYMENT"] * 50 + ["DEBIT"] * 50, CATEGORIES)
    current = drift.category_proportions(["PAYMENT"] * 100, CATEGORIES)
    value = drift.psi_categorical(reference, current)
    assert np.isfinite(value) and value > drift.DEFAULT_RED


def test_unknown_categories_are_bucketed_not_dropped():
    props = drift.category_proportions(["PAYMENT", "CRYPTO_SWAP"], CATEGORIES)
    assert props["__unseen__"] == pytest.approx(0.5)


def test_proportions_cover_every_declared_category():
    props = drift.category_proportions(["PAYMENT"], CATEGORIES)
    assert set(CATEGORIES) <= set(props)


# --------------------------------------------------------------------------- #
# Status thresholds
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value,expected", [
    (0.0, drift.GREEN), (0.09, drift.GREEN),
    (0.10, drift.AMBER), (0.24, drift.AMBER),
    (0.25, drift.RED), (5.0, drift.RED),
])
def test_psi_status_boundaries(value, expected):
    assert drift.psi_status(value) == expected


def test_no_data_is_amber_not_green():
    """A window with no rows is unknown, and unknown must never read as fine."""
    assert drift.psi_status(float("nan")) == drift.AMBER


def test_overall_status_takes_the_worst():
    assert drift.overall_status([drift.GREEN, drift.AMBER, drift.RED]) == drift.RED
    assert drift.overall_status([drift.GREEN, drift.AMBER]) == drift.AMBER
    assert drift.overall_status([drift.GREEN, drift.GREEN]) == drift.GREEN
    assert drift.overall_status([]) == drift.GREEN


# --------------------------------------------------------------------------- #
# Missing rate
# --------------------------------------------------------------------------- #

def test_missing_rate_counts_none_and_nan():
    assert drift.missing_rate([1.0, None, float("nan"), 4.0]) == pytest.approx(0.5)


def test_missing_rate_of_a_complete_column_is_zero():
    assert drift.missing_rate([1.0, 2.0, 3.0]) == 0.0


@pytest.mark.parametrize("delta,expected", [
    (0.0, drift.GREEN), (0.005, drift.GREEN),
    (0.02, drift.AMBER), (-0.02, drift.AMBER),
    (0.10, drift.RED), (-0.10, drift.RED),
])
def test_missing_rate_status_uses_absolute_change(delta, expected):
    assert drift.missing_rate_status(delta, 0.01, 0.05) == expected


# --------------------------------------------------------------------------- #
# Prediction distribution
# --------------------------------------------------------------------------- #

def test_score_distribution_reports_the_expected_keys():
    stats = drift.score_distribution([0.1, 0.2, 0.9, 0.95], 0.5)
    for key in ["n_scored", "avg_score", "median_score", "p90_score",
                "p99_score", "predicted_positive_rate", "predicted_positive_count"]:
        assert key in stats


def test_predicted_positive_rate_is_computed_by_hand():
    stats = drift.score_distribution([0.1, 0.2, 0.9, 0.95], 0.5)
    assert stats["predicted_positive_rate"] == pytest.approx(0.5)
    assert stats["predicted_positive_count"] == 2.0


def test_empty_scores_yield_nan_not_zero():
    stats = drift.score_distribution([], 0.5)
    assert stats["n_scored"] == 0.0
    assert np.isnan(stats["avg_score"])


def test_score_histogram_partitions_all_rows():
    scores = np.random.default_rng(9).random(1_000)
    rows = drift.score_histogram(scores)
    assert len(rows) == 20
    assert sum(r["row_count"] for r in rows) == 1_000
    assert sum(r["row_share"] for r in rows) == pytest.approx(1.0)


def test_histogram_buckets_are_contiguous():
    rows = drift.score_histogram([0.5])
    for earlier, later in zip(rows, rows[1:]):
        assert earlier["bucket_upper"] == pytest.approx(later["bucket_lower"])


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #

def test_windows_cover_the_period_without_gaps_or_overlaps():
    windows = drift.make_windows(521, 744, 6)
    assert windows[0]["start_step"] == 521
    assert windows[-1]["end_step"] == 744
    for earlier, later in zip(windows, windows[1:]):
        assert later["start_step"] == earlier["end_step"] + 1


def test_window_ids_are_unique_and_ordered():
    windows = drift.make_windows(1, 100, 5)
    ids = [w["window_id"] for w in windows]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_window_count_is_capped_by_the_available_steps():
    assert len(drift.make_windows(1, 3, 10)) == 3


def test_a_single_window_is_allowed():
    windows = drift.make_windows(10, 20, 1)
    assert len(windows) == 1
    assert (windows[0]["start_step"], windows[0]["end_step"]) == (10, 20)


def test_invalid_window_requests_are_rejected():
    with pytest.raises(ValueError):
        drift.make_windows(1, 100, 0)
    with pytest.raises(ValueError):
        drift.make_windows(100, 1, 5)


# --------------------------------------------------------------------------- #
# The distinction this project insists on
# --------------------------------------------------------------------------- #

def test_the_three_drift_kinds_are_distinct_labels():
    kinds = {drift.FEATURE_DRIFT, drift.PREDICTION_DRIFT, drift.PERFORMANCE}
    assert len(kinds) == 3


def test_feature_drift_can_occur_without_any_performance_change():
    """The core claim the monitoring design rests on, demonstrated concretely.

    Shift a feature hard enough to trip RED, while the score ranking — and
    therefore PR-AUC — is untouched. Feature drift alone cannot prove decay.
    """
    from fraud_common import metrics

    rng = np.random.default_rng(11)
    n = 20_000
    reference_feature = rng.normal(size=n)
    binning = drift.build_numeric_binning("f", reference_feature, bins=10)

    shifted_feature = rng.normal(loc=3.0, size=n)
    assert drift.psi_status(drift.psi_numeric(binning, shifted_feature)) == drift.RED

    labels = (rng.random(n) < 0.02).astype(int)
    scores = np.clip(rng.beta(2, 20, n) + labels * rng.beta(6, 3, n), 0, 1)
    before = metrics.pr_auc(labels, scores)
    after = metrics.pr_auc(labels, scores)      # same ranking, same performance
    assert before == pytest.approx(after)
