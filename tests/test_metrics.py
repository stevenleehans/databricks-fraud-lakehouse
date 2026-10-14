"""Metric calculations, against hand-computed values where possible."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_common import metrics


# --------------------------------------------------------------------------- #
# Confusion matrix
# --------------------------------------------------------------------------- #

def test_confusion_matrix_is_computed_by_hand():
    y_true = [1, 1, 0, 0, 1, 0]
    y_score = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
    # threshold 0.5 -> predicted positive: rows 0, 1, 2
    tp, fp, tn, fn = metrics.confusion_at_threshold(y_true, y_score, 0.5)
    assert (tp, fp, tn, fn) == (2, 1, 2, 1)


def test_threshold_is_inclusive_at_the_boundary():
    tp, fp, tn, fn = metrics.confusion_at_threshold([1], [0.5], 0.5)
    assert tp == 1


def test_confusion_counts_sum_to_the_row_count():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_score = rng.random(500)
    assert sum(metrics.confusion_at_threshold(y_true, y_score, 0.4)) == 500


# --------------------------------------------------------------------------- #
# Precision / recall / F1 / FPR
# --------------------------------------------------------------------------- #

def test_precision_recall_f1_match_hand_computation():
    y_true = [1, 1, 0, 0, 1, 0]
    y_score = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
    result = metrics.evaluate(y_true, y_score, 0.5)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["false_positive_rate"] == pytest.approx(1 / 3)


def test_perfect_ranking_scores_one():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    result = metrics.evaluate(y_true, y_score, 0.5)
    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)


def test_metrics_do_not_divide_by_zero_when_nothing_is_predicted_positive():
    result = metrics.evaluate([0, 1, 0], [0.1, 0.2, 0.05], 0.9)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    assert result["predicted_positive_rate"] == 0.0


# --------------------------------------------------------------------------- #
# PR-AUC
# --------------------------------------------------------------------------- #

def test_pr_auc_of_a_random_ranker_approaches_prevalence():
    """Average precision of a random ranker converges on the base rate.

    This is why PR-AUC is the right selector at 0.13 % prevalence: the
    no-skill baseline is 0.0013, not 0.5 as it is for ROC-AUC.
    """
    rng = np.random.default_rng(3)
    n = 60_000
    y_true = (rng.random(n) < 0.02).astype(int)
    y_score = rng.random(n)
    assert metrics.pr_auc(y_true, y_score) == pytest.approx(0.02, abs=0.006)


def test_roc_auc_of_a_random_ranker_approaches_one_half():
    rng = np.random.default_rng(4)
    n = 40_000
    y_true = (rng.random(n) < 0.02).astype(int)
    y_score = rng.random(n)
    assert metrics.roc_auc(y_true, y_score) == pytest.approx(0.5, abs=0.02)


def test_pr_auc_beats_prevalence_for_an_informative_score(scored_frame):
    prevalence = scored_frame["is_fraud"].mean()
    value = metrics.pr_auc(scored_frame["is_fraud"], scored_frame["fraud_score"])
    assert value > prevalence * 5


def test_pr_auc_is_nan_without_positives():
    assert np.isnan(metrics.pr_auc([0, 0, 0], [0.1, 0.2, 0.3]))


def test_roc_auc_is_nan_for_a_single_class():
    assert np.isnan(metrics.roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_accuracy_is_not_offered_as_a_selection_metric():
    result = metrics.evaluate([0, 1], [0.1, 0.9], 0.5)
    assert "accuracy" not in result
    assert metrics.PRIMARY_METRIC == "pr_auc"


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #

def test_non_binary_target_is_rejected():
    with pytest.raises(ValueError, match="binary"):
        metrics.evaluate([0, 1, 2], [0.1, 0.2, 0.3], 0.5)


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        metrics.evaluate([0, 1], [0.1], 0.5)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        metrics.evaluate([], [], 0.5)


# --------------------------------------------------------------------------- #
# Fraud value captured
# --------------------------------------------------------------------------- #

def test_fraud_amount_capture_is_computed_by_hand():
    y_true = [1, 1, 0]
    y_score = [0.9, 0.1, 0.9]
    amounts = [100.0, 300.0, 50.0]
    result = metrics.evaluate(y_true, y_score, 0.5, amounts=amounts)
    assert result["fraud_amount_total"] == pytest.approx(400.0)
    assert result["fraud_amount_captured"] == pytest.approx(100.0)
    assert result["fraud_amount_capture_rate"] == pytest.approx(0.25)
    assert result["legit_amount_blocked"] == pytest.approx(50.0)


def test_amount_length_must_align():
    with pytest.raises(ValueError, match="amounts"):
        metrics.evaluate([0, 1], [0.1, 0.9], 0.5, amounts=[1.0])


# --------------------------------------------------------------------------- #
# Threshold selection
# --------------------------------------------------------------------------- #

def test_chosen_threshold_respects_the_false_positive_budget(scored_frame):
    y_true = scored_frame["is_fraud"].values
    y_score = scored_frame["fraud_score"].values
    threshold = metrics.choose_threshold_for_max_fpr(y_true, y_score, 0.01)
    achieved = metrics.evaluate(y_true, y_score, threshold)["false_positive_rate"]
    assert achieved <= 0.01 + 1e-9


def test_a_looser_budget_never_reduces_recall(scored_frame):
    y_true = scored_frame["is_fraud"].values
    y_score = scored_frame["fraud_score"].values
    strict = metrics.choose_threshold_for_max_fpr(y_true, y_score, 0.005)
    loose = metrics.choose_threshold_for_max_fpr(y_true, y_score, 0.05)
    assert metrics.recall_at_threshold(y_true, y_score, loose) >= \
           metrics.recall_at_threshold(y_true, y_score, strict)


def test_threshold_is_always_a_probability(scored_frame):
    for budget in (0.001, 0.01, 0.1):
        t = metrics.choose_threshold_for_max_fpr(
            scored_frame["is_fraud"], scored_frame["fraud_score"], budget
        )
        assert 0.0 <= t <= 1.0


def test_threshold_selection_falls_back_without_positives():
    assert metrics.choose_threshold_for_max_fpr(
        [0, 0, 0], [0.1, 0.2, 0.3], 0.01, default=0.42
    ) == 0.42


def test_threshold_selection_rejects_an_invalid_budget():
    with pytest.raises(ValueError):
        metrics.choose_threshold_for_max_fpr([0, 1], [0.1, 0.9], 1.5)


# --------------------------------------------------------------------------- #
# Model comparison
# --------------------------------------------------------------------------- #

def test_the_higher_pr_auc_wins():
    best, score, _ = metrics.compare_models({
        "baseline": {"pr_auc": 0.31, "roc_auc": 0.98},
        "trees": {"pr_auc": 0.44, "roc_auc": 0.95},
    })
    assert best == "trees"
    assert score == pytest.approx(0.44)


def test_roc_auc_does_not_decide_the_winner():
    """The baseline has the better ROC-AUC and must still lose."""
    best, _, _ = metrics.compare_models({
        "baseline": {"pr_auc": 0.31, "roc_auc": 0.99},
        "trees": {"pr_auc": 0.44, "roc_auc": 0.90},
    })
    assert best == "trees"


def test_a_nan_model_never_wins():
    best, _, _ = metrics.compare_models({
        "broken": {"pr_auc": float("nan")},
        "working": {"pr_auc": 0.10},
    })
    assert best == "working"


def test_comparison_fails_when_every_model_is_nan():
    with pytest.raises(ValueError, match="finite"):
        metrics.compare_models({"a": {"pr_auc": float("nan")}})


def test_comparison_rejects_an_empty_leaderboard():
    with pytest.raises(ValueError):
        metrics.compare_models({})


# --------------------------------------------------------------------------- #
# Bundle shape
# --------------------------------------------------------------------------- #

def test_evaluate_reports_every_required_metric(scored_frame):
    result = metrics.evaluate(
        scored_frame["is_fraud"], scored_frame["fraud_score"], 0.5,
        amounts=scored_frame["amount"],
    )
    for required in [
        "pr_auc", "roc_auc", "precision", "recall", "f1",
        "false_positive_rate", "true_positives", "false_positives",
        "true_negatives", "false_negatives", "prevalence",
        "predicted_positive_rate", "fraud_amount_capture_rate",
    ]:
        assert required in result, f"{required} missing from the metric bundle"


def test_prefix_is_applied_to_every_key():
    result = metrics.evaluate([0, 1], [0.1, 0.9], 0.5, prefix="test_")
    assert all(k.startswith("test_") for k in result)
