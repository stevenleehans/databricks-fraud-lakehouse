"""The fraud loss prevention simulator's arithmetic.

This is the only module in the project whose output is a *money* figure, which
makes it the easiest one to be quietly wrong about: an off-by-one in the
confusion counts still produces a plausible-looking dollar number, and nobody
reading a dashboard tile can tell. So the财 maths is pinned to hand-computed
values rather than to itself.

The other thing these tests defend is honesty: the assumptions must travel with
every figure, and the disclaimer must be on the summary row.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraud_common import impact


# --------------------------------------------------------------------------- #
# Assumptions
# --------------------------------------------------------------------------- #

def test_default_assumptions_match_the_documented_poc_values():
    a = impact.ImpactAssumptions()
    assert a.prevention_rate == 0.80
    assert a.cost_per_review == 5.00
    assert a.customer_friction_cost == 10.00
    assert a.analyst_daily_capacity == 200
    assert a.currency == "USD"


@pytest.mark.parametrize("kwargs", [
    {"prevention_rate": 1.5},
    {"prevention_rate": -0.1},
    {"cost_per_review": -1},
    {"customer_friction_cost": -1},
    {"analyst_daily_capacity": 0},
    {"currency": ""},
])
def test_impossible_assumptions_are_rejected(kwargs):
    with pytest.raises(ValueError):
        impact.ImpactAssumptions(**kwargs).validate()


def test_assumptions_are_carried_on_every_output_row():
    """A savings figure without its assumptions is unciteable."""
    columns = impact.ImpactAssumptions().as_output_columns()
    for required in ("assumption_prevention_rate", "assumption_cost_per_review",
                     "assumption_customer_friction_cost",
                     "assumption_analyst_daily_capacity"):
        assert required in columns


def test_config_assumptions_agree_with_the_library_defaults():
    """`config.py` and `impact.py` state the same defaults in two places; a
    silent divergence would make the notebook and the tests disagree."""
    from fraud_common.config import load_config

    cfg = load_config()
    a = impact.ImpactAssumptions()
    assert cfg.impact_prevention_rate == a.prevention_rate
    assert cfg.impact_cost_per_review == a.cost_per_review
    assert cfg.impact_customer_friction_cost == a.customer_friction_cost
    assert cfg.impact_analyst_daily_capacity == a.analyst_daily_capacity
    assert cfg.impact_currency == a.currency


# --------------------------------------------------------------------------- #
# Column resolution
# --------------------------------------------------------------------------- #

def test_the_projects_own_column_names_resolve_first():
    columns = ["transaction_id", "amount", "fraud_score", "label_true", "score"]
    assert impact.resolve_column(columns, impact.SCORE_COLUMN_CANDIDATES, "score") \
        == "fraud_score"
    assert impact.resolve_column(columns, impact.LABEL_COLUMN_CANDIDATES, "label") \
        == "label_true"


@pytest.mark.parametrize("alias", [
    "probability", "prediction_probability", "fraud_probability",
    "score", "predicted_probability",
])
def test_a_foreign_scoring_table_still_resolves(alias):
    assert impact.resolve_column(
        ["amount", "is_fraud", alias], impact.SCORE_COLUMN_CANDIDATES, "score"
    ) == alias


def test_resolution_is_case_insensitive():
    assert impact.resolve_column(
        ["Amount", "isFraud", "Probability"], impact.SCORE_COLUMN_CANDIDATES, "score"
    ) == "Probability"


def test_an_override_wins_over_auto_detection():
    columns = ["fraud_score", "legacy_score"]
    assert impact.resolve_column(
        columns, impact.SCORE_COLUMN_CANDIDATES, "score", override="legacy_score"
    ) == "legacy_score"


def test_an_unresolvable_column_names_what_it_tried():
    with pytest.raises(KeyError) as exc:
        impact.resolve_column(["a", "b"], impact.SCORE_COLUMN_CANDIDATES, "score")
    assert "COLUMN MAPPING" in str(exc.value)


def test_a_bad_override_fails_loudly_rather_than_falling_back():
    with pytest.raises(KeyError):
        impact.resolve_column(
            ["fraud_score"], impact.SCORE_COLUMN_CANDIDATES, "score",
            override="typo_score",
        )


# --------------------------------------------------------------------------- #
# Threshold grid
# --------------------------------------------------------------------------- #

def test_the_default_grid_runs_from_005_to_095():
    grid = impact.threshold_grid()
    assert grid[0] == 0.05 and grid[-1] == 0.95
    assert len(grid) == 19
    # float dust would produce 0.30000000000000004 as a table key
    assert all(round(t, 6) == t for t in grid)


def test_the_grid_excludes_the_degenerate_endpoints():
    """0 flags everything, 1 flags nothing; neither is an operating point."""
    grid = impact.threshold_grid()
    assert 0.0 not in grid and 1.0 not in grid


@pytest.mark.parametrize("kwargs", [
    {"step": 0}, {"start": -0.1}, {"stop": 1.5}, {"start": 0.9, "stop": 0.1},
])
def test_a_nonsensical_grid_is_rejected(kwargs):
    with pytest.raises(ValueError):
        impact.threshold_grid(**kwargs)


# --------------------------------------------------------------------------- #
# Confusion counts
# --------------------------------------------------------------------------- #

# Hand-built: 5 transactions, 2 fraudulent. At threshold 0.5 the model catches
# the 1000-value fraud, misses the 400-value one, and wrongly flags one
# legitimate 50-value transaction.
Y_TRUE = [1, 1, 0, 0, 0]
Y_SCORE = [0.90, 0.10, 0.60, 0.20, 0.05]
AMOUNTS = [1000.0, 400.0, 50.0, 70.0, 30.0]


def test_counts_are_the_hand_computed_ones():
    counts = impact.counts_from_arrays(Y_TRUE, Y_SCORE, AMOUNTS, 0.5)
    assert counts["true_positive_count"] == 1
    assert counts["false_positive_count"] == 1
    assert counts["false_negative_count"] == 1
    assert counts["true_negative_count"] == 2
    assert counts["flagged_transaction_count"] == 2
    assert counts["flagged_transaction_amount"] == 1050.0
    assert counts["detected_fraud_amount"] == 1000.0
    assert counts["missed_fraud_amount"] == 400.0


def test_the_flag_rule_is_inclusive_matching_the_rest_of_the_project():
    """`>=`, exactly as in metrics.confusion_at_threshold and notebook 06."""
    counts = impact.counts_from_arrays([1], [0.5], [10.0], 0.5)
    assert counts["true_positive_count"] == 1


def test_detected_and_missed_fraud_always_reconcile_to_the_total():
    rng = np.random.default_rng(11)
    n = 2_000
    y = (rng.random(n) < 0.03).astype(int)
    s = np.clip(rng.beta(2, 20, n) + y * rng.beta(5, 3, n), 0, 1)
    amounts = rng.lognormal(7, 1.5, n)
    population = impact.population_from_arrays(y, amounts)
    for t in impact.threshold_grid():
        counts = impact.counts_from_arrays(y, s, amounts, t)
        total = counts["detected_fraud_amount"] + counts["missed_fraud_amount"]
        assert total == pytest.approx(population["actual_fraud_amount"])
        assert (counts["true_positive_count"] + counts["false_positive_count"]
                + counts["false_negative_count"] + counts["true_negative_count"]) == n


def test_a_non_binary_label_is_rejected():
    with pytest.raises(ValueError):
        impact.counts_from_arrays([0, 2], [0.1, 0.9], [1.0, 1.0], 0.5)


def test_misaligned_inputs_are_rejected():
    with pytest.raises(ValueError):
        impact.counts_from_arrays([0, 1], [0.1], [1.0, 1.0], 0.5)


# --------------------------------------------------------------------------- #
# The financial model
# --------------------------------------------------------------------------- #

def test_the_money_maths_is_the_documented_formula():
    """Pinned by hand, from the five-row fixture above at threshold 0.5.

        gross  = 1000 detected * 0.80          =  800
        review =    2 flagged  * 5.00          =   10
        friction =  1 false positive * 10.00   =   10
        net    = 800 - 10 - 10                 =  780
    """
    assumptions = impact.ImpactAssumptions()
    row = impact.derive_impact_row(
        0.5,
        impact.counts_from_arrays(Y_TRUE, Y_SCORE, AMOUNTS, 0.5),
        impact.population_from_arrays(Y_TRUE, AMOUNTS),
        assumptions,
    )
    assert row["gross_estimated_savings"] == pytest.approx(800.0)
    assert row["review_cost"] == pytest.approx(10.0)
    assert row["false_positive_friction_cost"] == pytest.approx(10.0)
    assert row["estimated_net_savings"] == pytest.approx(780.0)
    assert row["average_saving_per_review"] == pytest.approx(390.0)
    assert row["estimated_review_days_required"] == pytest.approx(2 / 200)
    assert row["precision"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(0.5)
    assert row["false_positive_rate"] == pytest.approx(1 / 3)
    # money-weighted, not count-weighted: 1000 of 1400 fraudulent value
    assert row["fraud_detection_rate"] == pytest.approx(1000 / 1400)


def test_review_cost_is_charged_on_false_positives_too():
    """The queue cannot tell a true positive from a false one until it is
    worked, so charging reviews only on true positives would flatter the
    business case at exactly the thresholds that flag the most noise."""
    counts = impact.counts_from_arrays(Y_TRUE, Y_SCORE, AMOUNTS, 0.5)
    assumptions = impact.ImpactAssumptions(customer_friction_cost=0.0)
    row = impact.derive_impact_row(
        0.5, counts, impact.population_from_arrays(Y_TRUE, AMOUNTS), assumptions
    )
    assert row["review_cost"] == pytest.approx(
        counts["flagged_transaction_count"] * assumptions.cost_per_review
    )
    assert counts["flagged_transaction_count"] > counts["true_positive_count"]


def test_zero_prevention_rate_makes_every_operating_point_lossmaking():
    """Sanity floor: if flagging a fraud recovers nothing, reviewing is pure
    cost. A model with a positive saving under prevention_rate=0 would be a
    sign the formula had drifted."""
    rows = impact.simulate_thresholds(
        Y_TRUE, Y_SCORE, AMOUNTS, impact.ImpactAssumptions(prevention_rate=0.0)
    )
    assert all(r["estimated_net_savings"] <= 0 for r in rows)


def test_free_reviews_and_no_friction_reduce_net_to_gross():
    rows = impact.simulate_thresholds(
        Y_TRUE, Y_SCORE, AMOUNTS,
        impact.ImpactAssumptions(cost_per_review=0.0, customer_friction_cost=0.0),
    )
    for row in rows:
        assert row["estimated_net_savings"] == pytest.approx(
            row["gross_estimated_savings"]
        )


def test_a_threshold_nothing_reaches_produces_no_savings_and_no_cost():
    """Guards the zero-division paths: an empty alert queue must be 0, not NaN
    -- a NaN silently blanks a dashboard tile."""
    row = impact.derive_impact_row(
        0.99,
        impact.counts_from_arrays(Y_TRUE, Y_SCORE, AMOUNTS, 0.99),
        impact.population_from_arrays(Y_TRUE, AMOUNTS),
        impact.ImpactAssumptions(),
    )
    assert row["flagged_transaction_count"] == 0
    assert row["estimated_net_savings"] == 0.0
    assert row["average_saving_per_review"] == 0.0
    assert row["precision"] == 0.0
    assert row["recall"] == 0.0
    assert row["estimated_review_days_required"] == 0.0


def test_missing_count_fields_fail_rather_than_default_to_zero():
    with pytest.raises(KeyError):
        impact.derive_impact_row(
            0.5, {"flagged_transaction_count": 1},
            impact.population_from_arrays(Y_TRUE, AMOUNTS),
            impact.ImpactAssumptions(),
        )


def test_every_metric_the_dashboard_reads_is_produced():
    row = impact.simulate_thresholds(Y_TRUE, Y_SCORE, AMOUNTS)[0]
    for required in [
        "total_transactions", "total_transaction_amount", "actual_fraud_count",
        "actual_fraud_amount", "flagged_transaction_count",
        "flagged_transaction_amount", "true_positive_count",
        "false_positive_count", "false_negative_count", "true_negative_count",
        "detected_fraud_amount", "missed_fraud_amount",
        "gross_estimated_savings", "review_cost",
        "false_positive_friction_cost", "estimated_net_savings",
        "fraud_detection_rate", "false_positive_rate", "precision", "recall",
        "average_saving_per_review", "estimated_review_days_required",
    ]:
        assert required in row, f"simulation table would be missing {required}"


# --------------------------------------------------------------------------- #
# Monotonicity — properties that must hold for any model, on any data
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def swept():
    rng = np.random.default_rng(2024)
    n = 20_000
    y = (rng.random(n) < 0.012).astype(int)
    score = np.clip(rng.beta(2, 25, n) + y * rng.beta(6, 3, n), 0, 1)
    amounts = np.round(rng.lognormal(8.0, 1.6, n), 2)
    return impact.simulate_thresholds(y, score, amounts)


def test_raising_the_threshold_never_flags_more(swept):
    counts = [r["flagged_transaction_count"] for r in swept]
    assert counts == sorted(counts, reverse=True)


def test_raising_the_threshold_never_catches_more_fraud_value(swept):
    detected = [r["detected_fraud_amount"] for r in swept]
    assert detected == sorted(detected, reverse=True)


def test_gross_savings_and_review_cost_both_fall_with_the_threshold(swept):
    """They move the same way; the interesting behaviour is that they fall at
    different *rates*, which is what creates an optimum in between."""
    for series in ("gross_estimated_savings", "review_cost",
                   "false_positive_friction_cost"):
        values = [r[series] for r in swept]
        assert values == sorted(values, reverse=True), series


def test_review_workload_tracks_the_alert_volume(swept):
    for row in swept:
        assert row["estimated_review_days_required"] == pytest.approx(
            row["flagged_transaction_count"] / row["assumption_analyst_daily_capacity"]
        )


# --------------------------------------------------------------------------- #
# Recommendation and summary
# --------------------------------------------------------------------------- #

def test_the_recommendation_is_the_maximum_net_savings_row(swept):
    best = impact.recommend_threshold(swept)
    assert best["estimated_net_savings"] == max(
        r["estimated_net_savings"] for r in swept
    )


def test_a_tie_breaks_toward_the_smaller_review_queue():
    """Two thresholds worth the same money are not equally attractive: the one
    that flags fewer transactions needs fewer analysts."""
    rows = [
        {"threshold": 0.3, "estimated_net_savings": 100.0},
        {"threshold": 0.7, "estimated_net_savings": 100.0},
    ]
    assert impact.recommend_threshold(rows)["threshold"] == 0.7


def test_recommending_from_an_empty_sweep_fails(swept):
    with pytest.raises(ValueError):
        impact.recommend_threshold([])


def test_the_summary_carries_every_field_the_kpi_tiles_read(swept):
    summary = impact.build_summary(swept, impact.ImpactAssumptions())
    for field in impact.SUMMARY_FIELDS:
        assert field in summary


def test_the_summary_never_presents_the_estimate_as_a_real_saving(swept):
    summary = impact.build_summary(swept, impact.ImpactAssumptions())
    disclaimer = summary["result_disclaimer"].lower()
    assert "simulated" in disclaimer
    assert "not a confirmed" in disclaimer or "not a guaranteed" in disclaimer
    assert "production validation" in disclaimer


def test_the_summary_says_the_business_may_choose_another_threshold(swept):
    note = impact.build_summary(swept, impact.ImpactAssumptions())["business_override_note"]
    assert "capacity" in note.lower()
    assert "different threshold" in note.lower()


def test_the_recommendation_sentence_states_its_basis_and_its_caveat(swept):
    text = impact.recommendation_text(
        impact.recommend_threshold(swept), impact.ImpactAssumptions()
    )
    assert "highest estimated net savings" in text
    assert "validated with business stakeholders" in text


def test_summary_context_columns_are_preserved(swept):
    summary = impact.build_summary(
        swept, impact.ImpactAssumptions(), {"source_table": "cat.sch.scored"}
    )
    assert summary["source_table"] == "cat.sch.scored"


# --------------------------------------------------------------------------- #
# Operational replay
#
# The transpose of the sweep: one threshold, many periods. The risk here is a
# cumulative series that looks like a plausible curve while being wrong -- a
# missed sort, a double-counted period, a capacity number that silently uses
# the daily figure for an hourly grain.
# --------------------------------------------------------------------------- #

def _ops_row(threshold=0.5, headcount=3, fraction=1 / 24):
    return impact.derive_operations_row(
        "2024-01-01 05:00", threshold,
        impact.counts_from_arrays(Y_TRUE, Y_SCORE, AMOUNTS, threshold),
        impact.population_from_arrays(Y_TRUE, AMOUNTS),
        impact.ImpactAssumptions(), headcount, fraction,
    )


def test_operational_cost_is_review_plus_friction_and_nothing_else():
    """One definition of cost across both dashboards. A third definition
    invented here would make the two pages quietly incomparable."""
    row = _ops_row()
    assert row["operational_cost"] == pytest.approx(
        row["review_cost"] + row["false_positive_friction_cost"]
    )
    assert row["operational_cost"] == pytest.approx(20.0)   # 2*5 + 1*10


def test_net_savings_is_gross_minus_operational_cost():
    row = _ops_row()
    assert row["estimated_net_savings"] == pytest.approx(
        row["gross_estimated_savings"] - row["operational_cost"]
    )


def test_savings_to_cost_ratio_is_gross_per_unit_of_operational_cost():
    row = _ops_row()
    assert row["savings_to_cost_ratio"] == pytest.approx(800.0 / 20.0)


def test_the_replay_threshold_is_the_one_it_was_given():
    """Frozen, never re-chosen per period -- re-optimising each hour would be
    hindsight no production desk has."""
    assert _ops_row(threshold=0.75)["threshold"] == 0.75


def test_capacity_scales_to_the_period_grain():
    """An hourly series must not silently use the daily capacity figure."""
    hourly = _ops_row(fraction=1 / 24)
    daily = _ops_row(fraction=1.0)
    assert hourly["review_capacity"] == pytest.approx(3 * 200 / 24)
    assert daily["review_capacity"] == pytest.approx(3 * 200)


def test_a_queue_larger_than_capacity_becomes_a_backlog():
    counts = impact.counts_from_arrays(Y_TRUE, Y_SCORE, AMOUNTS, 0.5)
    population = impact.population_from_arrays(Y_TRUE, AMOUNTS)
    # capacity of 1 alert against 2 flagged
    row = impact.derive_operations_row(
        "p", 0.5, counts, population, impact.ImpactAssumptions(),
        analyst_headcount=1, period_fraction_of_day=1 / 200,
    )
    assert row["review_capacity"] == pytest.approx(1.0)
    assert row["queue_backlog_count"] == pytest.approx(1.0)
    assert row["capacity_exceeded"] is True
    assert row["capacity_utilisation"] == pytest.approx(2.0)


def test_an_unstaffed_or_zero_length_period_is_rejected():
    counts = impact.counts_from_arrays(Y_TRUE, Y_SCORE, AMOUNTS, 0.5)
    population = impact.population_from_arrays(Y_TRUE, AMOUNTS)
    for kwargs in ({"analyst_headcount": 0}, {"period_fraction_of_day": 0}):
        with pytest.raises(ValueError):
            impact.derive_operations_row(
                "p", 0.5, counts, population, impact.ImpactAssumptions(), **kwargs)


def test_counts_and_money_answer_different_questions():
    """"How many did we catch" and "how much did we save" are separate columns
    on purpose -- a dashboard that conflates them hides the expensive misses."""
    row = _ops_row()
    assert row["caught_fraud_count"] == 1
    assert row["missed_fraud_count"] == 1
    assert row["detected_fraud_amount"] == pytest.approx(1000.0)
    assert row["missed_fraud_amount"] == pytest.approx(400.0)


def test_cumulative_totals_are_running_sums_in_the_order_given():
    rows = impact.add_cumulative([_ops_row(), _ops_row(), _ops_row()])
    assert [r["period_index"] for r in rows] == [0, 1, 2]
    assert [r["cumulative_net_savings"] for r in rows] == pytest.approx([780, 1560, 2340])
    assert rows[-1]["cumulative_operational_cost"] == pytest.approx(60.0)
    assert rows[-1]["cumulative_caught_fraud_count"] == 3


def test_the_cumulative_ratio_uses_totals_not_an_average_of_ratios():
    """Averaging per-period ratios would let a quiet hour with one cheap alert
    dominate a busy one -- the roll-up must divide the totals."""
    rows = impact.add_cumulative([_ops_row(), _ops_row()])
    last = rows[-1]
    assert last["cumulative_savings_to_cost_ratio"] == pytest.approx(
        last["cumulative_gross_savings"] / last["cumulative_operational_cost"]
    )


def test_cumulative_does_not_sort_for_you():
    """A cumulative series built on unsorted rows still looks like a plausible
    curve, so the helper must preserve caller order rather than hide the bug."""
    a = dict(_ops_row(), period_key="second")
    b = dict(_ops_row(), period_key="first")
    assert [r["period_key"] for r in impact.add_cumulative([a, b])] == ["second", "first"]


def test_the_operations_summary_rolls_up_the_whole_period():
    rows = impact.add_cumulative([_ops_row() for _ in range(4)])
    summary = impact.operations_summary(
        rows, impact.ImpactAssumptions(), {"period_grain": "hour"})
    assert summary["periods_observed"] == 4
    assert summary["total_caught_fraud_count"] == 4
    assert summary["total_net_savings"] == pytest.approx(4 * 780.0)
    assert summary["total_operational_cost"] == pytest.approx(4 * 20.0)
    assert summary["savings_to_cost_ratio"] == pytest.approx(40.0)
    assert summary["period_grain"] == "hour"


def test_the_summary_counts_periods_that_lost_money_or_overflowed():
    good = _ops_row()
    bad = impact.derive_operations_row(
        "p", 0.5,
        impact.counts_from_arrays(Y_TRUE, Y_SCORE, AMOUNTS, 0.5),
        impact.population_from_arrays(Y_TRUE, AMOUNTS),
        impact.ImpactAssumptions(prevention_rate=0.0),   # cost with no benefit
        analyst_headcount=1, period_fraction_of_day=1 / 200,
    )
    summary = impact.operations_summary(
        impact.add_cumulative([good, bad]), impact.ImpactAssumptions())
    assert summary["periods_net_negative"] == 1
    assert summary["periods_over_capacity"] == 1
    assert summary["periods_over_capacity_pct"] == pytest.approx(0.5)
    assert summary["peak_backlog_count"] == pytest.approx(1.0)


def test_the_summary_says_it_is_a_replay_not_a_production_record():
    summary = impact.operations_summary(
        impact.add_cumulative([_ops_row()]), impact.ImpactAssumptions())
    note = summary["operating_note"].lower()
    assert "replay, not a production record" in note
    assert "changes fraudster behaviour" in note
    assert "simulated" in summary["result_disclaimer"].lower()


def test_summarising_nothing_fails():
    with pytest.raises(ValueError):
        impact.operations_summary([], impact.ImpactAssumptions())


def test_the_step_to_timestamp_derivation_is_shared_with_silver():
    """Two notebooks derive a time axis from `step`. If they ever disagree, one
    dashboard's clock silently drifts from the other's."""
    import os

    from fraud_common import schema

    src = os.path.join(os.path.dirname(__file__), "..", "src")
    silver = open(os.path.join(src, "02_transform_silver.py"), encoding="utf-8").read()
    ops = open(os.path.join(src, "10_fraud_operations_timeseries.py"),
               encoding="utf-8").read()
    assert schema.STEP_TO_TIMESTAMP_INTERVAL in silver
    assert "step_to_timestamp_sql" in ops
    assert schema.SIMULATION_START_TS in schema.step_to_timestamp_sql()
