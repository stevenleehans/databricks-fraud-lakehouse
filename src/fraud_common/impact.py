"""Fraud loss prevention simulation -- model scores turned into money.

The rest of this package answers *is the model any good?* (PR-AUC, PSI,
precision/recall). This module answers the question a business sponsor actually
asks: **if we operated this model at threshold t, what would it be worth?**

Nothing here is a measured saving. Every number is a counterfactual built from
three ingredients:

  1. historical labels (``label_true``) -- what turned out to be fraud,
  2. model scores (``fraud_score``) -- what the model would have flagged,
  3. business assumptions (``ImpactAssumptions``) -- what an intervention is
     worth and what it costs.

Ingredient 3 is a guess supplied by the business, not a property of the data,
so the output is an *estimate under stated assumptions* and is labelled that way
everywhere it is written. See ``DISCLAIMER``.

Design mirrors the rest of ``fraud_common``: pure Python + numpy, no Spark and
no MLflow at import time, so the arithmetic is unit-testable on a laptop. The
notebook computes the confusion counts in Spark (one pass over the scored
table) and feeds them to :func:`derive_impact_row`, which owns the money maths.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

DISCLAIMER = (
    "Simulated estimate. Derived from historical labels, model scores and "
    "configurable business assumptions. NOT a confirmed or guaranteed saving; "
    "requires production validation before it is quoted as a business case."
)

# Candidate column names, most-specific first. The scored table this project
# writes uses the first entry of each list; the alternatives exist so the
# notebook also runs against a scoring table produced by someone else's job.
SCORE_COLUMN_CANDIDATES = (
    "fraud_score",
    "prediction_probability",
    "predicted_probability",
    "fraud_probability",
    "probability",
    "score",
)
LABEL_COLUMN_CANDIDATES = (
    "label_true",
    "is_fraud",
    "isFraud",
    "label",
    "y_true",
    "actual_fraud",
)
AMOUNT_COLUMN_CANDIDATES = (
    "amount",
    "transaction_amount",
    "txn_amount",
    "amt",
)
PREDICTED_LABEL_COLUMN_CANDIDATES = (
    "predicted_label",
    "prediction",
    "predicted_fraud",
    "is_flagged",
)


# --------------------------------------------------------------------------- #
# Assumptions
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ImpactAssumptions:
    """The four numbers the business owns, plus the currency they are in.

    Defaults are deliberately conservative placeholders for a PoC. They are the
    first thing a stakeholder should argue with -- which is the point of making
    them a named, versioned object that is written into every output row.
    """

    # Share of the fraudulent value that is actually recovered when a
    # fraudulent transaction is flagged. Never 1.0: a review takes time, some
    # funds are already gone, and some blocks are overridden.
    prevention_rate: float = 0.80

    # Fully-loaded cost of one manual review (analyst time, tooling, comms).
    # Charged on EVERY flagged transaction -- true and false positives alike,
    # because the queue does not know which is which until it is worked.
    cost_per_review: float = 5.00

    # Cost of wrongly interrupting a legitimate customer: support contact,
    # abandoned transaction, goodwill. Charged on false positives only, and
    # charged in ADDITION to the review cost.
    customer_friction_cost: float = 10.00

    # Transactions one analyst can clear in a working day. Converts the alert
    # volume into a staffing question rather than an abstract count.
    analyst_daily_capacity: int = 200

    currency: str = "USD"

    def validate(self) -> "ImpactAssumptions":
        if not 0.0 <= self.prevention_rate <= 1.0:
            raise ValueError("prevention_rate must be in [0, 1]")
        if self.cost_per_review < 0:
            raise ValueError("cost_per_review must be >= 0")
        if self.customer_friction_cost < 0:
            raise ValueError("customer_friction_cost must be >= 0")
        if self.analyst_daily_capacity <= 0:
            raise ValueError("analyst_daily_capacity must be > 0")
        if not self.currency:
            raise ValueError("currency must be a non-empty string")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def as_output_columns(self) -> Dict[str, Any]:
        """Assumption fields prefixed for the summary table.

        Every published figure carries the assumptions that produced it, so a
        row can never be read out of context.
        """
        return {
            "assumption_prevention_rate": float(self.prevention_rate),
            "assumption_cost_per_review": float(self.cost_per_review),
            "assumption_customer_friction_cost": float(self.customer_friction_cost),
            "assumption_analyst_daily_capacity": int(self.analyst_daily_capacity),
            "assumption_currency": str(self.currency),
        }


# --------------------------------------------------------------------------- #
# Column resolution
# --------------------------------------------------------------------------- #

def resolve_column(
    available: Iterable[str],
    candidates: Sequence[str],
    role: str,
    override: Optional[str] = None,
) -> str:
    """Pick the column to use for ``role``, case-insensitively.

    ``override`` wins outright, so the notebook's column-mapping cell can point
    at anything without touching this module.
    """
    present = list(available)
    lookup = {name.lower(): name for name in present}

    if override:
        if override in present:
            return override
        if override.lower() in lookup:
            return lookup[override.lower()]
        raise KeyError(
            f"column override {override!r} for the {role} column is not in the "
            f"input table. Available columns: {sorted(present)}"
        )

    for candidate in candidates:
        if candidate in present:
            return candidate
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise KeyError(
        f"could not find a {role} column. Tried {list(candidates)}; the table "
        f"has {sorted(present)}. Set the override in the notebook's COLUMN "
        "MAPPING cell."
    )


# --------------------------------------------------------------------------- #
# Threshold grid
# --------------------------------------------------------------------------- #

def threshold_grid(start: float = 0.05, stop: float = 0.95, step: float = 0.05) -> List[float]:
    """Inclusive grid of decision thresholds, rounded to avoid float dust.

    0.05 -> 0.95 rather than 0.0 -> 1.0: a threshold of 0 flags every
    transaction and a threshold of 1 flags none, and neither is an operating
    point anyone would consider.
    """
    if not 0.0 <= start <= 1.0 or not 0.0 <= stop <= 1.0:
        raise ValueError("threshold bounds must be in [0, 1]")
    if step <= 0:
        raise ValueError("threshold step must be > 0")
    if stop < start:
        raise ValueError("threshold stop must be >= start")
    count = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 6) for i in range(count)]


# --------------------------------------------------------------------------- #
# The money maths
# --------------------------------------------------------------------------- #

def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


#: Confusion/amount counts the caller must supply for one threshold. Spark
#: produces these in a single aggregation pass; the pandas path below produces
#: the identical dict from numpy arrays.
COUNT_FIELDS = (
    "flagged_transaction_count",
    "flagged_transaction_amount",
    "true_positive_count",
    "false_positive_count",
    "false_negative_count",
    "true_negative_count",
    "detected_fraud_amount",
    "missed_fraud_amount",
)

POPULATION_FIELDS = (
    "total_transactions",
    "total_transaction_amount",
    "actual_fraud_count",
    "actual_fraud_amount",
)


def derive_impact_row(
    threshold: float,
    counts: Dict[str, float],
    population: Dict[str, float],
    assumptions: ImpactAssumptions,
) -> Dict[str, Any]:
    """Turn one threshold's confusion counts into the business impact row.

    This function is the single definition of the financial model. Both the
    Spark path and the pandas path call it, so the dashboard and the tests can
    never disagree about how a saving was computed.

    The model, stated in full:

        gross_estimated_savings     = detected_fraud_amount * prevention_rate
        review_cost                 = flagged_count      * cost_per_review
        false_positive_friction_cost= false_positive_cnt * customer_friction_cost
        estimated_net_savings       = gross - review_cost - friction_cost

    Note what is deliberately NOT modelled: recovery on missed fraud, the
    time-value of a blocked payment, cross-sell lost to a churned customer, and
    any fixed cost of running the platform. Adding them is a business decision,
    not a code change -- and pretending they are zero is more honest than
    inventing a number for them.
    """
    missing = [f for f in COUNT_FIELDS if f not in counts]
    if missing:
        raise KeyError(f"missing count fields: {missing}")
    missing = [f for f in POPULATION_FIELDS if f not in population]
    if missing:
        raise KeyError(f"missing population fields: {missing}")

    flagged_count = float(counts["flagged_transaction_count"])
    true_positive_count = float(counts["true_positive_count"])
    false_positive_count = float(counts["false_positive_count"])
    false_negative_count = float(counts["false_negative_count"])
    true_negative_count = float(counts["true_negative_count"])
    detected_fraud_amount = float(counts["detected_fraud_amount"])
    missed_fraud_amount = float(counts["missed_fraud_amount"])

    # --- money -----------------------------------------------------------
    gross_estimated_savings = detected_fraud_amount * assumptions.prevention_rate
    review_cost = flagged_count * assumptions.cost_per_review
    false_positive_friction_cost = false_positive_count * assumptions.customer_friction_cost
    estimated_net_savings = (
        gross_estimated_savings - review_cost - false_positive_friction_cost
    )

    # --- model quality at this operating point ---------------------------
    precision = _safe_div(true_positive_count, true_positive_count + false_positive_count)
    recall = _safe_div(true_positive_count, true_positive_count + false_negative_count)
    false_positive_rate = _safe_div(
        false_positive_count, false_positive_count + true_negative_count
    )

    row: Dict[str, Any] = {
        "threshold": float(threshold),

        # population (identical on every row; repeated so a single row of the
        # simulation table is self-contained in a dashboard tooltip)
        "total_transactions": int(population["total_transactions"]),
        "total_transaction_amount": float(population["total_transaction_amount"]),
        "actual_fraud_count": int(population["actual_fraud_count"]),
        "actual_fraud_amount": float(population["actual_fraud_amount"]),

        # confusion at this threshold
        "flagged_transaction_count": int(flagged_count),
        "flagged_transaction_amount": float(counts["flagged_transaction_amount"]),
        "true_positive_count": int(true_positive_count),
        "false_positive_count": int(false_positive_count),
        "false_negative_count": int(false_negative_count),
        "true_negative_count": int(true_negative_count),

        # fraud value split
        "detected_fraud_amount": detected_fraud_amount,
        "missed_fraud_amount": missed_fraud_amount,

        # simulated money
        "gross_estimated_savings": gross_estimated_savings,
        "review_cost": review_cost,
        "false_positive_friction_cost": false_positive_friction_cost,
        "estimated_net_savings": estimated_net_savings,

        # rates
        "fraud_detection_rate": _safe_div(
            detected_fraud_amount, population["actual_fraud_amount"]
        ),
        "false_positive_rate": false_positive_rate,
        "precision": precision,
        "recall": recall,

        # operations
        "average_saving_per_review": _safe_div(estimated_net_savings, flagged_count),
        "estimated_review_days_required": _safe_div(
            flagged_count, assumptions.analyst_daily_capacity
        ),
        "alerts_per_1k_transactions": 1000.0 * _safe_div(
            flagged_count, population["total_transactions"]
        ),
    }
    row.update(assumptions.as_output_columns())
    return row


def counts_from_arrays(
    y_true: Sequence,
    y_score: Sequence,
    amounts: Sequence,
    threshold: float,
) -> Dict[str, float]:
    """Confusion counts and amount splits for one threshold (numpy path).

    ``>=`` is the flag rule everywhere in this project, including
    ``metrics.confusion_at_threshold`` and notebook 06's ``predicted_label``.
    """
    yt = np.asarray(y_true).astype(float).ravel()
    ys = np.asarray(y_score).astype(float).ravel()
    amt = np.asarray(amounts).astype(float).ravel()
    if not (yt.shape == ys.shape == amt.shape):
        raise ValueError(
            f"shape mismatch: y_true {yt.shape}, y_score {ys.shape}, amounts {amt.shape}"
        )
    bad = set(np.unique(yt)) - {0.0, 1.0}
    if bad:
        raise ValueError(f"y_true must be binary 0/1; found {sorted(bad)}")

    flagged = ys >= threshold
    fraud = yt == 1
    return {
        "flagged_transaction_count": float(flagged.sum()),
        "flagged_transaction_amount": float(amt[flagged].sum()),
        "true_positive_count": float((flagged & fraud).sum()),
        "false_positive_count": float((flagged & ~fraud).sum()),
        "false_negative_count": float((~flagged & fraud).sum()),
        "true_negative_count": float((~flagged & ~fraud).sum()),
        "detected_fraud_amount": float(amt[flagged & fraud].sum()),
        "missed_fraud_amount": float(amt[~flagged & fraud].sum()),
    }


def population_from_arrays(y_true: Sequence, amounts: Sequence) -> Dict[str, float]:
    yt = np.asarray(y_true).astype(float).ravel()
    amt = np.asarray(amounts).astype(float).ravel()
    return {
        "total_transactions": float(yt.size),
        "total_transaction_amount": float(amt.sum()),
        "actual_fraud_count": float((yt == 1).sum()),
        "actual_fraud_amount": float(amt[yt == 1].sum()),
    }


def simulate_thresholds(
    y_true: Sequence,
    y_score: Sequence,
    amounts: Sequence,
    assumptions: Optional[ImpactAssumptions] = None,
    thresholds: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Full sweep from in-memory arrays. Used by the tests and as the
    small-data fallback; the notebook uses the Spark aggregation instead."""
    assumptions = (assumptions or ImpactAssumptions()).validate()
    grid = list(thresholds) if thresholds is not None else threshold_grid()
    population = population_from_arrays(y_true, amounts)
    return [
        derive_impact_row(
            t, counts_from_arrays(y_true, y_score, amounts, t), population, assumptions
        )
        for t in grid
    ]


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #

def recommend_threshold(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The row with the highest ``estimated_net_savings``.

    Ties break toward the HIGHER threshold, i.e. the smaller review queue. Two
    thresholds worth the same money are not equally attractive to operate: the
    one that flags fewer transactions needs fewer analysts and annoys fewer
    customers.

    This is a starting point for a conversation, not a decision. Net savings is
    a single scalar built on assumed costs; a real operating point also has to
    respect the review capacity that actually exists and the organisation's
    tolerance for stopping good customers.
    """
    if not rows:
        raise ValueError("no simulated thresholds to choose from")
    return max(
        rows,
        key=lambda r: (float(r["estimated_net_savings"]), float(r["threshold"])),
    )


RECOMMENDATION_TEMPLATE = (
    "Recommended operating threshold is {threshold:.2f} because it produces the "
    "highest estimated net savings ({currency} {net:,.0f}) under the current "
    "business assumptions. At this threshold the model would flag "
    "{flagged:,} transactions ({review_days:,.1f} analyst-days), catch "
    "{currency} {detected:,.0f} of fraud and miss {currency} {missed:,.0f}. "
    "This should be validated with business stakeholders before production use."
)


def recommendation_text(row: Dict[str, Any], assumptions: ImpactAssumptions) -> str:
    return RECOMMENDATION_TEMPLATE.format(
        threshold=float(row["threshold"]),
        currency=assumptions.currency,
        net=float(row["estimated_net_savings"]),
        flagged=int(row["flagged_transaction_count"]),
        review_days=float(row["estimated_review_days_required"]),
        detected=float(row["detected_fraud_amount"]),
        missed=float(row["missed_fraud_amount"]),
    )


SUMMARY_FIELDS = (
    "recommended_threshold",
    "estimated_net_savings",
    "gross_estimated_savings",
    "detected_fraud_amount",
    "missed_fraud_amount",
    "flagged_transaction_count",
    "false_positive_count",
    "precision",
    "recall",
    "false_positive_rate",
    "estimated_review_days_required",
    "assumption_prevention_rate",
    "assumption_cost_per_review",
    "assumption_customer_friction_cost",
    "assumption_analyst_daily_capacity",
)


def build_summary(
    rows: Sequence[Dict[str, Any]],
    assumptions: ImpactAssumptions,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One-row executive summary built around the recommended threshold."""
    best = recommend_threshold(rows)
    summary: Dict[str, Any] = {
        "recommended_threshold": float(best["threshold"]),
        "estimated_net_savings": float(best["estimated_net_savings"]),
        "gross_estimated_savings": float(best["gross_estimated_savings"]),
        "review_cost": float(best["review_cost"]),
        "false_positive_friction_cost": float(best["false_positive_friction_cost"]),
        "detected_fraud_amount": float(best["detected_fraud_amount"]),
        "missed_fraud_amount": float(best["missed_fraud_amount"]),
        "total_transactions": int(best["total_transactions"]),
        "total_transaction_amount": float(best["total_transaction_amount"]),
        "actual_fraud_count": int(best["actual_fraud_count"]),
        "actual_fraud_amount": float(best["actual_fraud_amount"]),
        "flagged_transaction_count": int(best["flagged_transaction_count"]),
        "flagged_transaction_amount": float(best["flagged_transaction_amount"]),
        "true_positive_count": int(best["true_positive_count"]),
        "false_positive_count": int(best["false_positive_count"]),
        "false_negative_count": int(best["false_negative_count"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "false_positive_rate": float(best["false_positive_rate"]),
        "fraud_detection_rate": float(best["fraud_detection_rate"]),
        "average_saving_per_review": float(best["average_saving_per_review"]),
        "estimated_review_days_required": float(best["estimated_review_days_required"]),
        "threshold_count_simulated": len(rows),
        "recommendation_note": recommendation_text(best, assumptions),
        "selection_basis": "maximum estimated_net_savings across the simulated grid",
        "business_override_note": (
            "Business users may operate at a different threshold: a lower one "
            "catches more fraud but needs more review capacity, a higher one "
            "reduces customer friction at the cost of missed fraud. Compare "
            "estimated_review_days_required against the analyst capacity that "
            "actually exists before adopting this threshold."
        ),
        "result_disclaimer": DISCLAIMER,
    }
    summary.update(assumptions.as_output_columns())
    summary.update(context or {})

    missing = [f for f in SUMMARY_FIELDS if f not in summary]
    if missing:
        raise KeyError(f"summary is missing required fields: {missing}")
    return summary

# --------------------------------------------------------------------------- #
# Operational time series
#
# Dashboard 2 asks "which threshold should we run?" -- one population, many
# thresholds. This section asks the opposite question: "we DID ship it at one
# threshold; what did each hour of operating it look like?" -- one threshold,
# many periods. Same financial model, transposed.
# --------------------------------------------------------------------------- #

#: An operating period's alert queue is worked by a staffed team, so the
#: operational question is not only "what did it cost?" but "did it fit?".
DEFAULT_ANALYST_HEADCOUNT = 3


def derive_operations_row(
    period_key: str,
    threshold: float,
    counts: Dict[str, float],
    population: Dict[str, float],
    assumptions: ImpactAssumptions,
    analyst_headcount: int = DEFAULT_ANALYST_HEADCOUNT,
    period_fraction_of_day: float = 1.0,
) -> Dict[str, Any]:
    """One period (hour or day) of running the model at a FIXED threshold.

    Everything financial is delegated to :func:`derive_impact_row` so the
    time series and the threshold sweep cannot disagree about what a saving is.
    What is added here is the operational overlay:

    ``operational_cost``
        review cost + false-positive friction. This is the total cost of
        *running* the alert queue for the period, and it is the number the
        savings get compared against. Deliberately the same two components
        dashboard 2 subtracts -- a third definition of cost across two
        dashboards would be indefensible.

    ``savings_to_cost_ratio``
        gross savings per 1 unit of operational cost. The single most quotable
        operational number: "every dollar we spend reviewing returns N".

    ``review_capacity``
        what the staffed team could actually clear in this period. Alerts above
        it do not vanish -- they become ``queue_backlog_count``, which is how a
        real fraud desk fails: not by being wrong, but by being behind.

    ``period_fraction_of_day`` scales the daily analyst capacity to the period
    grain (1/24 for an hourly series). It is a parameter rather than a constant
    because an hourly series and a daily series must not silently use the same
    capacity number.
    """
    if analyst_headcount <= 0:
        raise ValueError("analyst_headcount must be > 0")
    if period_fraction_of_day <= 0:
        raise ValueError("period_fraction_of_day must be > 0")

    # `threshold` is frozen for the whole series -- it is a property of the
    # deployment being replayed, not something re-chosen per period.
    row = derive_impact_row(threshold, counts, population, assumptions)

    flagged = float(row["flagged_transaction_count"])
    operational_cost = float(row["review_cost"]) + float(row["false_positive_friction_cost"])
    review_capacity = (
        analyst_headcount * assumptions.analyst_daily_capacity * period_fraction_of_day
    )

    row.update({
        "period_key": str(period_key),
        "operational_cost": operational_cost,
        "savings_to_cost_ratio": _safe_div(row["gross_estimated_savings"], operational_cost),
        "net_savings_is_positive": bool(row["estimated_net_savings"] > 0),

        # capacity / staffing
        "analyst_headcount": int(analyst_headcount),
        "review_capacity": float(review_capacity),
        "capacity_utilisation": _safe_div(flagged, review_capacity),
        "queue_backlog_count": float(max(flagged - review_capacity, 0.0)),
        "capacity_exceeded": bool(flagged > review_capacity),

        # counts, for the "how many did we catch" tiles -- the money columns
        # answer "how much", these answer "how many"
        "caught_fraud_count": int(row["true_positive_count"]),
        "missed_fraud_count": int(row["false_negative_count"]),
    })
    return row


CUMULATIVE_SOURCES = (
    ("estimated_net_savings", "cumulative_net_savings"),
    ("gross_estimated_savings", "cumulative_gross_savings"),
    ("operational_cost", "cumulative_operational_cost"),
    ("detected_fraud_amount", "cumulative_detected_fraud_amount"),
    ("missed_fraud_amount", "cumulative_missed_fraud_amount"),
    ("caught_fraud_count", "cumulative_caught_fraud_count"),
    ("missed_fraud_count", "cumulative_missed_fraud_count"),
    ("flagged_transaction_count", "cumulative_flagged_count"),
)


def add_cumulative(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Running totals in the order given -- the "since go-live" view.

    The caller must pass the rows already sorted by time. Sorting here would
    hide a caller that forgot to, and a cumulative series built on unsorted
    input is wrong in a way that still looks like a plausible curve.
    """
    out: List[Dict[str, Any]] = []
    totals = {target: 0.0 for _, target in CUMULATIVE_SOURCES}
    for index, row in enumerate(rows):
        new_row = dict(row)
        for source, target in CUMULATIVE_SOURCES:
            totals[target] += float(row[source])
            new_row[target] = totals[target]
        new_row["period_index"] = index
        new_row["cumulative_savings_to_cost_ratio"] = _safe_div(
            totals["cumulative_gross_savings"], totals["cumulative_operational_cost"]
        )
        out.append(new_row)
    return out


def operations_summary(
    rows: Sequence[Dict[str, Any]],
    assumptions: ImpactAssumptions,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Whole-period roll-up: what shipping this model would have looked like."""
    if not rows:
        raise ValueError("no operating periods to summarise")
    last = rows[-1]
    periods = len(rows)
    breached = [r for r in rows if r["capacity_exceeded"]]
    loss_making = [r for r in rows if not r["net_savings_is_positive"]]

    summary: Dict[str, Any] = {
        "periods_observed": periods,
        "period_grain": str((context or {}).get("period_grain", "unknown")),
        "first_period": str(rows[0]["period_key"]),
        "last_period": str(last["period_key"]),

        "total_transactions": int(sum(r["total_transactions"] for r in rows)),
        "total_alerts": int(sum(r["flagged_transaction_count"] for r in rows)),
        "total_caught_fraud_count": int(last["cumulative_caught_fraud_count"]),
        "total_missed_fraud_count": int(last["cumulative_missed_fraud_count"]),
        "total_detected_fraud_amount": float(last["cumulative_detected_fraud_amount"]),
        "total_missed_fraud_amount": float(last["cumulative_missed_fraud_amount"]),

        "total_gross_savings": float(last["cumulative_gross_savings"]),
        "total_operational_cost": float(last["cumulative_operational_cost"]),
        "total_net_savings": float(last["cumulative_net_savings"]),
        "savings_to_cost_ratio": float(last["cumulative_savings_to_cost_ratio"]),

        "avg_alerts_per_period": _safe_div(
            sum(r["flagged_transaction_count"] for r in rows), periods),
        "peak_alerts_in_a_period": float(
            max(r["flagged_transaction_count"] for r in rows)),
        "periods_over_capacity": len(breached),
        "periods_over_capacity_pct": _safe_div(len(breached), periods),
        "peak_backlog_count": float(max(r["queue_backlog_count"] for r in rows)),
        "periods_net_negative": len(loss_making),

        "result_disclaimer": DISCLAIMER,
        "operating_note": (
            "A replay, not a production record. The model is applied at one "
            "frozen threshold to transactions it did not influence: in reality "
            "blocking fraud changes fraudster behaviour, and reviewed alerts "
            "change what the next model sees."
        ),
    }
    summary.update(assumptions.as_output_columns())
    summary.update(context or {})
    return summary
