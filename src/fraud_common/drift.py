"""Drift statistics: PSI, categorical shift, missingness and score movement.

Three *different* things get measured here and the monitoring tables keep them
apart on purpose:

  feature drift        the input distribution moved. Says nothing on its own
                       about whether the model got worse -- a bank running a
                       marketing campaign shifts `amount` without any change
                       in fraud behaviour.
  prediction drift     the score distribution or alert rate moved. Observable
                       immediately, still not proof of degradation.
  performance change   PR-AUC / precision / recall actually fell. The only
                       direct evidence, and it needs labels, which in fraud
                       arrive days-to-weeks late (chargebacks, investigations).

Conflating the first with the third is the most common monitoring error, so
``drift_kind`` is a first-class column in every table this module feeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

GREEN = "GREEN"
AMBER = "AMBER"
RED = "RED"

FEATURE_DRIFT = "feature_drift"
PREDICTION_DRIFT = "prediction_drift"
PERFORMANCE = "performance"

# Conventional PSI reading, carried into the dashboard legend:
#   < 0.10  stable | 0.10-0.25 moderate shift, investigate | > 0.25 significant
DEFAULT_AMBER = 0.10
DEFAULT_RED = 0.25


@dataclass(frozen=True)
class NumericBinning:
    """Reference bin edges. Computed once on the training period and reused
    for every monitoring window -- re-deriving edges per window would compare
    each window against itself and report zero drift forever."""

    feature: str
    edges: List[float]
    reference_proportions: List[float]

    def as_dict(self) -> dict:
        return {
            "feature": self.feature,
            "edges": list(self.edges),
            "reference_proportions": list(self.reference_proportions),
        }


def _clean(values: Sequence) -> np.ndarray:
    arr = np.asarray(values, dtype="float64").ravel()
    return arr[np.isfinite(arr)]


def build_numeric_binning(
    feature: str,
    reference: Sequence,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> NumericBinning:
    """Quantile bin edges from the reference sample.

    Quantiles rather than equal width: `amount` spans 0 to ~9.2e7 with a long
    tail, so equal-width bins would put ~99 % of mass in bin 0 and make PSI
    numerically meaningless.
    """
    ref = _clean(reference)
    if ref.size == 0:
        raise ValueError(f"reference sample for {feature!r} is empty")
    if bins < 2:
        raise ValueError("bins must be >= 2")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))

    # Low-cardinality features (binary flags like `is_night`, small integer
    # features) collapse to one or two unique quantile edges. Replacing the
    # ends with +/-inf would then leave *no* interior boundary, i.e. a single
    # bin, which makes PSI identically zero and hides real drift. Rebuild the
    # edges from the distinct values instead, splitting at their midpoints.
    distinct = np.unique(ref)
    if edges.size < 3 and 1 < distinct.size <= max(bins, 2):
        midpoints = (distinct[:-1] + distinct[1:]) / 2.0
        edges = np.concatenate(([distinct[0] - 1.0], midpoints, [distinct[-1] + 1.0]))
    elif edges.size < 2:
        # Genuinely constant feature: a nominal 2-bin split keeps PSI defined
        # and lets any future spread show up rather than crashing.
        edges = np.array([ref[0] - 0.5, ref[0] + 0.5])

    edges = np.asarray(edges, dtype="float64")
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    ref_props = _to_proportions(ref_counts, epsilon)
    return NumericBinning(feature, [float(e) for e in edges], [float(p) for p in ref_props])


def bucket_case_sql(column: str, edges: Sequence[float]) -> str:
    """Spark SQL CASE mapping a value onto its frozen reference bin index.

    Lives here rather than in the notebook so it can be tested. The empty
    interior case is the one that matters: with only the two infinite edges
    there is no boundary to test, and naively joining zero WHEN clauses emits
    `CASE  ELSE 0 END`, which is a SQL parse error rather than a wrong number.
    """
    interior = [e for e in list(edges)[1:-1] if np.isfinite(e)]
    if not interior:
        return "0"          # single bin: every row lands in bucket 0
    clauses = " ".join(
        f"WHEN {column} < {edge!r} THEN {index}"
        for index, edge in enumerate(interior)
    )
    return f"CASE {clauses} ELSE {len(interior)} END"


def _to_proportions(counts: np.ndarray, epsilon: float) -> np.ndarray:
    counts = np.asarray(counts, dtype="float64")
    total = counts.sum()
    if total <= 0:
        return np.full(counts.shape, epsilon)
    props = counts / total
    # Floor empty bins: PSI's log ratio is undefined at zero, and an emptied
    # bin is exactly the signal we want to keep, not drop.
    return np.maximum(props, epsilon)


def psi_from_proportions(
    reference_props: Sequence[float],
    current_props: Sequence[float],
) -> float:
    ref = np.asarray(reference_props, dtype="float64")
    cur = np.asarray(current_props, dtype="float64")
    if ref.shape != cur.shape:
        raise ValueError("proportion vectors must have the same length")
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_numeric(
    binning: NumericBinning,
    current: Sequence,
    epsilon: float = 1e-6,
) -> float:
    """PSI of ``current`` against the frozen reference binning."""
    cur = _clean(current)
    if cur.size == 0:
        return float("nan")
    counts, _ = np.histogram(cur, bins=np.asarray(binning.edges, dtype="float64"))
    cur_props = _to_proportions(counts, epsilon)
    return psi_from_proportions(binning.reference_proportions, cur_props)


def category_proportions(
    values: Sequence,
    categories: Sequence[str],
    epsilon: float = 1e-6,
) -> Dict[str, float]:
    """Proportion per category, over a *fixed* category list.

    The list is fixed so a category vanishing in a window still contributes to
    PSI instead of silently changing the vector length.
    """
    vals = [v for v in values if v is not None and v == v]
    total = len(vals)
    counts = {c: 0 for c in categories}
    unseen = 0
    for v in vals:
        key = str(v)
        if key in counts:
            counts[key] += 1
        else:
            unseen += 1
    props = {c: max(counts[c] / total, epsilon) if total else epsilon for c in categories}
    props["__unseen__"] = max(unseen / total, epsilon) if total else epsilon
    return props


def psi_categorical(
    reference_props: Dict[str, float],
    current_props: Dict[str, float],
) -> float:
    keys = sorted(set(reference_props) | set(current_props))
    ref = [reference_props.get(k, 0.0) for k in keys]
    cur = [current_props.get(k, 0.0) for k in keys]
    eps = 1e-6
    ref = [max(v, eps) for v in ref]
    cur = [max(v, eps) for v in cur]
    return psi_from_proportions(ref, cur)


def psi_status(value: float, amber: float = DEFAULT_AMBER, red: float = DEFAULT_RED) -> str:
    if value != value:          # NaN -> no data for this window
        return AMBER
    if value >= red:
        return RED
    if value >= amber:
        return AMBER
    return GREEN


def missing_rate(values: Sequence) -> float:
    arr = list(values)
    if not arr:
        return float("nan")
    missing = sum(1 for v in arr if v is None or (isinstance(v, float) and v != v))
    return missing / len(arr)


def missing_rate_status(delta: float, amber: float, red: float) -> str:
    d = abs(delta)
    if d != d:
        return AMBER
    if d >= red:
        return RED
    if d >= amber:
        return AMBER
    return GREEN


def score_distribution(scores: Sequence, threshold: float) -> Dict[str, float]:
    """Prediction-drift summary for one window."""
    arr = _clean(scores)
    if arr.size == 0:
        return {
            "n_scored": 0.0,
            "avg_score": float("nan"),
            "median_score": float("nan"),
            "p90_score": float("nan"),
            "p99_score": float("nan"),
            "max_score": float("nan"),
            "std_score": float("nan"),
            "predicted_positive_rate": float("nan"),
            "predicted_positive_count": 0.0,
        }
    positives = float(np.sum(arr >= threshold))
    return {
        "n_scored": float(arr.size),
        "avg_score": float(np.mean(arr)),
        "median_score": float(np.median(arr)),
        "p90_score": float(np.quantile(arr, 0.90)),
        "p99_score": float(np.quantile(arr, 0.99)),
        "max_score": float(np.max(arr)),
        "std_score": float(np.std(arr)),
        "predicted_positive_rate": positives / arr.size,
        "predicted_positive_count": positives,
    }


SCORE_HISTOGRAM_EDGES: List[float] = [i / 20.0 for i in range(21)]  # 20 bins of 0.05


def score_histogram(
    scores: Sequence,
    edges: Optional[Sequence[float]] = None,
) -> List[Dict[str, float]]:
    """Rows for the score-distribution-over-time dashboard tile."""
    arr = _clean(scores)
    edge_list = list(edges) if edges is not None else list(SCORE_HISTOGRAM_EDGES)
    counts, _ = np.histogram(arr, bins=np.asarray(edge_list, dtype="float64"))
    total = float(counts.sum()) or 1.0
    return [
        {
            "bucket_index": int(i),
            "bucket_lower": float(edge_list[i]),
            "bucket_upper": float(edge_list[i + 1]),
            "row_count": float(counts[i]),
            "row_share": float(counts[i]) / total,
        }
        for i in range(len(counts))
    ]


def make_windows(
    min_step: int,
    max_step: int,
    n_windows: int,
) -> List[Dict[str, int]]:
    """Split the monitoring period into equal-width `step` windows.

    Equal width in *time*, not in row count, so the volume tile can actually
    show a volume change.
    """
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")
    if max_step < min_step:
        raise ValueError("max_step must be >= min_step")
    span = max_step - min_step + 1
    n = min(n_windows, span)
    size = span / n
    windows = []
    for i in range(n):
        start = min_step + int(round(i * size))
        end = min_step + int(round((i + 1) * size)) - 1
        end = min(end, max_step)
        if end < start:
            continue
        windows.append({
            "window_index": i,
            "window_id": f"w{i:02d}",
            "start_step": int(start),
            "end_step": int(end),
        })
    return windows


def overall_status(statuses: Sequence[str]) -> str:
    """Roll individual signals up into one traffic light (worst wins)."""
    s = set(statuses)
    if RED in s:
        return RED
    if AMBER in s:
        return AMBER
    return GREEN
