"""Evaluation metrics for a severely imbalanced binary target.

Model selection uses **PR-AUC** (average precision). With ~0.13 % positives,
accuracy is ~99.87 % for a model that predicts "never fraud", and ROC-AUC is
optimistic because the huge true-negative pool dominates the false-positive
rate. PR-AUC only rewards ranking real fraud above real legitimate traffic.

ROC-AUC, precision, recall, F1, FPR and the confusion matrix are all reported,
but never used to pick the winner.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

PRIMARY_METRIC = "pr_auc"
PRIMARY_METRIC_GREATER_IS_BETTER = True


def _as_arrays(y_true: Sequence, y_score: Sequence):
    yt = np.asarray(y_true).astype(float).ravel()
    ys = np.asarray(y_score).astype(float).ravel()
    if yt.shape != ys.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_score {ys.shape}")
    if yt.size == 0:
        raise ValueError("empty evaluation set")
    bad = set(np.unique(yt)) - {0.0, 1.0}
    if bad:
        raise ValueError(f"y_true must be binary 0/1; found {sorted(bad)}")
    return yt, ys


def pr_auc(y_true: Sequence, y_score: Sequence) -> float:
    """Average precision -- the step-wise (non-interpolated) area under PR.

    Returns NaN when a split contains no positives; that is a data problem the
    caller must surface, not a zero to be averaged into a leaderboard.
    """
    yt, ys = _as_arrays(y_true, y_score)
    if yt.sum() == 0:
        return float("nan")
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(yt, ys))


def roc_auc(y_true: Sequence, y_score: Sequence) -> float:
    yt, ys = _as_arrays(y_true, y_score)
    if yt.sum() == 0 or yt.sum() == yt.size:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(yt, ys))


def confusion_at_threshold(y_true: Sequence, y_score: Sequence, threshold: float):
    yt, ys = _as_arrays(y_true, y_score)
    pred = (ys >= threshold).astype(float)
    tp = int(np.sum((pred == 1) & (yt == 1)))
    fp = int(np.sum((pred == 1) & (yt == 0)))
    tn = int(np.sum((pred == 0) & (yt == 0)))
    fn = int(np.sum((pred == 0) & (yt == 1)))
    return tp, fp, tn, fn


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def evaluate(
    y_true: Sequence,
    y_score: Sequence,
    threshold: float,
    amounts: Optional[Sequence] = None,
    prefix: str = "",
) -> Dict[str, float]:
    """Full metric bundle at one operating point.

    ``amounts`` (optional) enables the money-weighted view: what share of the
    fraudulent value the alerts would have intercepted. That is usually the
    number a payments business actually cares about.
    """
    yt, ys = _as_arrays(y_true, y_score)
    tp, fp, tn, fn = confusion_at_threshold(yt, ys, threshold)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    out: Dict[str, float] = {
        "pr_auc": pr_auc(yt, ys),
        "roc_auc": roc_auc(yt, ys),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": _safe_div(fp, fp + tn),
        "false_negative_rate": _safe_div(fn, fn + tp),
        "true_positives": float(tp),
        "false_positives": float(fp),
        "true_negatives": float(tn),
        "false_negatives": float(fn),
        "threshold": float(threshold),
        "n_rows": float(yt.size),
        "n_positives": float(yt.sum()),
        "prevalence": _safe_div(yt.sum(), yt.size),
        "predicted_positive_rate": _safe_div((ys >= threshold).sum(), yt.size),
        "alerts_per_1k_transactions": 1000.0 * _safe_div((ys >= threshold).sum(), yt.size),
    }

    if amounts is not None:
        amt = np.asarray(amounts).astype(float).ravel()
        if amt.shape != yt.shape:
            raise ValueError("amounts must align with y_true")
        fraud_value = float(amt[yt == 1].sum())
        caught = float(amt[(yt == 1) & (ys >= threshold)].sum())
        out["fraud_amount_total"] = fraud_value
        out["fraud_amount_captured"] = caught
        out["fraud_amount_capture_rate"] = _safe_div(caught, fraud_value)
        out["legit_amount_blocked"] = float(amt[(yt == 0) & (ys >= threshold)].sum())

    if prefix:
        out = {f"{prefix}{k}": v for k, v in out.items()}
    return out


def choose_threshold_for_max_fpr(
    y_true: Sequence,
    y_score: Sequence,
    max_fpr: float,
    default: float = 0.5,
) -> float:
    """Highest-recall threshold whose FPR stays inside the alert budget.

    Selected on the *validation* split only, then frozen. Tuning the threshold
    on test would leak the test set into the operating point.
    """
    yt, ys = _as_arrays(y_true, y_score)
    if not 0.0 < max_fpr < 1.0:
        raise ValueError("max_fpr must be in (0, 1)")
    if yt.sum() == 0 or yt.sum() == yt.size:
        return float(default)

    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(yt, ys)
    feasible = fpr <= max_fpr
    if not feasible.any():
        return float(default)
    # roc_curve is sorted by decreasing threshold, so the last feasible index
    # is the most permissive (highest recall) point inside the budget.
    idx = int(np.max(np.flatnonzero(feasible)))
    chosen = float(thresholds[idx])
    # roc_curve prepends an artificial threshold of inf (>= max score + 1).
    if not np.isfinite(chosen):
        chosen = float(np.max(ys))
    return float(min(max(chosen, 0.0), 1.0))


def recall_at_threshold(y_true: Sequence, y_score: Sequence, threshold: float) -> float:
    tp, _fp, _tn, fn = confusion_at_threshold(y_true, y_score, threshold)
    return _safe_div(tp, tp + fn)


def precision_at_threshold(y_true: Sequence, y_score: Sequence, threshold: float) -> float:
    tp, fp, _tn, _fn = confusion_at_threshold(y_true, y_score, threshold)
    return _safe_div(tp, tp + fp)


def compare_models(results: Dict[str, Dict[str, float]], metric: str = PRIMARY_METRIC):
    """Pick the winner by ``metric``; NaN never wins.

    ``results`` maps run name -> metric dict (validation metrics, not test).
    """
    if not results:
        raise ValueError("no model results to compare")
    scored = {
        name: m.get(metric, float("nan"))
        for name, m in results.items()
    }
    finite = {k: v for k, v in scored.items() if v == v}  # drop NaN
    if not finite:
        raise ValueError(f"no model produced a finite {metric}")
    best = max(finite, key=finite.get) if PRIMARY_METRIC_GREATER_IS_BETTER \
        else min(finite, key=finite.get)
    return best, finite[best], scored
