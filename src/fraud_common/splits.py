"""Chronological splitting and deterministic sampling.

Fraud detection is a forecasting problem: a model is always asked about
transactions that happen *after* the ones it learned from. A random split lets
the model see the future, which inflates PR-AUC and makes the PoC dishonest.
Everything here partitions strictly on ``step`` (1 step == 1 simulated hour).

Sampling uses SHA-256 of the row key rather than a random number generator so
that the same rows are selected in Spark and in pandas, on every rerun, on any
cluster size -- ``.sample(seed=...)`` alone is not stable across repartitioning.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

TRAIN = "train"
VALID = "valid"
TEST = "test"
SPLIT_ORDER: List[str] = [TRAIN, VALID, TEST]

# 15 hex chars == 60 bits of the digest, comfortably inside a signed BIGINT so
# the identical arithmetic works in Spark SQL via conv(substr(...), 16, 10).
# The divisor is 2**60 exactly -- a power of two is representable without loss
# in IEEE-754 double, so Python and Spark compute byte-identical ratios.
_HASH_HEX_CHARS = 15
_HASH_DENOM_INT = 1 << (4 * _HASH_HEX_CHARS)   # 1152921504606846976
_HASH_MAX = float(_HASH_DENOM_INT)


@dataclass(frozen=True)
class SplitBoundaries:
    """Inclusive upper `step` of the train and validation periods."""

    min_step: int
    train_end_step: int
    valid_end_step: int
    max_step: int

    def __post_init__(self) -> None:
        if not self.min_step <= self.train_end_step < self.valid_end_step < self.max_step:
            raise ValueError(
                "split boundaries must be strictly increasing and leave every "
                f"split non-empty: {self}"
            )

    def assign(self, step: int) -> str:
        if step <= self.train_end_step:
            return TRAIN
        if step <= self.valid_end_step:
            return VALID
        return TEST

    def sql_expression(self, step_col: str = "step") -> str:
        """CASE expression for Spark ``selectExpr`` / ``withColumn``."""
        return (
            f"CASE WHEN {step_col} <= {self.train_end_step} THEN '{TRAIN}' "
            f"WHEN {step_col} <= {self.valid_end_step} THEN '{VALID}' "
            f"ELSE '{TEST}' END"
        )

    def as_dict(self) -> dict:
        return {
            "min_step": self.min_step,
            "train_end_step": self.train_end_step,
            "valid_end_step": self.valid_end_step,
            "max_step": self.max_step,
        }


def compute_boundaries(
    min_step: int,
    max_step: int,
    train_fraction: float,
    valid_fraction: float,
) -> SplitBoundaries:
    """Cut the observed step range into train / valid / test periods.

    Splits on the step *range*, not on row count. PaySim's hourly volume is
    wildly uneven (a pronounced overnight trough), so a row-count split would
    place a boundary mid-hour and let the same hour appear on both sides of it.
    A range split guarantees every hour lands wholly in one period.
    """
    if max_step <= min_step:
        raise ValueError(f"max_step ({max_step}) must exceed min_step ({min_step})")
    span = max_step - min_step + 1
    if span < 3:
        raise ValueError("need at least 3 distinct steps to form three splits")

    train_end = min_step + int(span * train_fraction) - 1
    valid_end = min_step + int(span * (train_fraction + valid_fraction)) - 1

    # Clamp so that no split is empty even on tiny/degenerate ranges.
    train_end = max(min_step, min(train_end, max_step - 2))
    valid_end = max(train_end + 1, min(valid_end, max_step - 1))

    return SplitBoundaries(min_step, train_end, valid_end, max_step)


# --------------------------------------------------------------------------- #
# Deterministic, engine-portable sampling
# --------------------------------------------------------------------------- #

def hash_unit_interval(key: str, seed: int) -> float:
    """Map ``key`` to a stable pseudo-uniform value in [0, 1).

    Deliberately mirrors the Spark SQL in ``sample_sql_expression`` bit for bit
    so a locally-computed sample matches the cluster's.
    """
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:_HASH_HEX_CHARS], 16) / _HASH_MAX


def sample_sql_expression(
    key_col: str,
    seed: int,
    fraction: float,
    keep_all_when: str | None = None,
) -> str:
    """Spark SQL predicate equivalent to ``hash_unit_interval(key) < fraction``.

    ``keep_all_when`` is a SQL boolean (typically ``is_fraud = 1``) whose rows
    bypass sampling entirely -- the minority class is never thinned.
    """
    core = (
        f"(CONV(SUBSTR(SHA2(CONCAT('{seed}', ':', {key_col}), 256), 1, "
        f"{_HASH_HEX_CHARS}), 16, 10) / CAST({_HASH_DENOM_INT} AS DOUBLE))"
        f" < {fraction}"
    )
    if keep_all_when:
        return f"(({keep_all_when}) OR ({core}))"
    return core


def deterministic_keep_mask(
    keys: Sequence[str],
    fraction: float,
    seed: int,
) -> List[bool]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    return [hash_unit_interval(k, seed) < fraction for k in keys]


def sample_training_frame(
    df,
    key_col: str,
    label_col: str,
    negative_fraction: float,
    seed: int,
):
    """Down-sample the majority class of a training frame, reproducibly.

    Every fraud row in the training period is retained -- with ~0.13 %
    prevalence, discarding positives is the fastest way to destroy recall.
    Only legitimate rows are thinned, which is what makes the run fit inside
    Free Edition compute.

    Validation and test frames are deliberately *not* passed through this
    function: they must keep their natural prevalence or the reported
    precision is meaningless.
    """
    if label_col not in df.columns:
        raise KeyError(f"label column {label_col!r} not present")
    if key_col not in df.columns:
        raise KeyError(f"key column {key_col!r} not present")

    keys = df[key_col].astype(str).tolist()
    keep = deterministic_keep_mask(keys, negative_fraction, seed)
    is_positive = df[label_col].astype(int).values == 1
    mask = [p or k for p, k in zip(is_positive, keep)]
    return df.loc[mask]


def split_frames(df, boundaries: SplitBoundaries, step_col: str = "step"):
    """Return ``(train, valid, test)`` pandas frames, chronologically ordered."""
    step = df[step_col]
    train = df.loc[step <= boundaries.train_end_step]
    valid = df.loc[
        (step > boundaries.train_end_step) & (step <= boundaries.valid_end_step)
    ]
    test = df.loc[step > boundaries.valid_end_step]
    return train, valid, test


def assert_chronological(frames: Iterable[Tuple[str, object]], step_col: str = "step") -> None:
    """Guard that no split overlaps another in time.

    ``frames`` is an ordered iterable of ``(name, dataframe)`` in the intended
    chronological order.
    """
    previous_name = None
    previous_max = None
    for name, frame in frames:
        if len(frame) == 0:
            raise ValueError(f"split {name!r} is empty")
        lo = int(frame[step_col].min())
        hi = int(frame[step_col].max())
        if previous_max is not None and lo <= previous_max:
            raise ValueError(
                f"chronological violation: {name!r} starts at step {lo} but "
                f"{previous_name!r} extends to step {previous_max}"
            )
        previous_name, previous_max = name, hi
