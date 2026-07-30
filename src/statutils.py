"""Small statistics helpers shared across feature tiers.

Every function here returns math.nan (never raises) when the input is too
small or degenerate to define the statistic. Feature functions are expected
to propagate NaN rather than crash the whole extraction run over one group,
per the project's "leave NaN as NaN, don't impute" output policy.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def coefficient_of_variation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return math.nan
    mean = float(np.mean(values))
    if mean == 0:
        return math.nan
    return float(np.std(values, ddof=1)) / mean


def percentile_ratio(values: Sequence[float], hi: float, lo: float) -> float:
    if len(values) < 2:
        return math.nan
    lo_val = float(np.percentile(values, lo))
    if lo_val == 0:
        return math.nan
    hi_val = float(np.percentile(values, hi))
    return hi_val / lo_val


def shannon_entropy(counts: Sequence[int], base: float = 2.0) -> float:
    total = sum(counts)
    if total == 0:
        return math.nan
    entropy = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c / total
        entropy -= p * math.log(p, base)
    return entropy


def binned_entropy(values: Sequence[float], n_bins: int, base: float = 2.0) -> float:
    """Shannon entropy of `values` after bucketing into n_bins equal-width
    bins spanning [min(values), max(values)]. Bin edges are derived from the
    group's own observed range (not a fixed global scale), so the entropy
    measures the *shape* of that group's pause distribution.
    """
    if len(values) < 2:
        return math.nan
    arr = np.asarray(values, dtype=float)
    if arr.min() == arr.max():
        return 0.0  # all identical -> a single bin holds everything, zero entropy
    hist, _ = np.histogram(arr, bins=n_bins)
    return shannon_entropy(hist.tolist(), base=base)


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.nan
    return numerator / denominator
