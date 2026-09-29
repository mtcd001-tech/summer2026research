"""Unit tests for scripts/analysis/feature_importance.py's pure-logic
pieces: Cohen's d, correlation-based redundancy grouping, and rank
computation. Does not exercise permutation_importance/real runs/ data --
that path is exercised manually against the real dataset (see the report),
not worth mocking joblib models for here."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "analysis" / "feature_importance.py"
_spec = importlib.util.spec_from_file_location("feature_importance", _MODULE_PATH)
feature_importance = importlib.util.module_from_spec(_spec)
sys.modules["feature_importance"] = feature_importance
_spec.loader.exec_module(feature_importance)


def test_cohens_d_hand_computed():
    # bona fide (session 1): values [1, 2, 3] -> mean=2, var=1 (ddof=1)
    # session 2: values [4, 5, 6] -> mean=5, var=1
    # pooled_sd = sqrt((1+1)/2) = 1 -> d = (5-2)/1 = 3
    wide = pd.DataFrame({
        "session": [1, 1, 1, 2, 2, 2],
        "f1": [1, 2, 3, 4, 5, 6],
    })
    result = feature_importance.compute_cohens_d(wide, ["f1"])
    assert result.loc["f1", "s2"] == pytest.approx(3.0)
    assert result.loc["f1", "cohens_d_max"] == pytest.approx(3.0)


def test_cohens_d_sign_is_preserved():
    # session 2 has a LOWER mean than bona fide -> d should be negative
    wide = pd.DataFrame({
        "session": [1, 1, 1, 2, 2, 2],
        "f1": [4, 5, 6, 1, 2, 3],
    })
    result = feature_importance.compute_cohens_d(wide, ["f1"])
    assert result.loc["f1", "s2"] < 0


def test_cohens_d_max_is_max_absolute_value_across_conditions():
    # each group has some variance (not a constant) so pooled_sd is never
    # zero; s2/s4 match bona fide (d~0), s3/s5 swing away from it in
    # opposite directions by different amounts -- cohens_d_max must be
    # whichever of those has the larger |d|, regardless of sign.
    wide = pd.DataFrame({
        "session": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "f1": [1, 2, 1, 2, 9, 11, 1, 2, -7, -9],
    })
    result = feature_importance.compute_cohens_d(wide, ["f1"])
    abs_by_condition = result[["s2", "s3", "s4", "s5"]].loc["f1"].abs()
    assert result.loc["f1", "cohens_d_max"] == pytest.approx(abs_by_condition.max())
    assert abs_by_condition["s2"] == pytest.approx(0.0)
    assert abs_by_condition["s4"] == pytest.approx(0.0)
    assert result.loc["f1", "cohens_d_max"] > abs_by_condition["s3"]  # s5 (below) wins here


def test_cohens_d_zero_pooled_sd_is_nan_not_inf():
    # constant values everywhere -> zero variance -> undefined d, not a
    # divide-by-zero crash
    wide = pd.DataFrame({
        "session": [1, 1, 2, 2],
        "f1": [5, 5, 5, 5],
    })
    result = feature_importance.compute_cohens_d(wide, ["f1"])
    assert np.isnan(result.loc["f1", "s2"])


def test_correlation_groups_finds_duplicate_pair():
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    wide = pd.DataFrame({
        "a": base,
        "b": base + rng.normal(scale=0.01, size=200),  # near-identical to a
        "c": rng.normal(size=200),  # independent
    })
    corr, groups = feature_importance.compute_correlation_groups(wide, ["a", "b", "c"], threshold=0.9)
    assert groups == [["a", "b"]]
    assert "c" not in [f for g in groups for f in g]


def test_correlation_groups_transitive_cluster():
    rng = np.random.default_rng(1)
    base = rng.normal(size=200)
    wide = pd.DataFrame({
        "a": base,
        "b": base + rng.normal(scale=0.01, size=200),
        "c": base + rng.normal(scale=0.02, size=200),
        "d": rng.normal(size=200),
    })
    corr, groups = feature_importance.compute_correlation_groups(wide, ["a", "b", "c", "d"], threshold=0.9)
    assert len(groups) == 1
    assert sorted(groups[0]) == ["a", "b", "c"]


def test_correlation_groups_ignores_nan_correlation():
    # 'a' and 'b' never co-occur (structural NaN, like tier1_code_* vs
    # tier1_explanation_* in the real pivot) -- must not be treated as a
    # redundant pair just because pandas' pairwise corr returns NaN.
    wide = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, np.nan, np.nan, np.nan],
        "b": [np.nan, np.nan, np.nan, 1.0, 2.0, 3.0],
    })
    corr, groups = feature_importance.compute_correlation_groups(wide, ["a", "b"], threshold=0.9)
    assert groups == []


def test_rank_desc_highest_value_gets_rank_1():
    s = pd.Series({"a": 0.5, "b": 0.9, "c": 0.1})
    ranks = feature_importance._rank_desc(s)
    assert ranks["b"] == 1
    assert ranks["a"] == 2
    assert ranks["c"] == 3


def test_rank_desc_nan_ranked_last_not_dropped():
    s = pd.Series({"a": 0.5, "b": np.nan, "c": 0.9})
    ranks = feature_importance._rank_desc(s)
    assert ranks["c"] == 1
    assert ranks["a"] == 2
    assert ranks["b"] == 3  # present, ranked last -- not NaN itself
