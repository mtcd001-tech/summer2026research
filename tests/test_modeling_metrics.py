"""Unit tests for modeling/metrics.py's compute_far_frr, against hand-computed
values -- FAR/FRR definitions follow the ICTAI/IJCB convention with bona
fide (label 0) as the positive class."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modeling"))
from metrics import compute_far_frr  # noqa: E402


def test_binary_far_frr():
    # rows/cols = [bona fide, assisted]
    #   bona fide row:  70 correct, 30 misclassified as assisted -> FRR = 30/100
    #   assisted row:   20 misclassified as bona fide, 180 correct -> FAR = 20/200
    cm = [[70, 30], [20, 180]]
    result = compute_far_frr(cm, [0, 1])
    assert result["frr"] == pytest.approx(0.3)
    assert result["far"] == pytest.approx(20 / 200)
    assert "far_by_condition" not in result


def test_binary_far_frr_label_order_independent():
    # same confusion structure, but labels list order is [1, 0] and the
    # matrix rows/cols follow that same order -- result must be identical.
    cm = [[180, 20], [30, 70]]  # row0=assisted, row1=bona fide
    result = compute_far_frr(cm, [1, 0])
    assert result["frr"] == pytest.approx(0.3)
    assert result["far"] == pytest.approx(20 / 200)


def test_multiclass_far_frr_and_per_condition():
    # labels 0..4 = s1 bona fide, s2, s3, s4, s5. Each row totals 100.
    cm = [
        [60, 10, 10, 10, 10],   # true bona fide: 40/100 misclassified -> FRR=0.4
        [5, 90, 0, 0, 5],       # s2: 5/100 called bona fide
        [30, 0, 60, 0, 10],     # s3: 30/100 called bona fide (the weak spot)
        [8, 0, 0, 90, 2],       # s4: 8/100 called bona fide
        [2, 0, 0, 0, 98],       # s5: 2/100 called bona fide
    ]
    result = compute_far_frr(cm, [0, 1, 2, 3, 4])

    assert result["frr"] == pytest.approx(0.4)
    # pooled FAR = (5+30+8+2) / 400 = 45/400
    assert result["far"] == pytest.approx(45 / 400)
    assert result["far_by_condition"] == {
        1: pytest.approx(0.05),
        2: pytest.approx(0.30),
        3: pytest.approx(0.08),
        4: pytest.approx(0.02),
    }


def test_zero_row_total_yields_nan_not_crash():
    cm = [[0, 0], [10, 90]]
    result = compute_far_frr(cm, [0, 1])
    assert math.isnan(result["frr"])
    assert result["far"] == pytest.approx(10 / 100)


def test_requires_bona_fide_label_present():
    with pytest.raises(ValueError):
        compute_far_frr([[1, 2], [3, 4]], [1, 2])


def test_requires_matching_shape():
    with pytest.raises(ValueError):
        compute_far_frr([[1, 2], [3, 4]], [0, 1, 2])
