"""FAR/FRR computed from a confusion matrix, matching the ICTAI/IJCB
reporting convention from the prior studies. Framework-agnostic (plain
list-of-lists/numpy in, plain dict of floats out) so the exact same function
can be used both live (mlp.py/svm.py/xgb.py's evaluate_model, at training
time) and to backfill older results.json files that predate these metrics,
from nothing but their already-saved confusion_matrix + label order --
no retraining required.

Reference: Vietnamese_keystrokes/src/pipeline_v2/step5_collect.py:33-41 has
the same confusion-matrix-collapsing idea, for a different label space
(bonafide/paraphrase/transcribe/fake-paraphrase/fake-transcribe as 5
distinct classes with a different positive-class convention) -- used here as
a guide, not copied.

Label space (this project): label 0 is always bona fide, the positive
class. Binary: {0: bona fide, 1: assisted}. Multiclass: {0: s1 bona fide,
1: s2, 2: s3, 3: s4, 4: s5}.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def compute_far_frr(cm: Sequence[Sequence[int]], labels: Sequence[int]) -> dict:
    """FAR/FRR from confusion matrix `cm` (rows/columns ordered per `labels`),
    treating label 0 (bona fide) as the positive class.

    FRR (False Rejection Rate) = proportion of bona-fide samples (true label
    0) predicted as anything else. Of all samples whose true label is 0, the
    fraction NOT predicted 0.

    FAR (False Acceptance Rate) = proportion of non-bona-fide samples
    predicted as bona fide. Of all samples whose true label is not 0, the
    fraction predicted 0. For multiclass this pools every non-bona-fide
    label together, so it's directly comparable to the binary task's FAR.

    For >2 labels, also returns `far_by_condition`: the same ratio computed
    separately for each non-bona-fide label (e.g. {1: 0.02, 2: 0.15, 3: 0.31,
    4: 0.04} for s2/s3/s4/s5) -- the pooled FAR hides which condition is
    actually leaking into bona fide, which matters more here than the pooled
    number.

    Returns NaN for a rate whose denominator (a row total) is zero.
    """
    cm = np.asarray(cm, dtype=float)
    labels = list(labels)
    if cm.shape[0] != len(labels) or cm.shape[1] != len(labels):
        raise ValueError(f"cm shape {cm.shape} doesn't match len(labels)={len(labels)}")
    if 0 not in labels:
        raise ValueError(f"compute_far_frr requires label 0 (bona fide) in labels={labels}")

    bf_idx = labels.index(0)
    row_sums = cm.sum(axis=1)

    bf_total = row_sums[bf_idx]
    frr = float((bf_total - cm[bf_idx, bf_idx]) / bf_total) if bf_total > 0 else float("nan")

    other_idx = [i for i in range(len(labels)) if i != bf_idx]
    other_total = float(sum(row_sums[i] for i in other_idx))
    far_numerator = float(sum(cm[i, bf_idx] for i in other_idx))
    far = far_numerator / other_total if other_total > 0 else float("nan")

    result = {"far": far, "frr": frr}

    if len(labels) > 2:
        far_by_condition = {}
        for i in other_idx:
            total_i = row_sums[i]
            far_by_condition[labels[i]] = float(cm[i, bf_idx] / total_i) if total_i > 0 else float("nan")
        result["far_by_condition"] = far_by_condition

    return result
