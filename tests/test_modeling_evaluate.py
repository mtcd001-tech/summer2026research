"""Confirms evaluate_model() in modeling/mlp.py, svm.py, xgb.py all wire in
FAR/FRR (via modeling/metrics.compute_far_frr) alongside accuracy/F1, using a
stub "model" (just a .predict()) so this doesn't require fitting/training
anything real."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_MODELING_DIR = Path(__file__).resolve().parent.parent / "modeling"
if str(_MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELING_DIR))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _MODELING_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _StubModel:
    def __init__(self, y_pred):
        self._y_pred = y_pred

    def predict(self, X):
        return self._y_pred


# y_true has 4 bona fide (label 0) and 4 assisted (label 1); 1 bona fide
# misclassified (FRR=1/4) and 1 assisted misclassified as bona fide (FAR=1/4).
_Y_TRUE = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
_Y_PRED = pd.Series([0, 0, 0, 1, 0, 1, 1, 1])
_X_TEST = pd.DataFrame({"f1": range(8)})


@pytest.mark.parametrize("module_name", ["mlp", "xgb"])
def test_evaluate_model_includes_far_frr(module_name):
    module = _load(module_name)
    model = _StubModel(_Y_PRED)
    result = module.evaluate_model(model, _X_TEST, _Y_TRUE, labels=[0, 1])

    assert result["far"] == pytest.approx(0.25)
    assert result["frr"] == pytest.approx(0.25)
    assert "far_by_condition" not in result  # binary: only 2 labels


def test_svm_evaluate_model_includes_far_frr_and_labels():
    module = _load("svm")
    model = _StubModel(_Y_PRED)
    result = module.evaluate_model(model, _X_TEST, _Y_TRUE, labels=[0, 1])

    assert result["far"] == pytest.approx(0.25)
    assert result["frr"] == pytest.approx(0.25)
    assert result["labels"] == [0, 1]  # previously missing entirely from svm.py's output


@pytest.mark.parametrize("module_name", ["mlp", "svm", "xgb"])
def test_evaluate_model_skips_far_frr_without_labels(module_name):
    module = _load(module_name)
    model = _StubModel(_Y_PRED)
    result = module.evaluate_model(model, _X_TEST, _Y_TRUE, labels=None)

    assert "far" not in result
    assert "frr" not in result


def test_multiclass_far_by_condition_present():
    module = _load("xgb")
    y_true = pd.Series([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    y_pred = pd.Series([0, 1, 0, 1, 2, 2, 0, 3, 4, 4])  # s2 and s4 leak into bona fide
    model = _StubModel(y_pred)

    result = module.evaluate_model(model, pd.DataFrame({"f1": range(10)}), y_true, labels=[0, 1, 2, 3, 4])

    assert "far_by_condition" in result
    assert result["far_by_condition"][1] == pytest.approx(0.5)  # s2: 1/2 called bona fide
    assert result["far_by_condition"][3] == pytest.approx(0.5)  # s4: 1/2 called bona fide
    assert result["far_by_condition"][2] == pytest.approx(0.0)
    assert result["far_by_condition"][4] == pytest.approx(0.0)
