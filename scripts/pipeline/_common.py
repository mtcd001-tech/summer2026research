"""Shared constants/helpers for scripts/pipeline/step*.py.

Kept tiny and dependency-free (no pandas/sklearn) so importing it doesn't
drag in the modeling stack for scripts that don't need it (step3_collect.py
in particular only needs paths + json).
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
RUNS_DIR = REPO_ROOT / "runs"
MODELING_DIR = REPO_ROOT / "modeling"

TASKS = ["binary", "multiclass"]
MODELS = ["mlp", "svm", "xgb"]
FOLDS = [1, 2, 3, 4, 5]

CM_LABELS = {
    "binary": ["0", "1"],
    "multiclass": ["0", "1", "2", "3", "4"],
}

LABEL_NAMES = {
    "binary": {0: "bona fide", 1: "assisted"},
    "multiclass": {0: "s1 bona fide", 1: "s2", 2: "s3", 3: "s4", 4: "s5"},
}

# --quick mode: small enough to finish a full 5-fold step2_train.py run for
# any of the three models in well under a minute, purely to sanity-check
# that data flows through load -> feature-select -> GA -> fit -> eval -> save
# without errors. Not meant to produce a meaningful tuned model.
QUICK_GA_ARGS = ["--population", "4", "--generations", "1", "--cv-splits", "2"]
QUICK_MLP_EXTRA_ARGS = ["--max-iter", "50"]

# Full run GA budget (see modeling/run_all.py's retired docstring for the
# calibration this was based on: ~0.2pp from the Vietnamese_keystrokes
# original defaults' result, in about a third of the time).
FULL_GA_ARGS = ["--population", "50", "--generations", "10", "--cv-splits", "5"]
FULL_MLP_EXTRA_ARGS = ["--max-iter", "300"]


def dataset_paths(task: str, fold: int) -> tuple[Path, Path]:
    train_csv = DATASETS_DIR / task / f"train_fold{fold}.csv"
    test_csv = DATASETS_DIR / task / f"test_fold{fold}.csv"
    return train_csv, test_csv


def fold_outdir(task: str, model: str, fold: int) -> Path:
    return RUNS_DIR / task / model / f"fold{fold}"


def fold_results_path(task: str, model: str, fold: int) -> Path:
    return fold_outdir(task, model, fold) / "results.json"


def fold_is_done(task: str, model: str, fold: int) -> bool:
    return fold_results_path(task, model, fold).exists()
