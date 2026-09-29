"""Step 3: read all runs/{task}/{model}/fold{n}/results.json and print a
summary table -- mean +/- std accuracy, macro-F1, and FAR/FRR per (task,
model) across whatever folds are currently done, plus the summed confusion
matrix and per-condition FAR breakdown (s2/s3/s4/s5 individually) for
multiclass.

FAR/FRR follow the ICTAI/IJCB reporting convention from the prior studies,
with bona fide (label 0) as the positive class -- see modeling/metrics.py
for the exact definitions and modeling/mlp.py's module docstring for how
they're computed at training time.

Backfilling: a results.json written before FAR/FRR existed (or, for
modeling/svm.py, before it saved `labels` at all) is recomputed in place
from its already-stored confusion_matix -- no retraining needed. svm.py
results without a saved `labels` fall back to the (session, version)-order
_common.CM_LABELS[task] convention, since that's the only order
step2_train.py has ever passed. Backfilled files are rewritten to disk and
listed at the end of the run.

Safe to run at any time, including with a partially-complete runs/ directory
(e.g. after running step2_train.py for only one model/task so far, or with
--quick results mixed in) -- it just reports on whatever results.json files
exist right now and says how many folds each (task, model) has.

Usage:
    python scripts/pipeline/step3_collect.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CM_LABELS, FOLDS, LABEL_NAMES, MODELS, RUNS_DIR, TASKS, fold_results_path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "modeling"))
from metrics import compute_far_frr  # noqa: E402


def _backfill_far_frr(data: dict, task: str) -> bool:
    """Mutate `data` in place to add far/frr(/far_by_condition) from its
    already-stored confusion_matrix. Returns True if `data` was changed."""
    if "far" in data:
        return False
    if "confusion_matrix" not in data:
        return False

    labels = data.get("labels")
    if labels is None:
        # Older modeling/svm.py results didn't save `labels` at all.
        # step2_train.py has only ever passed --cm-labels from
        # _common.CM_LABELS[task], so that's the row/column order the
        # stored confusion_matrix must be in.
        labels = [int(x) for x in CM_LABELS[task]]
    else:
        labels = [int(x) for x in labels]

    if 0 not in labels:
        return False

    data.update(compute_far_frr(data["confusion_matrix"], labels))
    data.setdefault("labels", labels)
    return True


def load_fold_results(task: str, model: str, backfilled: list[str]) -> list[dict]:
    out = []
    for fold in FOLDS:
        path = fold_results_path(task, model, fold)
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if _backfill_far_frr(data, task):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            backfilled.append(f"{task}/{model}/fold{fold}")

        out.append(data)
    return out


def summarize(task: str, model: str, backfilled: list[str]) -> dict | None:
    fold_results = load_fold_results(task, model, backfilled)
    if not fold_results:
        return None

    accs = np.array([r["test_accuracy"] for r in fold_results], dtype=float)
    macro_f1s = np.array([r["test_macro_f1"] for r in fold_results], dtype=float)
    weighted_f1s = np.array([r["test_weighted_f1"] for r in fold_results], dtype=float)
    fars = np.array([r.get("far", float("nan")) for r in fold_results], dtype=float)
    frrs = np.array([r.get("frr", float("nan")) for r in fold_results], dtype=float)
    cms = [np.array(r["confusion_matrix"]) for r in fold_results]

    summary = {
        "n_folds": len(fold_results),
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std()),
        "macro_f1_mean": float(macro_f1s.mean()),
        "macro_f1_std": float(macro_f1s.std()),
        "weighted_f1_mean": float(weighted_f1s.mean()),
        "weighted_f1_std": float(weighted_f1s.std()),
        "far_mean": float(np.nanmean(fars)),
        "far_std": float(np.nanstd(fars)),
        "frr_mean": float(np.nanmean(frrs)),
        "frr_std": float(np.nanstd(frrs)),
        "summed_confusion_matrix": np.sum(cms, axis=0).tolist(),
        "per_fold_accuracy": accs.tolist(),
        "per_fold_macro_f1": macro_f1s.tolist(),
        "per_fold_far": fars.tolist(),
        "per_fold_frr": frrs.tolist(),
    }

    condition_keys = sorted(
        {k for r in fold_results for k in r.get("far_by_condition", {})},
        key=lambda x: int(x),
    )
    if condition_keys:
        far_by_condition = {}
        for cond in condition_keys:
            vals = np.array(
                [r.get("far_by_condition", {}).get(cond, float("nan")) for r in fold_results], dtype=float
            )
            far_by_condition[cond] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
        summary["far_by_condition"] = far_by_condition

    return summary


def print_cm(cm: np.ndarray, labels: dict[int, str]) -> None:
    names = [labels.get(i, str(i)) for i in range(cm.shape[0])]
    col_w = max(10, max(len(n) for n in names) + 2)
    print("  " + "".join(f"{n:>{col_w}}" for n in [""] + names) + "   (rows=true, cols=pred)")
    for i, row in enumerate(cm):
        print("  " + f"{names[i]:>{col_w}}" + "".join(f"{int(v):>{col_w}}" for v in row))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    parse_args(argv)  # no options besides -h/--help; errors on unrecognized args
    all_summaries: dict[str, dict] = {}
    backfilled: list[str] = []
    any_partial = False

    for task in TASKS:
        print("\n" + "=" * 78)
        print(f"TASK: {task}")
        print("=" * 78)
        for model in MODELS:
            summary = summarize(task, model, backfilled)
            if summary is None:
                print(f"\n{model}: no results yet (run step2_train.py --model {model} --task {task})")
                continue
            all_summaries[f"{task}/{model}"] = summary

            partial_note = ""
            if summary["n_folds"] < len(FOLDS):
                partial_note = f"  [PARTIAL: {summary['n_folds']}/{len(FOLDS)} folds]"
                any_partial = True

            print(f"\n{model}  (n_folds={summary['n_folds']}){partial_note}")
            print(f"  accuracy : {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}"
                  f"   (per-fold: {[round(a, 3) for a in summary['per_fold_accuracy']]})")
            print(f"  macro F1 : {summary['macro_f1_mean']:.4f} +/- {summary['macro_f1_std']:.4f}"
                  f"   (per-fold: {[round(a, 3) for a in summary['per_fold_macro_f1']]})")
            print(f"  weighted F1: {summary['weighted_f1_mean']:.4f} +/- {summary['weighted_f1_std']:.4f}")
            print(f"  FAR      : {summary['far_mean']:.4f} +/- {summary['far_std']:.4f}"
                  f"   (per-fold: {[round(a, 3) for a in summary['per_fold_far']]})")
            print(f"  FRR      : {summary['frr_mean']:.4f} +/- {summary['frr_std']:.4f}"
                  f"   (per-fold: {[round(a, 3) for a in summary['per_fold_frr']]})")

            if "far_by_condition" in summary:
                print("  FAR by condition (fraction of that condition's samples called bona fide):")
                names = LABEL_NAMES.get(task, {})
                for cond, stats in summary["far_by_condition"].items():
                    label_name = names.get(int(cond), cond)
                    print(f"    {label_name:>12s}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

            if task == "multiclass":
                print(f"  summed confusion matrix ({summary['n_folds']} folds):")
                print_cm(np.array(summary["summed_confusion_matrix"]), LABEL_NAMES[task])

    if backfilled:
        print(f"\nBackfilled FAR/FRR into {len(backfilled)} existing results.json (recomputed from their "
              f"stored confusion matrices, no retraining needed):")
        for entry in backfilled:
            print(f"  - {entry}")

    if any_partial:
        print("\nNote: some (task, model) results above are from a partial set of folds "
              "-- rerun step2_train.py for those to complete them before trusting the mean/std.")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved aggregate summary to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
