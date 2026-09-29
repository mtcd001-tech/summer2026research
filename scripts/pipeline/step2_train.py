"""Step 2: train ONE model on ONE task, across all 5 folds or a chosen subset.

Calls modeling/{model}.py's run_experiment() once per fold (subprocess, like
the retired modeling/run_all.py did), writing to
runs/{task}/{model}/fold{n}/results.json.

Re-runnable: a fold is skipped if its results.json already exists, unless
--force is passed. This means you can run one (model, task) pair, inspect
its results, then decide whether to run the next -- you never have to commit
to the whole 3-model x 2-task matrix at once. Re-running after a partial/
interrupted run only redoes the folds that didn't finish.

--fold restricts which of the 5 folds run, e.g. `--fold 1 3 5`. Omit it to run
all 5 (the default). The skip-if-results-exist/--force behavior is identical
per fold regardless of --fold -- it only changes which folds are considered.

--quick cuts the GA budget (population=4, generations=1, cv_splits=2, plus
max_iter=50 for mlp) so the whole 5-fold run finishes in well under a minute,
to sanity-check that data flows through load -> feature-select -> GA -> fit
-> eval -> save without errors. It is NOT meant to produce a meaningful
tuned model -- rerun without --quick (and with --force, since --quick's
results.json files would otherwise look "already done") once the wiring is
confirmed.

Usage:
    python scripts/pipeline/step2_train.py --model xgb --task binary
    python scripts/pipeline/step2_train.py --model svm --task multiclass --quick
    python scripts/pipeline/step2_train.py --model mlp --task binary --force
    python scripts/pipeline/step2_train.py --model xgb --task binary --fold 1 3 5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CM_LABELS, FOLDS, FULL_GA_ARGS, FULL_MLP_EXTRA_ARGS, MODELING_DIR,
    MODELS, QUICK_GA_ARGS, QUICK_MLP_EXTRA_ARGS, TASKS,
    dataset_paths, fold_is_done, fold_outdir,
)


def run_fold(model: str, task: str, fold: int, quick: bool, force: bool, seed: int = 42) -> bool:
    train_csv, test_csv = dataset_paths(task, fold)
    outdir = fold_outdir(task, model, fold)

    if not train_csv.exists() or not test_csv.exists():
        print(f"  [skip] fold {fold}: missing {train_csv.name}/{test_csv.name} "
              f"-- run step1_prepare.py --task {task} first")
        return False

    if not force and fold_is_done(task, model, fold):
        print(f"  [done] fold {fold} (results.json exists; use --force to redo)")
        return True

    script = MODELING_DIR / f"{model}.py"
    cmd = [
        sys.executable, str(script),
        "--train", str(train_csv),
        "--test", str(test_csv),
        "--outdir", str(outdir),
        "--seed", str(seed),
        "--cm-labels", *CM_LABELS[task],
    ]

    ga_args = QUICK_GA_ARGS if quick else FULL_GA_ARGS
    cmd += ga_args
    if model == "mlp":
        cmd += QUICK_MLP_EXTRA_ARGS if quick else FULL_MLP_EXTRA_ARGS

    print(f"  Running fold {fold}{' [quick]' if quick else ''}: {' '.join(cmd[1:])}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [error] fold {fold}: {e}")
        return False


def run_model_task(
    model: str,
    task: str,
    quick: bool,
    force: bool,
    seed: int = 42,
    folds: Optional[list[int]] = None,
) -> tuple[int, int]:
    folds = folds if folds is not None else FOLDS
    print(f"\n{'='*60}")
    print(f"  {model} / {task}{'  [quick]' if quick else ''}  folds={folds}")
    print(f"{'='*60}")

    done = 0
    for fold in folds:
        if run_fold(model, task, fold, quick=quick, force=force, seed=seed):
            done += 1
    print(f"\n  {model}/{task}: {done}/{len(folds)} folds complete")
    return done, len(folds)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument(
        "--fold", type=int, nargs="+", choices=FOLDS, default=None, metavar="N",
        help="One or more fold numbers (1-5) to run. Default: all 5.",
    )
    p.add_argument("--quick", action="store_true", help="Tiny GA budget, wiring sanity-check only.")
    p.add_argument("--force", action="store_true", help="Redo folds even if results.json already exists.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    done, total = run_model_task(
        args.model, args.task, quick=args.quick, force=args.force, seed=args.seed, folds=args.fold
    )
    return 0 if done == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
