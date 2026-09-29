"""Run all 6 (model, task) combinations in sequence, then collect results.

This is a convenience wrapper, not a separate mechanism: it just calls
step2_train.py's run_model_task() once per combination (each independently
re-runnable and skip-aware, per step2_train.py's docstring) and then
step3_collect.py's main() at the end. Interrupting this and re-running it
later resumes correctly -- already-complete folds are skipped, exactly as if
you'd called step2_train.py for each combination yourself.

Usage:
    python scripts/pipeline/run_all.py                 # full run, all 6 combos
    python scripts/pipeline/run_all.py --quick          # tiny GA budget, wiring check
    python scripts/pipeline/run_all.py --force          # redo everything
    python scripts/pipeline/run_all.py --model xgb      # restrict to one model
    python scripts/pipeline/run_all.py --task binary    # restrict to one task
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODELS, TASKS  # noqa: E402
import step2_train  # noqa: E402
import step3_collect  # noqa: E402


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, default=None, help="Restrict to one model.")
    p.add_argument("--task", choices=TASKS, default=None, help="Restrict to one task.")
    p.add_argument("--quick", action="store_true", help="Tiny GA budget, wiring sanity-check only.")
    p.add_argument("--force", action="store_true", help="Redo folds even if results.json already exists.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    models = [args.model] if args.model else MODELS
    tasks = [args.task] if args.task else TASKS

    total_done, total_folds = 0, 0
    for task in tasks:
        for model in models:
            done, total = step2_train.run_model_task(
                model, task, quick=args.quick, force=args.force, seed=args.seed
            )
            total_done += done
            total_folds += total

    print(f"\nAll combinations: {total_done}/{total_folds} folds complete.")

    # Explicit empty argv: step3_collect.main() defaults to parsing sys.argv,
    # which at this point still holds run_all.py's own CLI args (e.g.
    # --quick/--model) -- without this, step3_collect's parser (which takes
    # no options besides -h) would reject them as unrecognized.
    step3_collect.main([])

    return 0 if total_done == total_folds else 1


if __name__ == "__main__":
    raise SystemExit(main())
