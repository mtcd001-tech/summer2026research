"""Pivot features_clean.csv (long format) into per-fold wide train/test CSVs
ready for modeling/mlp.py, modeling/svm.py, modeling/xgb.py (ported from the
Vietnamese_keystrokes project's src/MLP.py / SVM.py / XGB.py).

Split is StratifiedGroupKFold(n_splits=5) grouped on user_id -- whole users
are assigned to either train or test (never split across both), following the
pattern in Vietnamese_keystrokes/src/pipeline_v2/step3_user_indep.py:44-49,
with stratification added since every user has a fixed, skewed label
distribution (1 bona fide session's worth of rows vs. 4 assisted sessions'
worth).

NOTE on structural NaN: the pivot index is (user, session, q_id, r_t) per
the task spec, so a "code" row has no tier1_explanation_*/tier2_explanation_*
values (and an "explanation" row has no tier1_code_*/tier2_code_*/tier3_llm_*
values) -- roughly half of each row's ~53 feature columns are NaN not because
they're "hard to compute" but because they don't apply to that r_t at all.
This is on top of the smaller genuine-missing-data NaN rate (e.g.
cursor_jump_rate needing >=2 cursor events). Both kinds get the same
downstream treatment (median imputation / passthrough to XGBoost) -- see the
report for why this is worth knowing about before trusting per-feature
importances.

Re-runnable: if datasets/{task}/train_fold{n}.csv and test_fold{n}.csv already
exist for every fold, this exits without re-pivoting/re-splitting unless
--force is passed.

Usage:
    python scripts/pipeline/step1_prepare.py --task binary
    python scripts/pipeline/step1_prepare.py --task multiclass --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATASETS_DIR  # noqa: E402

GROUP_INDEX = ["user", "session", "q_id", "r_t"]

# No separation (or worse) vs. their unstripped originals -- see the
# tier2.py feature-review that added the *_no_boilerplate variants.
DROPPED_FEATURES = {
    "tier2_code_token_entropy",
    "tier2_code_identifier_diversity_no_boilerplate",
    "tier2_code_identifier_reuse_hapax_no_boilerplate",
}

LABEL_FUNCS = {
    "binary": lambda session: 0 if session == 1 else 1,
    "multiclass": lambda session: session - 1,
}


def _resolve_version_collisions(long_df: pd.DataFrame) -> pd.DataFrame:
    """The pivot index (user, session, q_id, r_t) omits `version`. This is
    safe by construction now: src/extract.py's _is_human_typed rule keeps
    exactly one version per (session, r_t) -- version 1 only for session 1
    (bona fide), version 2 only for sessions 2-5 (version 1 there is always
    the ChatGPT paste box, confirmed by Ashley who built the collection
    platform, and is dropped unconditionally, not by a count heuristic). So
    a real collision should never reach this function. Kept as a
    defense-in-depth safety net anyway: if one ever showed up (e.g. from an
    older features_clean.csv extracted before this rule existed), silently
    pivoting it with aggfunc="first" would pick an arbitrary version's
    feature values with no record of it happening. Instead: when both
    versions are present, keep version 2 and drop version 1's row, logging
    exactly which groups this affected.
    """
    versions_per_group = long_df.groupby(GROUP_INDEX)["version"].nunique()
    colliding = versions_per_group[versions_per_group > 1].index
    if len(colliding) == 0:
        return long_df

    colliding_mask = long_df.set_index(GROUP_INDEX).index.isin(colliding)
    print(f"[warning] {len(colliding)} group(s) unexpectedly had BOTH versions present "
          f"-- keeping version 2, dropping version 1's row for each:")
    for key in colliding:
        print(f"    user={key[0]} session={key[1]} q_id={key[2]} r_t={key[3]}")

    drop_mask = colliding_mask & (long_df["version"].to_numpy() == 1)
    return long_df.loc[~drop_mask].copy()


def load_wide(features_csv: Path) -> pd.DataFrame:
    """Long (user, session, q_id, r_t, version, tier, feature, value) -> wide
    (one row per (user, session, q_id, r_t), one column per feature)."""
    long_df = pd.read_csv(features_csv)
    long_df = _resolve_version_collisions(long_df)

    wide = long_df.pivot_table(
        index=GROUP_INDEX, columns="feature", values="value", aggfunc="first"
    )
    wide = wide.drop(columns=[c for c in DROPPED_FEATURES if c in wide.columns])
    return wide.reset_index()


def add_label(wide: pd.DataFrame, task: str) -> pd.DataFrame:
    if task not in LABEL_FUNCS:
        raise ValueError(f"unknown --task {task!r}; choose from {sorted(LABEL_FUNCS)}")
    out = wide.copy()
    out["label"] = out["session"].map(LABEL_FUNCS[task])
    return out


def write_folds(wide: pd.DataFrame, outdir: Path, n_splits: int, seed: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in wide.columns if c not in GROUP_INDEX + ["label"]]
    # user_id is kept in the written CSVs (renamed from "user") specifically
    # so the model scripts can group their inner GA CV by participant too --
    # load_xy already drops user_id/session/section as metadata before it
    # ever reaches a model, so its presence doesn't change what the
    # classifier sees.
    groups = wide["user"]
    y = wide["label"]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(wide, y, groups), start=1):
        train_df = wide.iloc[train_idx]
        test_df = wide.iloc[test_idx]

        train_users = set(train_df["user"])
        test_users = set(test_df["user"])
        assert train_users.isdisjoint(test_users), (
            f"fold {fold}: user(s) leaked across train/test: "
            f"{train_users & test_users}"
        )

        train_out = train_df[["label"] + feature_cols].assign(user_id=train_df["user"].values)
        test_out = test_df[["label"] + feature_cols].assign(user_id=test_df["user"].values)

        train_csv = outdir / f"train_fold{fold}.csv"
        test_csv = outdir / f"test_fold{fold}.csv"
        train_out.to_csv(train_csv, index=False)
        test_out.to_csv(test_csv, index=False)

        print(
            f"fold {fold}: train {len(train_out):4d} rows ({len(train_users):2d} users, "
            f"labels={dict(sorted(train_out['label'].value_counts().to_dict().items()))}) | "
            f"test {len(test_out):4d} rows ({len(test_users):2d} users, "
            f"labels={dict(sorted(test_out['label'].value_counts().to_dict().items()))})"
        )


def already_done(outdir: Path, n_splits: int) -> bool:
    if not outdir.exists():
        return False
    return all(
        (outdir / f"train_fold{fold}.csv").exists() and (outdir / f"test_fold{fold}.csv").exists()
        for fold in range(1, n_splits + 1)
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=Path("features_clean.csv"), help="Long-format input CSV.")
    p.add_argument("--task", choices=sorted(LABEL_FUNCS), required=True, help="Label scheme.")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Directory for train_fold{n}.csv/test_fold{n}.csv. Default: datasets/{task}")
    p.add_argument("--n-splits", type=int, default=5, help="StratifiedGroupKFold splits.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--force", action="store_true", help="Re-pivot/re-split even if output CSVs already exist.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    outdir = args.outdir if args.outdir is not None else DATASETS_DIR / args.task

    if not args.force and already_done(outdir, args.n_splits):
        print(f"[skip] {outdir} already has all {args.n_splits} fold CSVs (use --force to regenerate)")
        return 0

    if not args.features.exists():
        raise FileNotFoundError(f"features CSV not found: {args.features}")

    wide = load_wide(args.features)
    wide = add_label(wide, args.task)

    print(f"pivoted: {len(wide)} groups x {len(wide.columns) - len(GROUP_INDEX) - 1} features")
    print(f"users: {wide['user'].nunique()}")
    print(f"label distribution: {dict(sorted(wide['label'].value_counts().to_dict().items()))}")
    print()

    write_folds(wide, outdir, args.n_splits, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
