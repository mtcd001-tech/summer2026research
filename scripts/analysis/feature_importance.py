"""Three complementary feature-importance rankings, plus a redundancy
analysis, to address a real gap in ranking features by Cohen's d alone:
Cohen's d is univariate and model-free, so it can't detect redundancy or
interactions -- e.g. kit_pause_time_fraction and kit_long_pause_fraction
both rank high while measuring nearly the same thing, and Cohen's d has no
way to say the second adds nothing once you have the first.

Does NOT train any models. Reuses the SVM multiclass runs that already
exist in runs/multiclass/svm/fold{1..5}/ (model.joblib + selected_features.txt
+ the fold's test CSV) for permutation importance and selection stability.

Four measures, all computed on the full 1,680-group wide feature table
(scripts/pipeline/step1_prepare.py's pivot) unless noted:

  1. COHEN'S D -- recomputed here (not reused from anywhere) so it lives in
     one script. Bona fide (session 1) vs. each of sessions 2-5, pooled SD:
     d = (mean_condition - mean_bonafide) / sqrt((var_condition + var_bonafide) / 2).
     Sign is kept (direction matters) -- this is why Cohen's d is used here
     rather than d-prime.

  2. MUTUAL INFORMATION -- sklearn.feature_selection.mutual_info_classif, for
     both binary and multiclass labels. Catches non-monotonic dependence
     Cohen's d misses. Reuses modeling/svm.py's select_features_mutual_info
     (same median-imputation-for-scoring-only behavior the trained models
     actually used) when importable; falls back to calling
     mutual_info_classif directly otherwise.

  3. PERMUTATION IMPORTANCE -- sklearn.inspection.permutation_importance on
     each existing trained model (--model, default svm), on that fold's TEST
     set only, using ONLY the features that fold's model was actually fit on
     (selected_features.txt) -- passing an unselected feature would mismatch
     the fitted pipeline's expected input shape. This is the only measure
     tied to what the model actually uses on genuinely unseen users (the
     user-independent test split), rather than a property of the raw data.
     Test features are passed raw/unscaled -- the saved model is a Pipeline
     with its own imputer+scaler, exactly like training.

  4. REDUNDANCY -- pairwise Pearson correlation across all features; any
     group with |r| > 0.9 is reported with each member's rank under all
     three measures above, so it's visible whether permutation importance
     picks one member and discounts the rest while Cohen's d/MI rank all of
     them high (the actual gap this script exists to close).

  5. SELECTION STABILITY -- how many of the 5 folds' selected_features.txt
     (MI-based, --feature-percentage 50) include each feature.

Reference for the confusion-matrix-adjacent FAR/FRR framing elsewhere in this
project is unrelated to this script; nothing here is copied from it.

Usage:
    python scripts/analysis/feature_importance.py
    python scripts/analysis/feature_importance.py --model svm
    python scripts/analysis/feature_importance.py --n-repeats 20 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PIPELINE_DIR = REPO_ROOT / "scripts" / "pipeline"
MODELING_DIR = REPO_ROOT / "modeling"
ANALYSIS_OUT_DIR = REPO_ROOT / "analysis"

sys.path.insert(0, str(PIPELINE_DIR))
from _common import DATASETS_DIR, FOLDS, RUNS_DIR  # noqa: E402
from step1_prepare import GROUP_INDEX, add_label, load_wide  # noqa: E402

sys.path.insert(0, str(MODELING_DIR))
try:
    from svm import select_features_mutual_info as _modeling_mi  # noqa: E402
    _HAVE_MODELING_MI = True
except ImportError:
    _HAVE_MODELING_MI = False
    from sklearn.feature_selection import mutual_info_classif

DEFAULT_FEATURES_CSV = REPO_ROOT / "features_clean.csv"
BONAFIDE_SESSION = 1
ASSISTED_SESSIONS = [2, 3, 4, 5]
CONDITION_NAMES = {2: "s2", 3: "s3", 4: "s4", 5: "s5"}
REDUNDANCY_THRESHOLD = 0.9
TOP_DISAGREEMENTS_N = 10


# ---------------------------------------------------------------------------
# 1. Cohen's d
# ---------------------------------------------------------------------------

def compute_cohens_d(wide: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Per-feature Cohen's d, bona fide vs. each of sessions 2-5, pooled SD,
    sign preserved. Returns a DataFrame indexed by feature with one column
    per condition (s2..s5) plus cohens_d_max (max absolute value)."""
    bf = wide.loc[wide["session"] == BONAFIDE_SESSION, feature_cols]
    mean_bf = bf.mean()
    var_bf = bf.var()  # ddof=1 (sample variance), the conventional choice for Cohen's d

    per_condition = {}
    for session in ASSISTED_SESSIONS:
        cond = wide.loc[wide["session"] == session, feature_cols]
        mean_c = cond.mean()
        var_c = cond.var()
        pooled_sd = np.sqrt((var_c + var_bf) / 2)
        d = (mean_c - mean_bf) / pooled_sd
        d = d.where(pooled_sd > 0, other=np.nan)
        per_condition[CONDITION_NAMES[session]] = d

    result = pd.DataFrame(per_condition)
    result["cohens_d_max"] = result.abs().max(axis=1)
    return result


# ---------------------------------------------------------------------------
# 2. Mutual information
# ---------------------------------------------------------------------------

def compute_mutual_info(X: pd.DataFrame, y: pd.Series, random_state: int) -> pd.Series:
    """MI score per feature. Reuses modeling/svm.py's
    select_features_mutual_info (median-imputes NaN for scoring only, same
    as at training time) when importable; otherwise median-imputes locally
    and calls mutual_info_classif directly."""
    if _HAVE_MODELING_MI:
        _selected, mi_df = _modeling_mi(X, y, feature_percentage=100.0, random_state=random_state)
        return mi_df.set_index("feature")["mi_score"]

    X_filled = X.fillna(X.median(numeric_only=True))
    mi = mutual_info_classif(X_filled, y, discrete_features=False, random_state=random_state)
    return pd.Series(mi, index=X.columns)


# ---------------------------------------------------------------------------
# 3. Permutation importance (existing trained models only -- no training)
# ---------------------------------------------------------------------------

def compute_permutation_importance(
    model: str, task: str, n_repeats: int, random_state: int, scoring: str = "f1_macro"
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Mean/std permutation importance per feature, averaged across
    whichever folds actually selected that feature (a feature never
    selected by any fold has no permutation importance -- NaN, not zero).
    Also returns the count of folds that contributed to each feature's
    mean/std, for transparency.
    """
    per_fold: dict[int, pd.Series] = {}

    for fold in FOLDS:
        outdir = RUNS_DIR / task / model / f"fold{fold}"
        model_path = outdir / "model.joblib"
        selected_path = outdir / "selected_features.txt"
        test_csv = DATASETS_DIR / task / f"test_fold{fold}.csv"

        if not (model_path.exists() and selected_path.exists() and test_csv.exists()):
            print(f"  [skip] fold {fold}: missing model.joblib/selected_features.txt/test CSV in {outdir}")
            continue

        pipeline = joblib.load(model_path)
        selected = [line.strip() for line in selected_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        test_df = pd.read_csv(test_csv)

        X_test = test_df[selected]  # raw, unscaled -- the pipeline has its own imputer+scaler
        y_test = test_df["label"]

        result = permutation_importance(
            pipeline, X_test, y_test,
            n_repeats=n_repeats, scoring=scoring, random_state=random_state,
        )
        per_fold[fold] = pd.Series(result.importances_mean, index=selected)
        print(f"  fold {fold}: permutation importance computed for {len(selected)} features")

    if not per_fold:
        raise RuntimeError(
            f"No usable runs found under {RUNS_DIR / task / model}/fold*/ "
            f"(need model.joblib + selected_features.txt + a matching test CSV in "
            f"{DATASETS_DIR / task}/). Nothing to compute permutation importance from."
        )

    per_fold_df = pd.DataFrame(per_fold)  # index=feature (union across folds), columns=fold number
    mean = per_fold_df.mean(axis=1, skipna=True)
    std = per_fold_df.std(axis=1, skipna=True)
    n_folds_present = per_fold_df.notna().sum(axis=1)
    return mean, std, n_folds_present


# ---------------------------------------------------------------------------
# 4. Redundancy: pairwise correlation + connected components at |r| > threshold
# ---------------------------------------------------------------------------

def compute_correlation_groups(
    wide: pd.DataFrame, feature_cols: list[str], threshold: float
) -> tuple[pd.DataFrame, list[list[str]]]:
    corr = wide[feature_cols].corr(method="pearson")  # pairwise-complete, NaN-tolerant

    parent = {f: f for f in feature_cols}

    def find(f: str) -> str:
        while parent[f] != f:
            f = parent[f]
        return f

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, f1 in enumerate(feature_cols):
        for f2 in feature_cols[i + 1:]:
            r = corr.loc[f1, f2]
            if pd.notna(r) and abs(r) >= threshold:
                union(f1, f2)

    groups: dict[str, list[str]] = defaultdict(list)
    for f in feature_cols:
        groups[find(f)].append(f)
    redundant_groups = [sorted(g) for g in groups.values() if len(g) > 1]
    redundant_groups.sort(key=len, reverse=True)
    return corr, redundant_groups


# ---------------------------------------------------------------------------
# 5. Selection stability
# ---------------------------------------------------------------------------

def compute_selection_frequency(model: str, task: str) -> pd.Series:
    counts: dict[str, int] = defaultdict(int)
    for fold in FOLDS:
        path = RUNS_DIR / task / model / f"fold{fold}" / "selected_features.txt"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            feat = line.strip()
            if feat:
                counts[feat] += 1
    return pd.Series(counts, name="selection_frequency")


# ---------------------------------------------------------------------------
# Assembly / reporting
# ---------------------------------------------------------------------------

def _rank_desc(series: pd.Series) -> pd.Series:
    """Rank 1 = highest value. NaNs ranked last (not dropped)."""
    return series.rank(ascending=False, method="min", na_option="bottom")


def build_report(
    features_csv: Path, model: str, task: str, n_repeats: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[list[str]]]:
    print(f"Loading and pivoting {features_csv} ...")
    wide = load_wide(features_csv)
    feature_cols = [c for c in wide.columns if c not in GROUP_INDEX]
    print(f"  {len(wide)} groups x {len(feature_cols)} features")

    print("\n[1/5] Cohen's d (bona fide vs. each assisted condition)...")
    cohens_d = compute_cohens_d(wide, feature_cols)

    print("\n[2/5] Mutual information (binary + multiclass labels)...")
    X = wide[feature_cols]
    y_binary = add_label(wide, "binary")["label"]
    y_multiclass = add_label(wide, "multiclass")["label"]
    mi_binary = compute_mutual_info(X, y_binary, random_state=seed)
    mi_multiclass = compute_mutual_info(X, y_multiclass, random_state=seed)
    print(f"  (using {'modeling/svm.py' if _HAVE_MODELING_MI else 'sklearn'} for MI scoring)")

    print(f"\n[3/5] Permutation importance from existing {model}/{task} runs (no training)...")
    perm_mean, perm_std, perm_n_folds = compute_permutation_importance(
        model=model, task=task, n_repeats=n_repeats, random_state=seed,
    )

    print(f"\n[4/5] Redundancy: pairwise correlation, |r| >= {REDUNDANCY_THRESHOLD}...")
    corr, redundant_groups = compute_correlation_groups(wide, feature_cols, REDUNDANCY_THRESHOLD)

    print(f"\n[5/5] Selection stability across {len(FOLDS)} folds ({model}/{task})...")
    selection_freq = compute_selection_frequency(model=model, task=task)

    # ---- Join everything ----
    report = pd.DataFrame(index=feature_cols)
    report.index.name = "feature"
    report["cohens_d_max"] = cohens_d["cohens_d_max"]
    report["cohens_d_per_condition"] = cohens_d[[CONDITION_NAMES[s] for s in ASSISTED_SESSIONS]].apply(
        lambda row: json.dumps({k: (None if pd.isna(v) else round(float(v), 4)) for k, v in row.items()}), axis=1
    )
    report["mutual_info_binary"] = mi_binary
    report["mutual_info_multiclass"] = mi_multiclass
    report["permutation_importance_mean"] = perm_mean
    report["permutation_importance_std"] = perm_std
    report["permutation_importance_n_folds"] = perm_n_folds.reindex(report.index).fillna(0).astype(int)
    report["selection_frequency"] = selection_freq.reindex(report.index).fillna(0).astype(int)

    report["cohens_d_rank"] = _rank_desc(report["cohens_d_max"])
    report["mutual_info_rank"] = _rank_desc(report["mutual_info_multiclass"])
    report["permutation_importance_rank"] = _rank_desc(report["permutation_importance_mean"])

    report = report.sort_values("permutation_importance_mean", ascending=False, na_position="last")
    return report.reset_index(), corr, redundant_groups


def print_spearman(report: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("SPEARMAN RANK CORRELATION BETWEEN THE THREE MEASURES")
    print("=" * 78)
    pairs = [
        ("cohens_d_max", "mutual_info_multiclass"),
        ("cohens_d_max", "permutation_importance_mean"),
        ("mutual_info_multiclass", "permutation_importance_mean"),
    ]
    for a, b in pairs:
        sub = report[[a, b]].dropna()
        rho, p = spearmanr(sub[a], sub[b])
        print(f"  {a:32s} vs {b:32s}: rho={rho:+.3f}  (n={len(sub)}, p={p:.3g})")


def print_redundant_groups(report: pd.DataFrame, redundant_groups: list[list[str]]) -> None:
    print("\n" + "=" * 78)
    print(f"REDUNDANCY: feature groups with pairwise |r| >= {REDUNDANCY_THRESHOLD}")
    print("=" * 78)
    if not redundant_groups:
        print("  none found")
        return

    ranked = report.set_index("feature")
    for group in redundant_groups:
        print(f"\n  group ({len(group)} features):")
        rows = ranked.loc[group, ["cohens_d_rank", "mutual_info_rank", "permutation_importance_rank"]]
        rows = rows.sort_values("permutation_importance_rank")
        for feat, row in rows.iterrows():
            print(
                f"    {feat:55s} cohens_d_rank={int(row['cohens_d_rank']):3d}  "
                f"mutual_info_rank={int(row['mutual_info_rank']):3d}  "
                f"permutation_rank={row['permutation_importance_rank']:.0f}"
            )


def print_disagreements(report: pd.DataFrame, top_n: int) -> None:
    print("\n" + "=" * 78)
    print("BIGGEST DISAGREEMENTS: Cohen's d rank vs. permutation importance rank")
    print("=" * 78)

    scored = report.dropna(subset=["cohens_d_rank", "permutation_importance_rank"]).copy()
    scored["rank_diff"] = scored["permutation_importance_rank"] - scored["cohens_d_rank"]

    print(f"\n  Ranked high by Cohen's d, low by permutation importance (likely redundant"
          f" -- {top_n}):")
    high_d_low_perm = scored.sort_values("rank_diff", ascending=False).head(top_n)
    for _, row in high_d_low_perm.iterrows():
        print(f"    {row['feature']:55s} cohens_d_rank={int(row['cohens_d_rank']):3d}"
              f"  permutation_rank={int(row['permutation_importance_rank']):3d}"
              f"  (diff={int(row['rank_diff']):+d})")

    print(f"\n  Ranked high by permutation importance, low by Cohen's d (likely only"
          f" useful in combination -- {top_n}):")
    high_perm_low_d = scored.sort_values("rank_diff", ascending=True).head(top_n)
    for _, row in high_perm_low_d.iterrows():
        print(f"    {row['feature']:55s} cohens_d_rank={int(row['cohens_d_rank']):3d}"
              f"  permutation_rank={int(row['permutation_importance_rank']):3d}"
              f"  (diff={int(row['rank_diff']):+d})")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES_CSV, help="Long-format features CSV.")
    p.add_argument(
        "--model", choices=["svm", "xgb", "mlp"], default="svm",
        help="Which trained model's runs/ to use for permutation importance + selection stability. "
             "Only svm has completed runs right now; xgb/mlp can be added once trained.",
    )
    p.add_argument("--task", default="multiclass", help="Task subdir under runs/ (only multiclass has trained models right now).")
    p.add_argument("--n-repeats", type=int, default=10, help="permutation_importance n_repeats.")
    p.add_argument("--seed", type=int, default=42, help="random_state for MI and permutation importance.")
    p.add_argument("--out", type=Path, default=ANALYSIS_OUT_DIR / "feature_importance.csv")
    p.add_argument("--top-n", type=int, default=TOP_DISAGREEMENTS_N, help="How many disagreements to print per direction.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    report, corr, redundant_groups = build_report(
        features_csv=args.features, model=args.model, task=args.task,
        n_repeats=args.n_repeats, seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    print(f"\nSaved joined feature-importance table to {args.out}")

    corr_out = args.out.parent / "feature_correlation.csv"
    corr.to_csv(corr_out)
    print(f"Saved full pairwise correlation matrix to {corr_out}")

    print("\n" + "=" * 78)
    print("TOP 15 BY PERMUTATION IMPORTANCE")
    print("=" * 78)
    cols = [
        "feature", "permutation_importance_mean", "permutation_importance_std",
        "cohens_d_max", "mutual_info_multiclass", "selection_frequency",
    ]
    with pd.option_context("display.width", 160, "display.max_colwidth", 60):
        print(report[cols].head(15).to_string(index=False))

    print_spearman(report)
    print_redundant_groups(report, redundant_groups)
    print_disagreements(report, args.top_n)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
