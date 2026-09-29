"""Shared model-free feature relevance for all five scenarios in each cohort.

Each scenario is compared with the other four (one versus rest). MI captures
univariate dependence; signed Cohen's d describes direction and effect size.
These are descriptive full-dataset scores, not held-out model importance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.scenarios import SCENARIOS


def scenario_scores(long: pd.DataFrame, dataset: str, seed: int = 42) -> pd.DataFrame:
    required = {"user", "session", "q_id", "r_t", "version", "feature", "value"}
    if missing := required - set(long.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not long.session.isin(SCENARIOS).all():
        raise ValueError("Expected session numbers 1 through 5")
    # Same explicit human-typed version rule as the extractor.
    human = long.loc[((long.session == 1) & (long.version == 1)) |
                     ((long.session != 1) & (long.version == 2))].copy()
    keys = ["user", "session", "q_id", "r_t", "feature"]
    if human.duplicated(keys).any():
        raise ValueError("Duplicate feature observations; refusing to silently average")
    rows = []
    for response_type, block in human.groupby("r_t", sort=True):
        wide = block.pivot(index=["user", "session", "q_id"], columns="feature", values="value")
        wide = wide.replace([np.inf, -np.inf], np.nan)
        sessions = wide.index.get_level_values("session")
        for session, scenario in SCENARIOS.items():
            positive = np.asarray(sessions == session)
            for feature in wide.columns:
                values = wide[feature]
                observed = values.notna().to_numpy()
                a, b = values[positive].dropna(), values[~positive].dropna()
                n_a, n_b = len(a), len(b)
                mi = d = np.nan
                status = "ok"
                if min(n_a, n_b) < 2:
                    status = "insufficient_observations"
                else:
                    pooled = np.sqrt(((n_a - 1) * a.var() + (n_b - 1) * b.var()) / (n_a + n_b - 2))
                    if pooled > 0:
                        d = (a.mean() - b.mean()) / pooled
                    x = values.to_numpy()[observed].reshape(-1, 1)
                    if np.unique(x).size == 1:
                        mi = 0.0
                        status = "constant"
                    else:
                        mi = mutual_info_classif(x, positive[observed].astype(int),
                                                 discrete_features=False, random_state=seed)[0]
                rows.append(dict(dataset=dataset, response_type=response_type, session=session,
                                 scenario=scenario, feature=feature, mutual_information=mi,
                                 cohens_d=d, n_scenario=n_a, n_rest=n_b,
                                 missing_scenario=int(positive.sum()) - n_a,
                                 missing_rest=int((~positive).sum()) - n_b, status=status))
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No human-typed feature observations")
    result["mi_rank"] = result.groupby(["dataset", "response_type", "scenario"])["mutual_information"].rank(
        ascending=False, method="min", na_option="keep")
    return result


def write_outputs(table: pd.DataFrame, out: Path, top_n: int = 20) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "scenario_importance.csv", index=False)
    for (dataset, response_type), block in table.groupby(["dataset", "response_type"]):
        folder = out / dataset / response_type
        folder.mkdir(parents=True, exist_ok=True)
        for metric in ("mutual_information", "cohens_d"):
            matrix = block.pivot(index="feature", columns="scenario", values=metric).reindex(columns=SCENARIOS.values())
            order = matrix.abs().max(axis=1).sort_values(ascending=False, kind="stable").index
            matrix = matrix.loc[order]
            matrix.to_csv(folder / f"{metric}_matrix.csv")
            shown = matrix.head(top_n) if top_n else matrix
            shown.to_csv(folder / f"{metric}_plotted.csv")
            fig, ax = plt.subplots(figsize=(12, max(5, len(shown) * .32 + 2)))
            cmap = plt.get_cmap("RdBu_r" if metric == "cohens_d" else "YlOrRd").with_extremes(bad="#dddddd")
            limit = max(float(np.nanmax(np.abs(shown.to_numpy()))) if shown.notna().any().any() else 1., .001)
            im = ax.imshow(shown.to_numpy(), aspect="auto", cmap=cmap,
                           vmin=-limit if metric == "cohens_d" else 0, vmax=limit)
            ax.set_xticks(range(5), shown.columns, rotation=20, ha="right")
            ax.set_yticks(range(len(shown)), shown.index, fontsize=8)
            for i in range(len(shown)):
                for j in range(5):
                    value = shown.iloc[i, j]
                    ax.text(j, i, f"{value:.2f}" if pd.notna(value) else "NA", ha="center", va="center",
                            fontsize=7, color="white" if pd.notna(value) and abs(value) > limit * .65 else "black")
            ax.set_title(f"{dataset} / {response_type}: {metric.replace('_', ' ')}\nEach scenario versus the other four; descriptive scores")
            fig.colorbar(im, ax=ax, label="MI (nats)" if metric == "mutual_information" else "Signed Cohen's d")
            fig.tight_layout()
            fig.savefig(folder / f"{metric}_heatmap.png", dpi=180)
            fig.savefig(folder / f"{metric}_heatmap.pdf")
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", type=Path, default=[ROOT / "dataCs1", ROOT / "dataCs2"])
    parser.add_argument("--out", type=Path, default=ROOT / "analysis/scenarios")
    parser.add_argument("--extract", action="store_true", help="Regenerate each dataset's processed/features.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=20, help="Heatmap rows; 0 means all features. CSVs always contain all features.")
    args = parser.parse_args(argv)
    if args.top_n < 0:
        parser.error("--top-n must be nonnegative")
    if len({p.name for p in args.datasets}) != len(args.datasets):
        parser.error("Dataset directory names must be unique")
    tables = []
    for dataset in args.datasets:
        features = dataset / "processed/features.csv"
        if args.extract:
            from src.dataset_features import extract_dataset
            extract_dataset(dataset, ROOT / "config.yaml")
        print(f"Scoring {features}", flush=True)
        tables.append(scenario_scores(pd.read_csv(features), dataset.name, args.seed))
    write_outputs(pd.concat(tables, ignore_index=True), args.out, args.top_n)
    (args.out / "metadata.json").write_text(json.dumps({
        "method": "one-versus-rest mutual information and sample-size pooled Cohen's d",
        "seed": args.seed, "datasets": [str(p.resolve()) for p in args.datasets],
        "missing_values": "feature-wise complete observations; no imputation",
        "scope": "descriptive full dataset; not held-out model importance or causal effects",
        "scenarios": SCENARIOS,
    }, indent=2), encoding="utf-8")
    print(f"Saved heatmaps and tables to {args.out}")


if __name__ == "__main__":
    main()
