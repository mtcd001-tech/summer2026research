"""Reusable cohort extraction and exports, built on the existing feature registry."""
from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd

from src.extract import extract_all, load_config, _missing_config_keys
from src.features import TIER1_FEATURES, TIER2_CODE_FEATURES, TIER2_EXPLANATION_FEATURES, TIER3_FEATURES
from src.loader import discover_sessions
from src.loader import iter_user_sessions
from src.streams import build_groups
from src.extract import _is_human_typed
from src.features.vietnamese_timing import DEFAULT_BIGRAMS, timing_catalog, timing_features, validate_bigrams
from src.scenarios import SCENARIOS

GROUP_COLUMNS = ["user", "session", "q_id", "r_t", "version"]


def feature_catalog() -> pd.DataFrame:
    """Derive names and implementations from the actual extraction registry."""
    rows = []
    for tier, response_type, prefix, registry in [
        ("tier1", "code", "tier1_code", TIER1_FEATURES),
        ("tier1", "explanation", "tier1_explanation", TIER1_FEATURES),
        ("tier2", "code", "tier2_code", TIER2_CODE_FEATURES),
        ("tier2", "explanation", "tier2_explanation", TIER2_EXPLANATION_FEATURES),
        ("tier3", "code", "tier3_llm", TIER3_FEATURES),
    ]:
        for name, func in registry.items():
            rows.append(dict(feature=f"{prefix}_{name}", tier=tier, r_t=response_type,
                             implementation=f"{func.__module__}.{func.__name__}",
                             description=inspect.getdoc(func) or name.replace("_", " ")))
    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


def wide_features(long: pd.DataFrame) -> pd.DataFrame:
    """Lossless pivot: retain all features and even groups with only NaNs."""
    if long.duplicated(GROUP_COLUMNS + ["feature"]).any():
        raise ValueError("Duplicate group/feature observations")
    return long.pivot(index=GROUP_COLUMNS, columns="feature", values="value").rename_axis(columns=None).reset_index()


def extract_timing_dataset(dataset: Path, bigrams=DEFAULT_BIGRAMS) -> dict:
    """Export per-response Vietnamese-style timings with identical cohort columns."""
    bigrams = validate_bigrams(bigrams)
    catalog = timing_catalog(bigrams)
    dataset = Path(dataset).resolve()
    rows = []
    for session in iter_user_sessions(dataset / "raw"):
        for group in build_groups(session).values():
            if group.r_t not in ("code", "explanation") or not _is_human_typed(group.session, group.version):
                continue
            row = dict(zip(GROUP_COLUMNS, group.key))
            row.update(dataset=dataset.name, scenario=SCENARIOS[group.session], label=group.session - 1)
            row.update(timing_features(group.key_events, bigrams))
            rows.append(row)
    if not rows:
        raise ValueError(f"No human-typed groups in {dataset}")
    meta = ["dataset"] + GROUP_COLUMNS + ["scenario", "label"]
    timing = pd.DataFrame(rows).reindex(columns=meta + list(catalog.feature))
    out = dataset / "processed"
    out.mkdir(parents=True, exist_ok=True)
    timing.to_csv(out / "features_timing.csv", index=False)
    catalog.to_csv(out / "timing_feature_catalog.csv", index=False)
    valid = timing[catalog.feature].count()
    pd.DataFrame({"valid": valid, "missing": len(timing) - valid}).rename_axis("feature").to_csv(out / "timing_feature_quality.csv")
    summary = dict(groups=len(timing), features=len(catalog), bigrams=list(bigrams),
                   vocabulary="fixed vocabulary, no corpus frequency selection",
                   timing_limits_ms=[50, 5000], iqr_multiplier=2, std_ddof=0,
                   empty="NaN", pairing="FIFO by physical code; repeats excluded",
                   rukd="most recent matched letter/space release to current keydown",
                   boundaries=GROUP_COLUMNS,
                   reference="Vietnamese_keystrokes/src/pipeline_v2/step1_extract.py")
    (out / "timing_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def combine_feature_sets(dataset: Path) -> None:
    """Join by full group identity, never by row position or participant alone."""
    out = Path(dataset) / "processed"
    existing = pd.read_csv(out / "features_wide.csv")
    timing = pd.read_csv(out / "features_timing.csv")
    keys = ["dataset"] + GROUP_COLUMNS + ["scenario", "label"]
    existing = existing.set_index(keys)
    timing = timing.set_index(keys)
    if not existing.index.is_unique or not timing.index.is_unique:
        raise ValueError("Duplicate group identities in feature tables")
    if len(existing) != len(timing) or not existing.index.isin(timing.index).all():
        raise ValueError("Feature sets have different group identities; rerun both sets with matching settings")
    pd.concat([existing, timing.reindex(existing.index)], axis=1).copy().reset_index().to_csv(
        out / "features_combined.csv", index=False)


def extract_dataset(dataset: Path, config_path: Path, enabled: set[str] | None = None) -> dict:
    """Extract a cohort into processed/; fail before publishing on feature errors."""
    dataset = Path(dataset).resolve()
    config_path = Path(config_path)
    config = load_config(config_path)
    if missing := _missing_config_keys(config):
        raise ValueError(f"Missing configuration: {missing}")
    catalog = feature_catalog()
    if enabled is not None:
        if not enabled or enabled - set(catalog.feature):
            raise ValueError(f"Empty or unknown feature selection: {sorted(enabled - set(catalog.feature))}")
        catalog = catalog[catalog.feature.isin(enabled)]
    pairs = discover_sessions(dataset / "raw")
    errors, totals, dropped, short = Counter(), Counter(), Counter(), Counter()
    long = pd.DataFrame(extract_all(dataset / "raw", config, enabled, errors, totals, dropped, short))
    if errors:
        raise RuntimeError(f"Feature extraction raised exceptions; outputs not replaced: {dict(errors)}")
    if long.empty:
        raise ValueError(f"No selected features extracted from {dataset}")
    wide = wide_features(long)
    out = dataset / "processed"
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "features.csv", index=False)
    metadata = wide[GROUP_COLUMNS].copy()
    metadata.insert(0, "dataset", dataset.name)
    metadata["scenario"] = metadata.session.map(SCENARIOS)
    metadata["label"] = metadata.session - 1
    wide = pd.concat([metadata, wide.drop(columns=GROUP_COLUMNS)], axis=1)
    wide.to_csv(out / "features_wide.csv", index=False)
    # A joint row aligns code and explanation by question; names already encode stream.
    joint_keys = ["dataset", "user", "session", "scenario", "label", "q_id", "version"]
    joint = long.pivot(index=["user", "session", "q_id", "version"], columns="feature", values="value").reset_index()
    joint.insert(0, "dataset", dataset.name)
    joint["scenario"] = joint.session.map(SCENARIOS)
    joint["label"] = joint.session - 1
    joint = joint[joint_keys + sorted(set(joint.columns) - set(joint_keys))]
    joint.to_csv(out / "features_questions.csv", index=False)
    catalog.to_csv(out / "feature_catalog.csv", index=False)
    quality = long.groupby(["r_t", "feature"]).value.agg(observations="size", valid="count").reset_index()
    quality["missing"] = quality.observations - quality.valid
    quality.to_csv(out / "feature_quality.csv", index=False)
    coverage = pd.DataFrame(pairs, columns=["user", "session"]).groupby("session").user.nunique().reindex(SCENARIOS, fill_value=0)
    pd.DataFrame({"scenario": SCENARIOS, "complete_participants": coverage}).rename_axis("session").to_csv(out / "coverage.csv")
    summary = dict(dataset=dataset.name, raw_root=str(dataset / "raw"),
                   groups=len(wide), questions=len(joint), feature_count=len(catalog),
                   long_rows=len(long), feature_errors=dict(errors),
                   config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), config=config,
                   dropped_groups={f"s{s}/v{v}": n for (s, v), n in sorted(dropped.items())},
                   short_groups={f"s{s}/v{v}": n for (s, v), n in sorted(short.items())},
                   human_version_rule="session 1: version 1; sessions 2-5: version 2",
                   preprocessing="raw values; no scaling, imputation or feature selection")
    (out / "extraction_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
