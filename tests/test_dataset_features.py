import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset_features import extract_dataset, extract_timing_dataset, combine_feature_sets, wide_features, feature_catalog
from src.features import all_feature_names
from src.features.vietnamese_timing import collect_timings, summarize, validate_bigrams
from tests.fixtures import key_event


def test_reference_timing_values_and_repeated_or_orphan_releases():
    events = [key_event(ts=0, key="a"), key_event(ts=10, key="a", repeat=True),
              key_event(ts=100, key="a", phase="keyup"),
              key_event(ts=120, key="a", phase="keyup"),
              key_event(ts=250, key="b"), key_event(ts=330, key="b", phase="keyup")]
    values = collect_timings(events, ["a->b"])
    assert values[("kht", "a")] == [100]
    assert values[("kit", "a->b")] == [250]
    assert values[("rukd", "a->b")] == [150]
    assert values[("kht", "b")] == [80]


def test_reference_statistics_and_empty_values():
    result = summarize([10, 100, 200, 300, 6000])
    assert result["mean"] == 200
    assert result["std"] == pytest.approx(np.std([100, 200, 300]))
    assert result["cv"] == pytest.approx(result["std"] / 200)
    assert result["range"] == 200
    assert result["median"] == 200
    assert all(np.isnan(v) for v in summarize([]).values())
    with pytest.raises(ValueError):
        validate_bigrams(["a->b", "a->b"])


def test_catalog_matches_registry_and_nan_groups_survive_pivot():
    assert set(feature_catalog().feature) == set(all_feature_names())
    row = dict(user="u", session=1, q_id=1, r_t="code", version=1, feature="f", value=np.nan)
    wide = wide_features(pd.DataFrame([row]))
    assert len(wide) == 1 and np.isnan(wide.f.iloc[0])
    with pytest.raises(ValueError, match="Duplicate"):
        wide_features(pd.DataFrame([row, row]))


def test_cohort_extraction_and_join(tmp_path):
    dataset = tmp_path / "cohort"
    folder = dataset / "raw/u/Bona fide"
    folder.mkdir(parents=True)
    events = [key_event(ts=0, key="a"), key_event(ts=100, key="a", phase="keyup"),
              key_event(ts=250, key="b"), key_event(ts=350, key="b", phase="keyup")]
    raw = [dict(q_id=1, r_t="code", version=1, event_type=e.event_type, data=e.data, timestamp=e.timestamp) for e in events]
    (folder / "s1_keystrokes.json").write_text(json.dumps({"keystrokes": raw}))
    (folder / "s1_responses.json").write_text(json.dumps({"responses": [{"q_id": 1, "code": "ab"}]}))
    config = Path(__file__).resolve().parents[1] / "config.yaml"
    result = extract_dataset(dataset, config, {"tier1_code_kit_cv"})
    assert result["groups"] == 1
    extract_timing_dataset(dataset, ["a->b"])
    combine_feature_sets(dataset)
    combined = pd.read_csv(dataset / "processed/features_combined.csv")
    assert len(combined) == 1
    assert combined["kit_a->b_mean"].iloc[0] == 250
    assert combined.scenario.iloc[0] == "Bona fide"
    assert combined.dataset.iloc[0] == "cohort"
    assert combined.label.iloc[0] == 0
    assert (dataset / "processed/feature_quality.csv").exists()


def test_combine_aligns_identity_and_rejects_unmatched_groups(tmp_path):
    out = tmp_path / "processed"
    out.mkdir()
    base = dict(dataset="d", user="u", session=1, q_id=1, version=1,
                scenario="Bona fide", label=0)
    existing = pd.DataFrame([dict(base, r_t="code", f=1.), dict(base, r_t="explanation", f=2.)])
    timing = pd.DataFrame([dict(base, r_t="explanation", timing=20.), dict(base, r_t="code", timing=10.)])
    existing.to_csv(out / "features_wide.csv", index=False)
    timing.to_csv(out / "features_timing.csv", index=False)
    combine_feature_sets(tmp_path)
    result = pd.read_csv(out / "features_combined.csv")
    assert result.timing.tolist() == [10., 20.]
    timing.iloc[:1].to_csv(out / "features_timing.csv", index=False)
    with pytest.raises(ValueError, match="different group identities"):
        combine_feature_sets(tmp_path)
