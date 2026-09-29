import numpy as np
import pandas as pd
import pytest

from scripts.analysis.scenario_importance import scenario_scores, write_outputs
from scripts.normalize_data import normalize, plan_moves
from src.scenarios import SCENARIOS


def observations():
    return pd.DataFrame([
        dict(user=f"u{i}", session=s, q_id=1, r_t="code", version=1 if s == 1 else 2,
             feature=f, value=float(i + s * 10) if f == "signal" else 7.)
        for s in SCENARIOS for i in range(4) for f in ("signal", "constant")
    ])


def test_all_scenarios_and_effect_size():
    data = observations()
    result = scenario_scores(data, "cohort")
    assert set(result.scenario) == set(SCENARIOS.values())
    row = result[(result.session == 1) & (result.feature == "signal")].iloc[0]
    a = data[(data.session == 1) & (data.feature == "signal")].value
    b = data[(data.session != 1) & (data.feature == "signal")].value
    expected = (a.mean() - b.mean()) / np.sqrt(((len(a)-1)*a.var() + (len(b)-1)*b.var())/(len(a)+len(b)-2))
    assert row.cohens_d == pytest.approx(expected)
    assert row.n_scenario == 4 and row.n_rest == 16
    assert (result[result.feature == "constant"].mutual_information == 0).all()


def test_missing_scenario_is_not_zero_and_duplicates_rejected():
    data = observations()
    result = scenario_scores(data[data.session != 5], "cohort")
    assert result[result.session == 5].mutual_information.isna().all()
    with pytest.raises(ValueError, match="Duplicate"):
        scenario_scores(pd.concat([data, data.iloc[:1]]), "cohort")


def test_plot_tables_match_and_exports_exist(tmp_path):
    result = scenario_scores(observations(), "cohort")
    write_outputs(result, tmp_path, top_n=1)
    folder = tmp_path / "cohort/code"
    full = pd.read_csv(folder / "mutual_information_matrix.csv", index_col=0)
    shown = pd.read_csv(folder / "mutual_information_plotted.csv", index_col=0)
    pd.testing.assert_frame_equal(full.head(1), shown)
    assert list(full.columns) == list(SCENARIOS.values())
    assert (folder / "mutual_information_heatmap.png").stat().st_size > 1000


def test_normalization_preserves_bytes_and_is_idempotent(tmp_path):
    root = tmp_path / "raw"
    source = root / "user/Transcribe/s2_keystrokes (12).json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b'{"keystrokes": []}')
    assert normalize(root, apply=False) == 1
    assert source.exists()
    normalize(root, apply=True)
    assert (root / "user/Transcription/s2_keystrokes.json").read_bytes() == b'{"keystrokes": []}'
    assert plan_moves(root) == []
    assert (tmp_path / "normalization_manifest.json").exists()


def test_normalization_collision_leaves_sources_intact(tmp_path):
    folder = tmp_path / "user/Transcription"
    folder.mkdir(parents=True)
    for name in ("s2_keystrokes.json", "s2_keystrokes (1).json"):
        (folder / name).write_text("{}")
    with pytest.raises(ValueError, match="collision"):
        normalize(tmp_path, apply=True)
    assert len(list(folder.iterdir())) == 2
