"""Unit tests for scripts/pipeline/step1_prepare.py's version-collision
handling: the pivot index (user, session, q_id, r_t) omits `version`. Since
src/extract.py's _is_human_typed rule now keeps exactly one version per
(session, r_t) by construction (version 1 only for session 1, version 2 only
for sessions 2-5), a real collision should no longer be possible -- this is
tested as a defense-in-depth safety net, not an expected occurrence.
Silently pivoting a collision with aggfunc="first" would drop one version's
feature values with no record of it happening."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "pipeline" / "step1_prepare.py"
_PIPELINE_DIR = str(_MODULE_PATH.parent)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

_spec = importlib.util.spec_from_file_location("step1_prepare", _MODULE_PATH)
step1_prepare = importlib.util.module_from_spec(_spec)
sys.modules["step1_prepare"] = step1_prepare
_spec.loader.exec_module(step1_prepare)


def _row(user, session, q_id, r_t, version, feature, value):
    return {
        "user": user, "session": session, "q_id": q_id, "r_t": r_t,
        "version": version, "tier": "tier1", "feature": feature, "value": value,
    }


def test_no_collision_leaves_dataframe_unchanged():
    long_df = pd.DataFrame([
        _row("u1", 2, 1, "code", 2, "tier1_code_kit_cv", 1.0),
        _row("u2", 3, 4, "explanation", 2, "tier1_explanation_kit_cv", 2.0),
    ])
    out = step1_prepare._resolve_version_collisions(long_df)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), long_df.reset_index(drop=True))


def test_collision_keeps_version_2_drops_version_1():
    long_df = pd.DataFrame([
        # colliding group: both version 1 and version 2 survived for u1/s2/q1/code
        _row("u1", 2, 1, "code", 1, "tier1_code_kit_cv", 111.0),
        _row("u1", 2, 1, "code", 2, "tier1_code_kit_cv", 222.0),
        # unaffected group
        _row("u2", 3, 4, "explanation", 2, "tier1_explanation_kit_cv", 9.0),
    ])
    out = step1_prepare._resolve_version_collisions(long_df)

    assert len(out) == 2
    kept = out[(out["user"] == "u1") & (out["session"] == 2) & (out["q_id"] == 1)]
    assert len(kept) == 1
    assert kept.iloc[0]["version"] == 2
    assert kept.iloc[0]["value"] == 222.0


def test_collision_warning_is_printed(capsys):
    long_df = pd.DataFrame([
        _row("u1", 2, 1, "code", 1, "tier1_code_kit_cv", 111.0),
        _row("u1", 2, 1, "code", 2, "tier1_code_kit_cv", 222.0),
    ])
    step1_prepare._resolve_version_collisions(long_df)
    captured = capsys.readouterr()
    assert "BOTH versions present" in captured.out
    assert "user=u1 session=2 q_id=1 r_t=code" in captured.out


def test_load_wide_produces_one_row_per_group_even_with_collision(tmp_path):
    long_df = pd.DataFrame([
        _row("u1", 2, 1, "code", 1, "tier1_code_kit_cv", 111.0),
        _row("u1", 2, 1, "code", 2, "tier1_code_kit_cv", 222.0),
        _row("u2", 3, 4, "explanation", 2, "tier1_explanation_kit_cv", 9.0),
    ])
    csv_path = tmp_path / "features.csv"
    long_df.to_csv(csv_path, index=False)

    wide = step1_prepare.load_wide(csv_path)

    assert len(wide) == 2  # one row per (user, session, q_id, r_t), not per version
    u1_row = wide[wide["user"] == "u1"].iloc[0]
    assert u1_row["tier1_code_kit_cv"] == 222.0  # version 2 kept, not silently version 1


def test_add_label_binary_and_multiclass():
    wide = pd.DataFrame({"user": ["u1", "u1"], "session": [1, 3], "q_id": [1, 1], "r_t": ["code", "code"]})

    binary = step1_prepare.add_label(wide, "binary")
    assert binary["label"].tolist() == [0, 1]

    multiclass = step1_prepare.add_label(wide, "multiclass")
    assert multiclass["label"].tolist() == [0, 2]

    with pytest.raises(ValueError):
        step1_prepare.add_label(wide, "not_a_real_task")
