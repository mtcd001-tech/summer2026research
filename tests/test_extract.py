"""Integration tests for src/extract.py: the CLI must fail with a clear
message (not a traceback) on an empty/missing dataset, and must produce a
well-formed long-format CSV for a minimal synthetic dataset."""
from __future__ import annotations

import csv
import json
import logging
from collections import Counter

import pytest
import yaml

from src.extract import CSV_FIELDS, TIER1_FEATURES, _group_rows, main
from src.features import all_feature_names
from src.loader import SESSION_FOLDERS
from tests.fixtures import load_test_config, make_group

# Read the real threshold rather than hardcoding a keydown count, so these
# fixtures don't silently fall out of sync with config.yaml again (that's
# exactly what broke them when this was raised from 20 to 100 for the larger
# dataset). It's a sanity-check threshold now, not an exclusion filter --
# see _is_human_typed in src/extract.py for the actual keep/drop rule.
_SANITY_MIN_KEYDOWN_COUNT = load_test_config()["tier1"]["filtering"]["sanity_check_min_keydown_count"]


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _typed_text(min_chars: int, pattern: str = "print(1)\n") -> str:
    """`pattern` repeated enough times to exceed `min_chars` characters
    (comfortably, with a +10 margin) -- one character == one keydown."""
    repeats = (min_chars + 10) // len(pattern) + 1
    return pattern * repeats


def test_main_fails_clearly_on_empty_data_root(tmp_path, capsys):
    out = tmp_path / "features.csv"
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    exit_code = main(["--data-root", str(empty_root), "--out", str(out)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert not out.exists()


def test_main_lists_features_without_needing_data(capsys):
    exit_code = main(["--list-features"])
    assert exit_code == 0
    captured = capsys.readouterr()
    names = captured.out.strip().splitlines()
    assert "tier1_code_kit_cv" in names
    assert "tier2_explanation_ttr" in names
    assert "tier3_llm_blank_line_ratio" in names


def _build_minimal_dataset(data_root):
    events = []
    ts = 0
    # session=1 version=1 is always kept by _is_human_typed regardless of
    # keydown count, but sized to clear the sanity-check threshold anyway so
    # this fixture doesn't trip the "short kept group" warning in tests that
    # don't care about it.
    code_text = _typed_text(_SANITY_MIN_KEYDOWN_COUNT)
    for ch_index, key in enumerate(code_text):
        events.append(
            {
                "s_n": 1, "r_t": "code", "q_id": 1, "version": 1,
                "event_type": "key",
                "data": {"key": key, "code": key, "key_event_phase": "keydown", "repeat": False, "line": 0, "ch": ch_index + 1},
                "timestamp": ts,
            }
        )
        ts += 100
    session_dir = data_root / "u1" / SESSION_FOLDERS[1]
    _write_json(session_dir / "s1_keystrokes.json", {"keystrokes": events, "questions": {"1": "print a number"}})
    _write_json(
        session_dir / "s1_responses.json",
        {"responses": [{"session": 1, "q_id": 1, "code": code_text, "explanation": "this prints one"}], "questions": {"1": "print a number"}},
    )


def test_main_writes_long_format_csv(tmp_path):
    data_root = tmp_path / "raw"
    _build_minimal_dataset(data_root)
    out = tmp_path / "features.csv"

    exit_code = main(["--data-root", str(data_root), "--out", str(out)])
    assert exit_code == 0
    assert out.exists()

    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "expected at least one feature row"
    assert set(rows[0]) == set(CSV_FIELDS)

    feature_names = {r["feature"] for r in rows}
    assert "tier1_code_kit_cv" in feature_names
    assert "tier2_code_identifier_diversity" in feature_names
    assert "tier3_llm_blank_line_ratio" in feature_names
    # explanation-only features should never appear on the code group and
    # vice versa
    code_rows = [r for r in rows if r["r_t"] == "code"]
    assert all(not f.startswith("tier2_explanation_") for f in {r["feature"] for r in code_rows})


# ---------------------------------------------------------------------------
# Explicit human-typed rule (Ashley, who built the collection platform:
# version 1 for sessions 2-5 is the ChatGPT paste box, no human typing by
# construction; version 2 is what the human actually typed). This replaced
# the old count-threshold heuristic -- these tests check the RULE (by
# session/version), not a keydown count.
# ---------------------------------------------------------------------------

def _build_dataset_session2_both_versions(data_root, *, version1_keydowns=3, version2_keydowns=None):
    """Session 2 (Transcribe), q_id=1, r_t=code, with BOTH version 1 (the
    paste box) and version 2 (human-typed) present. version1_keydowns/
    version2_keydowns control how many keydowns each gets, so tests can
    prove the drop rule is about `version`, not keydown count (e.g. giving
    version 1 plenty of keydowns and confirming it's still dropped)."""
    if version2_keydowns is None:
        version2_keydowns = int(_SANITY_MIN_KEYDOWN_COUNT) + 10

    events = []
    ts = 0
    for version, n_keydowns in ((1, version1_keydowns), (2, version2_keydowns)):
        text = _typed_text(n_keydowns)[:n_keydowns] if n_keydowns > 0 else ""
        for ch_index, key in enumerate(text):
            events.append(
                {
                    "s_n": 2, "r_t": "code", "q_id": 1, "version": version,
                    "event_type": "key",
                    "data": {"key": key, "code": key, "key_event_phase": "keydown", "repeat": False, "line": 0, "ch": ch_index + 1},
                    "timestamp": ts,
                }
            )
            ts += 100

    session_dir = data_root / "u1" / SESSION_FOLDERS[2]
    _write_json(session_dir / "s2_keystrokes.json", {"keystrokes": events, "questions": {"1": "q1"}})
    _write_json(
        session_dir / "s2_responses.json",
        {
            "responses": [
                {"session": 2, "q_id": 1, "chatgptAnswer": "pasted llm output", "retype": "human typed this"},
            ],
            "questions": {"1": "q1"},
        },
    )


def test_version1_paste_box_is_dropped_even_with_plenty_of_keydowns(tmp_path):
    """The rule is about `version`, not keydown count: version 1 gets MORE
    keydowns than version 2 here and must still be dropped."""
    data_root = tmp_path / "raw"
    _build_dataset_session2_both_versions(
        data_root, version1_keydowns=int(_SANITY_MIN_KEYDOWN_COUNT) + 50, version2_keydowns=int(_SANITY_MIN_KEYDOWN_COUNT) + 10
    )
    out = tmp_path / "features.csv"

    exit_code = main(["--data-root", str(data_root), "--out", str(out)])
    assert exit_code == 0
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    versions_present = {r["version"] for r in rows}
    assert "2" in versions_present
    assert "1" not in versions_present, "version 1 (paste box) must be dropped regardless of keydown count"


def test_version1_drop_is_logged_by_session_and_version(tmp_path, caplog):
    data_root = tmp_path / "raw"
    _build_dataset_session2_both_versions(data_root)
    out = tmp_path / "features.csv"
    caplog.set_level(logging.INFO)

    main(["--data-root", str(data_root), "--out", str(out)])

    messages = [r.getMessage() for r in caplog.records]
    assert any("dropped" in m and "not human-typed" in m for m in messages)
    assert any("session=2 version=1" in m for m in messages)


def test_short_but_kept_version2_group_is_flagged_not_dropped(tmp_path, caplog):
    """A human-typed (version 2) group below the sanity-check threshold is
    still included in the output -- just flagged, not excluded."""
    data_root = tmp_path / "raw"
    _build_dataset_session2_both_versions(data_root, version1_keydowns=3, version2_keydowns=5)
    out = tmp_path / "features.csv"
    caplog.set_level(logging.WARNING)

    exit_code = main(["--data-root", str(data_root), "--out", str(out)])
    assert exit_code == 0
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    versions_present = {r["version"] for r in rows}
    assert "2" in versions_present, "short version-2 group must still be kept, only flagged"

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("sanity check" in m for m in warnings)
    assert any("session=2 version=2" in m for m in warnings)


def test_session1_version1_is_always_kept(tmp_path):
    """Bona fide (session 1) has no AI paste box -- version 1 there IS the
    human-generated response and must never be dropped by the rule."""
    data_root = tmp_path / "raw"
    _build_minimal_dataset(data_root)  # session=1, version=1
    out = tmp_path / "features.csv"

    exit_code = main(["--data-root", str(data_root), "--out", str(out)])
    assert exit_code == 0
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "session 1 version 1 must be kept"


def test_main_respects_exclude_features(tmp_path):
    data_root = tmp_path / "raw"
    _build_minimal_dataset(data_root)
    out = tmp_path / "features.csv"

    exit_code = main(
        ["--data-root", str(data_root), "--out", str(out), "--exclude-features", "tier1_code_kit_cv"]
    )
    assert exit_code == 0
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert all(r["feature"] != "tier1_code_kit_cv" for r in rows)


# ---------------------------------------------------------------------------
# BUG 1: --features / --exclude-features with no values, or both given, or
# resolving to an empty selection.
# ---------------------------------------------------------------------------

def test_features_flag_requires_at_least_one_value(tmp_path):
    data_root = tmp_path / "raw"
    _build_minimal_dataset(data_root)
    out = tmp_path / "features.csv"
    with pytest.raises(SystemExit):
        main(["--data-root", str(data_root), "--out", str(out), "--features"])
    assert not out.exists()


def test_exclude_features_flag_requires_at_least_one_value(tmp_path):
    data_root = tmp_path / "raw"
    _build_minimal_dataset(data_root)
    out = tmp_path / "features.csv"
    with pytest.raises(SystemExit):
        main(["--data-root", str(data_root), "--out", str(out), "--exclude-features"])
    assert not out.exists()


def test_features_and_exclude_features_are_mutually_exclusive(tmp_path):
    data_root = tmp_path / "raw"
    _build_minimal_dataset(data_root)
    out = tmp_path / "features.csv"
    with pytest.raises(SystemExit):
        main(
            [
                "--data-root", str(data_root), "--out", str(out),
                "--features", "tier1_code_kit_cv",
                "--exclude-features", "tier1_code_kit_cv",
            ]
        )
    assert not out.exists()


def test_excluding_every_feature_errors_instead_of_writing_empty_csv(tmp_path, capsys):
    data_root = tmp_path / "raw"
    _build_minimal_dataset(data_root)
    out = tmp_path / "features.csv"

    exit_code = main(
        ["--data-root", str(data_root), "--out", str(out), "--exclude-features", *all_feature_names()]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert not out.exists()


# ---------------------------------------------------------------------------
# BUG 2: empty/malformed/incomplete config.yaml must fail loudly, not turn
# the whole run into a silent wall of NaN.
# ---------------------------------------------------------------------------

def test_empty_config_file_errors_instead_of_producing_all_nan(tmp_path, capsys):
    data_root = tmp_path / "raw"
    _build_minimal_dataset(data_root)
    out = tmp_path / "features.csv"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("", encoding="utf-8")

    exit_code = main(
        ["--data-root", str(data_root), "--out", str(out), "--config", str(config_path)]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert not out.exists()


def test_malformed_config_yaml_errors_cleanly(tmp_path, capsys):
    data_root = tmp_path / "raw"
    _build_minimal_dataset(data_root)
    out = tmp_path / "features.csv"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("tier1: [unclosed", encoding="utf-8")

    exit_code = main(
        ["--data-root", str(data_root), "--out", str(out), "--config", str(config_path)]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert not out.exists()


def test_missing_data_and_missing_config_give_the_same_style_of_error(tmp_path, capsys):
    """Missing dataset already produced a clean 'error: ...' message; missing
    config used to raise a raw traceback instead. Both paths should now agree."""
    data_root = tmp_path / "raw"
    _build_minimal_dataset(data_root)
    out = tmp_path / "features.csv"
    missing_config = tmp_path / "does_not_exist.yaml"

    exit_code = main(
        ["--data-root", str(data_root), "--out", str(out), "--config", str(missing_config)]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


def test_config_missing_required_keys_names_them(tmp_path, capsys):
    data_root = tmp_path / "raw"
    _build_minimal_dataset(data_root)
    out = tmp_path / "features.csv"
    config_path = tmp_path / "config.yaml"
    # A well-formed mapping, but missing every tier1 sub-key the feature
    # functions actually need.
    config_path.write_text(yaml.safe_dump({"tier1": {}, "tier2": {}, "tier3": {}}), encoding="utf-8")

    exit_code = main(
        ["--data-root", str(data_root), "--out", str(out), "--config", str(config_path)]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "tier1.keys.exclude_repeats" in captured.err
    assert not out.exists()


def test_valid_config_still_works(tmp_path):
    """Sanity check that the new validation doesn't false-positive on the
    project's real config.yaml."""
    data_root = tmp_path / "raw"
    _build_minimal_dataset(data_root)
    out = tmp_path / "features.csv"

    exit_code = main(["--data-root", str(data_root), "--out", str(out)])
    assert exit_code == 0
    assert out.exists()


# ---------------------------------------------------------------------------
# BUG 3: per-feature failure tally must be visible, distinct from
# legitimately-missing data.
# ---------------------------------------------------------------------------

def test_feature_that_always_raises_is_logged_prominently(tmp_path, monkeypatch, caplog):
    data_root = tmp_path / "raw"
    _build_minimal_dataset(data_root)
    out = tmp_path / "features.csv"

    def _boom(group, config):
        raise ValueError("synthetic failure for test")

    monkeypatch.setitem(TIER1_FEATURES, "always_fails", _boom)
    caplog.set_level(logging.WARNING)

    exit_code = main(["--data-root", str(data_root), "--out", str(out)])
    assert exit_code == 0

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    matches = [m for m in warnings if "tier1_code_always_fails" in m]
    assert matches, f"expected a warning naming the failing feature, got: {warnings}"
    assert any("ALL" in m for m in matches)


def test_feature_failure_counter_only_flags_the_failing_feature(tmp_path, monkeypatch, caplog):
    data_root = tmp_path / "raw"
    _build_minimal_dataset(data_root)
    out = tmp_path / "features.csv"

    def _boom(group, config):
        raise ValueError("synthetic failure for test")

    monkeypatch.setitem(TIER1_FEATURES, "always_fails", _boom)
    caplog.set_level(logging.WARNING)

    main(["--data-root", str(data_root), "--out", str(out)])

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("tier1_code_kit_cv" in m for m in warnings)


# ---------------------------------------------------------------------------
# Unrecognized r_t: the group is skipped entirely (chosen over emitting
# tier1-only rows) so every feature name ever written stays inside the
# tier1_code_*/tier1_explanation_* schema that all_feature_names() and
# --features/--exclude-features agree on.
# ---------------------------------------------------------------------------

def test_unrecognized_r_t_skips_the_group_entirely(caplog):
    config = load_test_config()
    group = make_group(r_t="mystery", text="whatever", prompt="p")
    caplog.set_level(logging.WARNING)

    rows = _group_rows(group, config, None, Counter(), Counter())

    assert rows == []
    assert any("unrecognized r_t" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Verification: all_feature_names() must agree exactly with the feature
# names _group_rows actually produces (expanded tier1_code_*/tier1_explanation_*
# names, not bare tier1_* names) -- otherwise --features/--exclude-features
# silently match nothing.
# ---------------------------------------------------------------------------

def test_group_rows_feature_names_are_subset_of_all_feature_names():
    config = load_test_config()
    all_names = set(all_feature_names())

    code_group = make_group(r_t="code", text="x = 1\n", prompt="write x")
    explanation_group = make_group(r_t="explanation", text="This is a test. Another sentence.", prompt="explain")

    for group in (code_group, explanation_group):
        rows = _group_rows(group, config, None, Counter(), Counter())
        produced = {r["feature"] for r in rows}
        assert produced, "expected at least one row"
        assert produced <= all_names, f"produced names not in all_feature_names(): {produced - all_names}"


def test_every_tier1_name_appears_in_both_code_and_explanation_variants():
    all_names = set(all_feature_names())
    tier1_code_names = {n for n in all_names if n.startswith("tier1_code_")}
    assert tier1_code_names, "expected at least one tier1_code_ name"

    for name in tier1_code_names:
        explanation_variant = "tier1_explanation_" + name[len("tier1_code_"):]
        assert explanation_variant in all_names
