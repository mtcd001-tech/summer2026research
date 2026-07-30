"""Integration tests for src/extract.py: the CLI must fail with a clear
message (not a traceback) on an empty/missing dataset, and must produce a
well-formed long-format CSV for a minimal synthetic dataset."""
from __future__ import annotations

import csv
import json

from src.extract import CSV_FIELDS, main


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


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
    for ch_index, key in enumerate("print(1)"):
        events.append(
            {
                "s_n": 1, "r_t": "code", "q_id": 1, "version": 1,
                "event_type": "key",
                "data": {"key": key, "code": key, "key_event_phase": "keydown", "repeat": False, "line": 0, "ch": ch_index + 1},
                "timestamp": ts,
            }
        )
        ts += 100
    _write_json(data_root / "u1" / "s1_keystrokes.json", {"keystrokes": events, "questions": {"1": "print a number"}})
    _write_json(
        data_root / "u1" / "s1_responses.json",
        {"responses": [{"session": 1, "q_id": 1, "code": "print(1)", "explanation": "this prints one"}], "questions": {"1": "print a number"}},
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
