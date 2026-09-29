"""Unit tests for scripts/validate_data.py's schema checks.

environment_change is a real, recognized event type (session-level
viewport/device metadata, q_id/version legitimately null) and must NOT be
reported as a schema issue -- only genuinely unrecognized event types should be."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "validate_data.py"
_spec = importlib.util.spec_from_file_location("validate_data", SCRIPT_PATH)
validate_data = importlib.util.module_from_spec(_spec)
sys.modules["validate_data"] = validate_data
_spec.loader.exec_module(validate_data)


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _environment_change_event(**overrides):
    ev = {
        "s_n": 1, "r_t": "environment", "q_id": None, "version": None,
        "event_type": "environment_change",
        "data": {"viewportWidth": 1280, "viewportHeight": 720, "devicePixelRatio": 1.5, "viewportScale": 1},
        "timestamp": 0,
    }
    ev.update(overrides)
    return ev


def test_environment_change_event_is_not_a_schema_problem(tmp_path):
    k_path = tmp_path / "s1_keystrokes.json"
    _write_json(k_path, {"keystrokes": [_environment_change_event()], "questions": {}})

    problems, event_type_counts = validate_data._check_keystrokes_file(k_path)

    assert problems == []
    assert event_type_counts["environment_change"] == 1


def test_environment_change_missing_data_keys_is_still_flagged(tmp_path):
    k_path = tmp_path / "s1_keystrokes.json"
    bad_event = _environment_change_event(data={"viewportWidth": 1280})
    _write_json(k_path, {"keystrokes": [bad_event], "questions": {}})

    problems, event_type_counts = validate_data._check_keystrokes_file(k_path)

    assert len(problems) == 1
    assert "missing data keys" in problems[0]
    assert event_type_counts["environment_change"] == 1


def test_genuinely_unrecognized_event_type_is_still_flagged(tmp_path):
    k_path = tmp_path / "s1_keystrokes.json"
    weird_event = _environment_change_event(event_type="totally_new_type")
    _write_json(k_path, {"keystrokes": [weird_event], "questions": {}})

    problems, event_type_counts = validate_data._check_keystrokes_file(k_path)

    assert len(problems) == 1
    assert "unrecognized event_type" in problems[0]


def test_main_reports_environment_change_summary_not_as_an_issue(tmp_path, capsys):
    data_root = tmp_path / "raw"
    session_dir = data_root / "u1" / validate_data.SESSION_FOLDERS[1]
    _write_json(session_dir / "s1_keystrokes.json", {"keystrokes": [_environment_change_event()], "questions": {}})
    _write_json(session_dir / "s1_responses.json", {"responses": [], "questions": {}})

    exit_code = validate_data.main(["--data-root", str(data_root)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "environment_change events: 1" in captured.out
    assert "Schema issues" not in captured.out
