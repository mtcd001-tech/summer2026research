"""Unit tests for src/loader.py: dataset discovery and schema normalization
against hand-built synthetic JSON fixtures (no real dataset exists yet)."""
from __future__ import annotations

import json
import logging

import pytest

from src.loader import DatasetError, discover_sessions, load_user_session


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_discover_sessions_raises_on_missing_root(tmp_path):
    with pytest.raises(DatasetError, match="No dataset found"):
        discover_sessions(tmp_path / "does_not_exist")


def test_discover_sessions_raises_on_empty_root(tmp_path):
    with pytest.raises(DatasetError):
        discover_sessions(tmp_path)


def test_discover_sessions_raises_when_no_complete_pairs(tmp_path):
    # a user dir exists but only has a keystrokes file, no responses file
    _write_json(tmp_path / "u1" / "s1_keystrokes.json", {"keystrokes": [], "questions": {}})
    with pytest.raises(DatasetError):
        discover_sessions(tmp_path)


def test_discover_sessions_finds_complete_pairs(tmp_path):
    _write_json(tmp_path / "u1" / "s1_keystrokes.json", {"keystrokes": [], "questions": {}})
    _write_json(tmp_path / "u1" / "s1_responses.json", {"responses": [], "questions": {}})
    _write_json(tmp_path / "u2" / "s2_keystrokes.json", {"keystrokes": [], "questions": {}})
    _write_json(tmp_path / "u2" / "s2_responses.json", {"responses": [], "questions": {}})
    pairs = discover_sessions(tmp_path)
    assert pairs == [("u1", 1), ("u2", 2)]


def _sample_event(q_id=1, r_t="code", version=1, event_type="key", data=None, timestamp=0):
    return {
        "s_n": 1,
        "r_t": r_t,
        "q_id": q_id,
        "version": version,
        "event_type": event_type,
        "data": data or {},
        "timestamp": timestamp,
    }


def test_load_user_session_session1_fields(tmp_path):
    _write_json(
        tmp_path / "u1" / "s1_keystrokes.json",
        {
            "keystrokes": [_sample_event(data={"key": "a", "code": "KeyA", "key_event_phase": "keydown", "repeat": False, "line": 0, "ch": 1})],
            "questions": {"1": "prompt text"},
        },
    )
    _write_json(
        tmp_path / "u1" / "s1_responses.json",
        {
            "responses": [{"session": 1, "question": 1, "q_id": 1, "code": "print(1)", "explanation": "it prints"}],
            "questions": {"1": "prompt text"},
        },
    )
    data = load_user_session(tmp_path, "u1", 1)
    assert len(data.events) == 1
    texts = {(r.r_t, r.version): r.text for r in data.responses}
    assert texts[("code", 1)] == "print(1)"
    assert texts[("explanation", 1)] == "it prints"
    assert data.questions[1] == "prompt text"


def test_load_user_session_session2_transcribe_fields(tmp_path):
    _write_json(tmp_path / "u1" / "s2_keystrokes.json", {"keystrokes": [], "questions": {"1": "prompt"}})
    _write_json(
        tmp_path / "u1" / "s2_responses.json",
        {
            "responses": [
                {
                    "session": 2, "question": 1, "q_id": 1,
                    "chatgptAnswer": "ai code",
                    "retype": "typed code",
                    "explanation_version_1": "exp v1",
                    "explanation_version_2": "exp v2",
                }
            ],
            "questions": {"1": "prompt"},
        },
    )
    data = load_user_session(tmp_path, "u1", 2)
    texts = {(r.r_t, r.version): r.text for r in data.responses}
    assert texts[("code", 1)] == "ai code"
    assert texts[("code", 2)] == "typed code"
    assert texts[("explanation", 1)] == "exp v1"
    assert texts[("explanation", 2)] == "exp v2"


def test_load_user_session_session3_mimicry_fields(tmp_path):
    _write_json(tmp_path / "u1" / "s3_keystrokes.json", {"keystrokes": [], "questions": {"1": "prompt"}})
    _write_json(
        tmp_path / "u1" / "s3_responses.json",
        {
            "responses": [
                {
                    "session": 3, "question": 1, "q_id": 1,
                    "code_version_1": "code v1",
                    "code_version_2": "code v2",
                    "explanation_version_1": "exp v1",
                    "explanation_version_2": "exp v2",
                }
            ],
            "questions": {"1": "prompt"},
        },
    )
    data = load_user_session(tmp_path, "u1", 3)
    texts = {(r.r_t, r.version): r.text for r in data.responses}
    assert texts[("code", 1)] == "code v1"
    assert texts[("code", 2)] == "code v2"


def test_malformed_event_is_skipped_and_logged(tmp_path, caplog):
    _write_json(
        tmp_path / "u1" / "s1_keystrokes.json",
        {
            "keystrokes": [
                {"s_n": 1, "r_t": "code"},  # missing required keys
                _sample_event(data={"key": "a", "code": "KeyA", "key_event_phase": "keydown", "repeat": False, "line": 0, "ch": 1}),
            ],
            "questions": {},
        },
    )
    _write_json(tmp_path / "u1" / "s1_responses.json", {"responses": [], "questions": {}})
    with caplog.at_level(logging.WARNING):
        data = load_user_session(tmp_path, "u1", 1)
    assert len(data.events) == 1  # malformed event skipped, valid one kept
    assert any("malformed event" in rec.message for rec in caplog.records)


def test_unrecognized_response_field_is_dropped_and_logged(tmp_path, caplog):
    _write_json(tmp_path / "u1" / "s1_keystrokes.json", {"keystrokes": [], "questions": {}})
    _write_json(
        tmp_path / "u1" / "s1_responses.json",
        {"responses": [{"session": 1, "q_id": 1, "totally_unknown_field": "some text"}], "questions": {}},
    )
    with caplog.at_level(logging.WARNING):
        data = load_user_session(tmp_path, "u1", 1)
    assert data.responses == []  # nothing recognized -> no ResponseText produced
    assert any("LOST" in rec.message for rec in caplog.records)
