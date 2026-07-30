"""Unit tests for src/streams.py: grouping and the reconstruct_text heuristic.

reconstruct_text is validation-only tooling (see its docstring / open
question #2), but it must still behave exactly as documented against a
hand-built sequence, since scripts/investigate.py's agreement-rate report is
only meaningful if the heuristic itself is bug-free.
"""
from __future__ import annotations

from src.loader import ResponseText, UserSessionData
from src.streams import build_groups, reconstruct_text
from tests.fixtures import key_event


def test_build_groups_splits_by_key():
    events = [
        key_event(ts=0, key="a", q_id=1, r_t="code", version=1),
        key_event(ts=10, key="b", q_id=1, r_t="explanation", version=1),
        key_event(ts=20, key="c", q_id=2, r_t="code", version=1),
    ]
    responses = [
        ResponseText(user="u1", session=1, q_id=1, r_t="code", version=1, text="ab", source_field="code"),
    ]
    session_data = UserSessionData(user="u1", session=1, events=events, responses=responses, questions={1: "prompt1"})
    groups = build_groups(session_data)

    assert set(groups) == {
        ("u1", 1, 1, "code", 1),
        ("u1", 1, 1, "explanation", 1),
        ("u1", 1, 2, "code", 1),
    }
    code_group = groups[("u1", 1, 1, "code", 1)]
    assert len(code_group.key_events) == 1
    assert code_group.text == "ab"
    assert code_group.prompt == "prompt1"
    # group with events but no matching response has text=None, not crash
    assert groups[("u1", 1, 2, "code", 1)].text is None


def test_reconstruct_text_simple_typing():
    # type "abc": each keydown's (line, ch) is the caret position AFTER
    # the character is inserted.
    events = [
        key_event(ts=0, key="a", line=0, ch=1),
        key_event(ts=10, key="b", line=0, ch=2),
        key_event(ts=20, key="c", line=0, ch=3),
    ]
    assert reconstruct_text(events) == "abc"


def test_reconstruct_text_backspace_removes_last_char():
    # type "abc", then Backspace (caret ends at ch=2, having removed 'c'),
    # then type "d" -> "abd"
    events = [
        key_event(ts=0, key="a", line=0, ch=1),
        key_event(ts=10, key="b", line=0, ch=2),
        key_event(ts=20, key="c", line=0, ch=3),
        key_event(ts=30, key="Backspace", line=0, ch=2),
        key_event(ts=40, key="d", line=0, ch=3),
    ]
    assert reconstruct_text(events) == "abd"


def test_reconstruct_text_backspace_mid_string():
    # type "abcd", then move caret conceptually back to ch=2 and Backspace
    # removes the char at that index ('b'... actually index ch=1 -> 'b'):
    # buffer "abcd", Backspace with post-caret ch=1 removes index 1 ('b')
    # -> "acd"
    events = [
        key_event(ts=0, key="a", line=0, ch=1),
        key_event(ts=10, key="b", line=0, ch=2),
        key_event(ts=20, key="c", line=0, ch=3),
        key_event(ts=30, key="d", line=0, ch=4),
        key_event(ts=40, key="Backspace", line=0, ch=1),
    ]
    assert reconstruct_text(events) == "acd"


def test_reconstruct_text_delete_key():
    # buffer "abc", Delete with caret at ch=1 (caret doesn't move) removes
    # the char AT that index ('b') -> "ac"
    events = [
        key_event(ts=0, key="a", line=0, ch=1),
        key_event(ts=10, key="b", line=0, ch=2),
        key_event(ts=20, key="c", line=0, ch=3),
        key_event(ts=30, key="Delete", line=0, ch=1),
    ]
    assert reconstruct_text(events) == "ac"


def test_reconstruct_text_ignores_repeats_and_modifiers():
    events = [
        key_event(ts=0, key="a", line=0, ch=1),
        key_event(ts=10, key="a", line=0, ch=1, repeat=True),  # ignored
        key_event(ts=20, key="Shift", line=0, ch=1),  # not text-changing
    ]
    assert reconstruct_text(events) == "a"
