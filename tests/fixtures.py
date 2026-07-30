"""Synthetic event/response generators with KNOWN expected feature values.

No real dataset exists yet, so every "expected" number in the test suite is
computed by hand from a generator's own inputs here -- never copied from a
sample file. See the docstring of each test for the by-hand derivation.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from src.loader import Event
from src.streams import Group

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_test_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def key_event(
    *, ts: int, key: str, code: str | None = None, phase: str = "keydown",
    repeat: bool = False, line: int = 0, ch: int = 0,
    user: str = "u1", session: int = 1, q_id: int = 1, r_t: str = "code", version: int = 1,
) -> Event:
    return Event(
        user=user, session=session, q_id=q_id, r_t=r_t, version=version,
        event_type="key",
        data={
            "key": key,
            "code": code or f"Key{key}",
            "key_event_phase": phase,
            "repeat": repeat,
            "line": line,
            "ch": ch,
        },
        timestamp=ts,
    )


def mouse_event(
    *, ts: int, x: float, y: float,
    user: str = "u1", session: int = 1, q_id: int = 1, r_t: str = "code", version: int = 1,
) -> Event:
    return Event(
        user=user, session=session, q_id=q_id, r_t=r_t, version=version,
        event_type="mouse", data={"x": x, "y": y}, timestamp=ts,
    )


def cursor_event(
    *, ts: int, line: int, ch: int,
    user: str = "u1", session: int = 1, q_id: int = 1, r_t: str = "code", version: int = 1,
) -> Event:
    return Event(
        user=user, session=session, q_id=q_id, r_t=r_t, version=version,
        event_type="cursor", data={"line": line, "ch": ch}, timestamp=ts,
    )


def make_group(
    *, key_events=None, mouse_events=None, cursor_events=None, text=None, prompt=None,
    user: str = "u1", session: int = 1, q_id: int = 1, r_t: str = "code", version: int = 1,
) -> Group:
    g = Group(user=user, session=session, q_id=q_id, r_t=r_t, version=version)
    g.key_events = list(key_events or [])
    g.mouse_events = list(mouse_events or [])
    g.cursor_events = list(cursor_events or [])
    g.text = text
    g.prompt = prompt
    return g
