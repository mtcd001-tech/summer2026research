"""Dataset discovery, JSON parsing, and schema normalization.

Raw keystroke/response JSON is normalized into plain dataclasses so the rest
of the pipeline never has to think about on-disk field-naming quirks again.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from src.scenarios import SCENARIOS, FOLDER_ALIASES

logger = logging.getLogger(__name__)

SESSIONS = (1, 2, 3, 4, 5)

# Each session lives in its own condition-named subfolder under the user
# folder (data/raw/<user>/<condition>/s{n}_keystrokes.json), rather than
# flat in the user folder. The file names still carry the session number.
SESSION_FOLDERS: dict[int, str] = SCENARIOS


def _session_dir(user_dir: Path, session: int) -> Path:
    existing = [user_dir / name for name in FOLDER_ALIASES[session]
                if (user_dir / name).is_dir()]
    if len(existing) > 1:
        raise DatasetError(f"Ambiguous scenario folders in {user_dir}: {existing}")
    if existing:
        return existing[0]
    return user_dir / SESSION_FOLDERS[session]


# Event types that carry session-level metadata rather than a keystroke tied
# to a (q_id, r_t, version) group -- e.g. "environment_change" reports
# viewport/device info with q_id/version legitimately null. These are
# recognized and dropped up front, not fed through the key/mouse/cursor
# schema (build_groups only ever routes "key"/"mouse"/"cursor" into a group
# anyway), and specifically not reported as "malformed" the way an event
# with a genuinely missing/unparseable field would be.
_UNGROUPED_EVENT_TYPES = {"environment_change"}

# response-JSON field name -> (r_t, version) this text belongs to.
#
# CONFIRMED (2026-08) by Ashley, who built the collection platform: version 1
# is always the ChatGPT copy-paste box (LLM output pasted in, no human typing
# by construction); version 2 is the box the human actually typed/
# transcribed/paraphrased into. Session 1 (bona fide) only ever has version 1,
# and it IS human-generated there -- there is no AI box in the bona fide
# condition, so "version 1" means something different in session 1 than it
# does in sessions 2-5.
#
# This matches both scripts/investigate.py's Q1 correlation check (version 2
# has ~400-650 mean keydowns and its length correlates ~0.9-0.98 with these
# field's lengths; version 1 has ~2-9 mean keydowns -- a paste action, not
# typing) AND a direct read of a real file: data/raw/User 1/Paraphrase/
# s4_responses.json q_id=1 has code_version_1 using the prompt's own verbose
# descriptive names ("student_name", "favorite_color", ...) while
# code_version_2 uses genuinely different, shortened names and values
# ("name"/"Red"/"Banh Mi" vs "student_name"/"Blue"/"Pho") -- exactly what a
# human paraphrase produces, not what pasting the same LLM output twice
# would. The matching keystroke counts for that file: version 1 = 2 keydowns
# (paste), version 2 = 211 keydowns (real typing). The mapping below already
# matched this once confirmed, so no values changed.
#
# src/extract.py keeps only (session=1, version=1) and (session 2-5,
# version=2) groups -- an explicit rule now, not a keydown-count heuristic --
# because version 1 for sessions 2-5 is defined to contain no human typing.
# This mapping only needs to get the *label* (which r_t/version a field's
# text belongs to) right; src/extract.py decides which groups are kept.
RESPONSE_FIELD_MAP: dict[str, tuple[str, int]] = {
    # session 1
    "code": ("code", 1),
    "explanation": ("explanation", 1),
    # sessions 2, 5
    "chatgptAnswer": ("code", 1),
    "retype": ("code", 2),
    # sessions 2-5 share these explanation field names
    "explanation_version_1": ("explanation", 1),
    "explanation_version_2": ("explanation", 2),
    # sessions 3, 4
    "code_version_1": ("code", 1),
    "code_version_2": ("code", 2),
}

_RESPONSE_META_FIELDS = {"session", "question", "q_id"}


@dataclass(frozen=True)
class Event:
    user: str
    session: int
    q_id: int
    r_t: str
    version: int
    event_type: str
    data: dict[str, Any]
    timestamp: int


@dataclass(frozen=True)
class ResponseText:
    user: str
    session: int
    q_id: int
    r_t: str
    version: int
    text: str
    source_field: str


@dataclass
class UserSessionData:
    user: str
    session: int
    events: list[Event]
    responses: list[ResponseText]
    questions: dict[int, str]


class DatasetError(RuntimeError):
    """Raised for a missing/empty/malformed dataset root.

    Meant to be caught at the CLI boundary (src/extract.py) and reported as a
    short actionable message, not a stack trace -- the dataset may simply not
    have been dropped in yet.
    """


def discover_sessions(data_root: Path) -> list[tuple[str, int]]:
    """Return sorted (user, session) pairs that have BOTH a keystrokes file
    and a responses file present under data_root/<user>/s{session}_*.json.
    """
    data_root = Path(data_root)
    if not data_root.exists() or not any(data_root.iterdir()):
        raise DatasetError(
            f"No dataset found at '{data_root}'. Expected "
            f"data_root/<user>/s{{n}}_keystrokes.json and s{{n}}_responses.json "
            f"-- see data/README.md for the exact layout. Nothing has been "
            f"placed there yet, so there is nothing to extract."
        )

    pairs: list[tuple[str, int]] = []
    for user_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for session in SESSIONS:
            session_dir = _session_dir(user_dir, session)
            k_path = session_dir / f"s{session}_keystrokes.json"
            r_path = session_dir / f"s{session}_responses.json"
            if k_path.exists() and r_path.exists():
                pairs.append((user_dir.name, session))
            elif k_path.exists() or r_path.exists():
                logger.warning(
                    "user=%s session=%s: only one of keystrokes/responses files "
                    "is present, skipping this user/session",
                    user_dir.name,
                    session,
                )

    if not pairs:
        raise DatasetError(
            f"'{data_root}' exists but contains no complete user/session file "
            f"pairs matching s{{n}}_keystrokes.json + s{{n}}_responses.json. "
            f"See data/README.md for the expected layout."
        )
    return pairs


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_questions(*raw_question_dicts: dict | None) -> dict[int, str]:
    out: dict[int, str] = {}
    for raw in raw_question_dicts:
        for k, v in (raw or {}).items():
            try:
                out.setdefault(int(k), v)
            except (TypeError, ValueError):
                logger.warning("unrecognized question id key %r; skipping", k)
    return out


def _normalize_events(user: str, session: int, raw_events: list[dict]) -> list[Event]:
    events: list[Event] = []
    ungrouped_counts: dict[str, int] = {}
    for i, raw in enumerate(raw_events):
        if raw.get("event_type") in _UNGROUPED_EVENT_TYPES:
            ungrouped_counts[raw["event_type"]] = ungrouped_counts.get(raw["event_type"], 0) + 1
            continue
        try:
            events.append(
                Event(
                    user=user,
                    session=session,
                    q_id=int(raw["q_id"]),
                    r_t=raw["r_t"],
                    version=int(raw["version"]),
                    event_type=raw["event_type"],
                    data=raw.get("data") or {},
                    timestamp=int(raw["timestamp"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "user=%s session=%s: skipping malformed event at index %d (%s)",
                user,
                session,
                i,
                exc,
            )
    for event_type, count in ungrouped_counts.items():
        logger.info(
            "user=%s session=%s: skipped %d %s event(s) -- session-level metadata, "
            "not tied to a question, not used by any feature",
            user, session, count, event_type,
        )
    return events


def _normalize_responses(user: str, session: int, raw_responses: list[dict]) -> list[ResponseText]:
    out: list[ResponseText] = []
    for entry in raw_responses:
        raw_q_id = entry.get("q_id", entry.get("question"))
        try:
            q_id = int(raw_q_id)
        except (TypeError, ValueError):
            logger.warning(
                "user=%s session=%s: response entry has no usable q_id/question "
                "field (%r); entry skipped entirely",
                user,
                session,
                raw_q_id,
            )
            continue

        found_any = False
        for field_name, (r_t, version) in RESPONSE_FIELD_MAP.items():
            if field_name not in entry or entry[field_name] is None:
                continue
            out.append(
                ResponseText(
                    user=user,
                    session=session,
                    q_id=q_id,
                    r_t=r_t,
                    version=version,
                    text=str(entry[field_name]),
                    source_field=field_name,
                )
            )
            found_any = True

        if not found_any:
            unknown_fields = sorted(set(entry) - _RESPONSE_META_FIELDS)
            logger.warning(
                "user=%s session=%s q_id=%s: none of the recognized response "
                "fields %s were present (entry had %s) -- response text LOST "
                "for this entry. Update RESPONSE_FIELD_MAP in src/loader.py if "
                "this is a new field naming convention.",
                user,
                session,
                q_id,
                sorted(RESPONSE_FIELD_MAP),
                unknown_fields,
            )
    return out


def load_user_session(data_root: Path, user: str, session: int) -> UserSessionData:
    user_dir = Path(data_root) / user
    session_dir = _session_dir(user_dir, session)
    k_path = session_dir / f"s{session}_keystrokes.json"
    r_path = session_dir / f"s{session}_responses.json"

    k_raw = _load_json(k_path)
    r_raw = _load_json(r_path)

    events = _normalize_events(user, session, k_raw.get("keystrokes", []))
    responses = _normalize_responses(user, session, r_raw.get("responses", []))
    questions = _normalize_questions(k_raw.get("questions"), r_raw.get("questions"))

    return UserSessionData(user=user, session=session, events=events, responses=responses, questions=questions)


def iter_user_sessions(data_root: Path) -> Iterator[UserSessionData]:
    for user, session in discover_sessions(data_root):
        try:
            yield load_user_session(data_root, user, session)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("user=%s session=%s: failed to load (%s); skipping", user, session, exc)
