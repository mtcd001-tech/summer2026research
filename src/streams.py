"""Group raw events/responses by (user, session, q_id, r_t, version), and a
best-effort keystroke-replay text reconstruction used only for validation.

Feature extraction itself uses the *stored* response text for tier2/tier3,
per the project's resolved decision (stored text is authoritative; keystroke
replay is a validation/sanity-check tool -- see scripts/investigate.py and
open question #2 in the project README).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.loader import Event, UserSessionData

GroupKey = tuple[str, int, int, str, int]


@dataclass
class Group:
    user: str
    session: int
    q_id: int
    r_t: str
    version: int
    key_events: list[Event] = field(default_factory=list)
    mouse_events: list[Event] = field(default_factory=list)
    cursor_events: list[Event] = field(default_factory=list)
    text: Optional[str] = None
    source_field: Optional[str] = None
    prompt: Optional[str] = None

    @property
    def key(self) -> GroupKey:
        return (self.user, self.session, self.q_id, self.r_t, self.version)


def build_groups(session_data: UserSessionData) -> dict[GroupKey, Group]:
    """Split one user/session's events and responses into per-(q_id, r_t,
    version) groups. Every event and response contributes to at most one
    group; a group can exist from events alone, responses alone, or both.
    """
    groups: dict[GroupKey, Group] = {}

    def _get(key_tuple: GroupKey) -> Group:
        g = groups.get(key_tuple)
        if g is None:
            user, session, q_id, r_t, version = key_tuple
            g = Group(user=user, session=session, q_id=q_id, r_t=r_t, version=version)
            groups[key_tuple] = g
        return g

    for ev in session_data.events:
        g = _get((ev.user, ev.session, ev.q_id, ev.r_t, ev.version))
        if ev.event_type == "key":
            g.key_events.append(ev)
        elif ev.event_type == "mouse":
            g.mouse_events.append(ev)
        elif ev.event_type == "cursor":
            g.cursor_events.append(ev)
        # any other event_type is ignored, not fatal

    for resp in session_data.responses:
        g = _get((resp.user, resp.session, resp.q_id, resp.r_t, resp.version))
        if g.text is None:
            g.text = resp.text
            g.source_field = resp.source_field
        # if a second response field maps to the SAME (q_id, r_t, version)
        # the first one wins; that would indicate two response fields
        # collapsed onto one grouping key, which is a mapping problem to
        # resolve in RESPONSE_FIELD_MAP, not something to silently overwrite.

    for g in groups.values():
        g.key_events.sort(key=lambda e: e.timestamp)
        g.mouse_events.sort(key=lambda e: e.timestamp)
        g.cursor_events.sort(key=lambda e: e.timestamp)
        g.prompt = session_data.questions.get(g.q_id)

    return groups


def reconstruct_text(key_events: list[Event]) -> str:
    """Best-effort replay of keydown events into a text buffer.

    KNOWN BROKEN -- DO NOT USE AS A DATA SOURCE. Measured against real data
    (scripts/investigate.py's Q2) across 2,674 groups: 0 exact matches, mean
    similarity 0.307, median similarity 0.123 vs. the stored response text.
    The likely cause is exactly the Enter-key split-point limitation
    documented below, but the practical takeaway is simpler: this
    reconstruction does not reproduce real responses closely enough to trust
    for anything beyond the validation check it was built for.

    This was always intended as validation-only, never as a feature input --
    stored response text is the authoritative source for tier2/tier3
    features (see the module docstring above) and stays that way. Nothing in
    src/features/*.py or src/extract.py calls this function; only
    scripts/investigate.py does, purely to measure the (poor) agreement rate.
    Do not wire this into the extraction path.

    Interpretation: each key event's (line, ch) is the caret position AFTER
    the keystroke is applied. This is a heuristic, not a confirmed fact about
    the logger.

    Known limitation: on "Enter", the post-keystroke caret (new_line, 0) does
    not tell us the column where the split happened on the previous line, so
    this implementation always splits at the END of the current line rather
    than at the true caret column. It will visibly diverge from ground truth
    whenever Enter is pressed mid-line -- that divergence is the leading
    suspect for the near-zero agreement rate measured above.
    """
    lines: list[str] = [""]

    for ev in key_events:
        d = ev.data
        if d.get("key_event_phase") != "keydown" or d.get("repeat"):
            continue
        key = d.get("key")
        line, ch = d.get("line"), d.get("ch")
        if line is None or ch is None:
            continue
        while line >= len(lines):
            lines.append("")

        if key == "Backspace":
            # post-keystroke caret `ch` sits where the deleted character used
            # to be (the buffer here is still PRE-edit), so the char removed
            # is at index `ch`, not `ch - 1`.
            if ch > 0:
                pos = min(ch, len(lines[line]))
                lines[line] = lines[line][:pos] + lines[line][pos + 1 :]
            elif line > 0:
                lines[line - 1] = lines[line - 1] + lines[line]
                del lines[line]
        elif key == "Delete":
            pos = min(ch, len(lines[line]))
            if pos < len(lines[line]):
                lines[line] = lines[line][:pos] + lines[line][pos + 1 :]
            elif line + 1 < len(lines):
                lines[line] = lines[line] + lines[line + 1]
                del lines[line + 1]
        elif key == "Enter":
            if line > 0:
                lines.insert(line, "")
        elif isinstance(key, str) and len(key) == 1:
            pos = max(0, min(ch - 1, len(lines[line])))
            lines[line] = lines[line][:pos] + key + lines[line][pos:]
        # modifier/navigation keys (Shift, Control, ArrowLeft, Home, ...)
        # never change text

    return "\n".join(lines)
