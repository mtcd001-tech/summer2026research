"""Tier 1: keystroke-timing, revision, burst, and mouse/cursor features.

Every function takes a `src.streams.Group` (already filtered to one
(user, session, q_id, r_t, version) cell) plus the parsed config dict, and
returns a single float (math.nan when the statistic can't be computed for
that group, e.g. too few events). Feature names here are UNPREFIXED; the
extraction CLI adds the "tier1_{r_t}_" prefix since these functions run
identically whichever r_t ("code" or "explanation") the group belongs to.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.loader import Event
from src.statutils import binned_entropy, coefficient_of_variation, percentile_ratio, safe_ratio
from src.streams import Group


@dataclass(frozen=True)
class KeyPair:
    code: str
    key: Optional[str]
    keydown_ts: int
    keyup_ts: Optional[int]

    @property
    def hold_time(self) -> float:
        if self.keyup_ts is None:
            return math.nan
        return float(self.keyup_ts - self.keydown_ts)


def _pair_keydown_keyup(key_events: list[Event], exclude_repeats: bool) -> list[KeyPair]:
    """Match each keydown to its keyup by `code`, FIFO per code (so rapid
    repeats of the same physical key can't cross-match to the wrong keyup).
    Orphan keyups (no keydown seen) are dropped; orphan keydowns (no keyup
    seen, e.g. truncated log) are kept with keyup_ts=None so they still count
    toward inter-key-interval timing, just not hold time.
    """
    pending: dict[str, list[tuple[int, Optional[str]]]] = {}
    pairs: list[KeyPair] = []
    for ev in key_events:
        d = ev.data
        phase = d.get("key_event_phase")
        code = d.get("code")
        if code is None:
            continue
        if phase == "keydown":
            if exclude_repeats and d.get("repeat"):
                continue
            pending.setdefault(code, []).append((ev.timestamp, d.get("key")))
        elif phase == "keyup":
            queue = pending.get(code)
            if queue:
                ts_down, key = queue.pop(0)
                pairs.append(KeyPair(code=code, key=key, keydown_ts=ts_down, keyup_ts=ev.timestamp))

    for code, queue in pending.items():
        for ts_down, key in queue:
            pairs.append(KeyPair(code=code, key=key, keydown_ts=ts_down, keyup_ts=None))

    pairs.sort(key=lambda p: p.keydown_ts)
    return pairs


def _pairs(group: Group, config: dict) -> list[KeyPair]:
    exclude_repeats = config["tier1"]["keys"]["exclude_repeats"]
    return _pair_keydown_keyup(group.key_events, exclude_repeats)


def _keydown_timestamps(pairs: list[KeyPair]) -> list[int]:
    return [p.keydown_ts for p in pairs]


def _inter_key_intervals(pairs: list[KeyPair]) -> list[float]:
    ts = _keydown_timestamps(pairs)
    return [float(b - a) for a, b in zip(ts, ts[1:])]


def _hold_times(pairs: list[KeyPair]) -> list[float]:
    return [p.hold_time for p in pairs if not math.isnan(p.hold_time)]


def _release_to_next_keydown(pairs: list[KeyPair]) -> list[float]:
    values = []
    for a, b in zip(pairs, pairs[1:]):
        if a.keyup_ts is None:
            continue
        values.append(float(b.keydown_ts - a.keyup_ts))
    return values


def _keydown_stream(group: Group, config: dict) -> list[dict]:
    """Ordered list of non-repeat keydown `data` payloads for this group."""
    exclude_repeats = config["tier1"]["keys"]["exclude_repeats"]
    out = []
    for ev in group.key_events:
        d = ev.data
        if d.get("key_event_phase") != "keydown":
            continue
        if exclude_repeats and d.get("repeat"):
            continue
        out.append(d)
    return out


def keydown_count(group: Group, config: dict) -> int:
    """Number of (non-repeat, per config) keydown events in this group.

    Public (unlike the other helpers here) because src/extract.py uses it as
    a secondary sanity check on which groups it keeps. Which groups are kept
    is an explicit rule (version 1 for sessions 2-5 is the ChatGPT paste box,
    confirmed by Ashley who built the collection platform, and is always
    dropped regardless of keydown count -- see RESPONSE_FIELD_MAP's
    docstring in src/loader.py); this count is only used to flag KEPT groups
    that are surprisingly short, not to decide what's kept.
    """
    return len(_keydown_stream(group, config))


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def kit_cv(group: Group, config: dict) -> float:
    return coefficient_of_variation(_inter_key_intervals(_pairs(group, config)))


def kit_p90_p50(group: Group, config: dict) -> float:
    return percentile_ratio(_inter_key_intervals(_pairs(group, config)), hi=90, lo=50)


def kit_p50_p10(group: Group, config: dict) -> float:
    return percentile_ratio(_inter_key_intervals(_pairs(group, config)), hi=50, lo=10)


def kit_long_pause_fraction(group: Group, config: dict) -> float:
    intervals = _inter_key_intervals(_pairs(group, config))
    if not intervals:
        return math.nan
    median = float(np.median(intervals))
    if median == 0:
        return math.nan
    mult = config["tier1"]["pause"]["long_pause_multiplier"]
    long_count = sum(1 for v in intervals if v > mult * median)
    return long_count / len(intervals)


def kit_pause_time_fraction(group: Group, config: dict) -> float:
    """Total time spent in "long pauses" (see kit_long_pause_fraction)
    relative to the group's total elapsed time (last keydown - first
    keydown). Elapsed time is measured over keydowns only, matching the
    inter-key-interval definition used for the numerator.
    """
    pairs = _pairs(group, config)
    intervals = _inter_key_intervals(pairs)
    if not intervals:
        return math.nan
    median = float(np.median(intervals))
    if median == 0:
        return math.nan
    mult = config["tier1"]["pause"]["long_pause_multiplier"]
    pause_time = sum(v for v in intervals if v > mult * median)
    ts = _keydown_timestamps(pairs)
    total_time = ts[-1] - ts[0]
    return safe_ratio(pause_time, total_time)


def kit_pause_entropy(group: Group, config: dict) -> float:
    intervals = _inter_key_intervals(_pairs(group, config))
    n_bins = config["tier1"]["pause"]["entropy_bins"]
    return binned_entropy(intervals, n_bins)


def kht_cv(group: Group, config: dict) -> float:
    return coefficient_of_variation(_hold_times(_pairs(group, config)))


def kht_p90_p50(group: Group, config: dict) -> float:
    return percentile_ratio(_hold_times(_pairs(group, config)), hi=90, lo=50)


def kht_p50_p10(group: Group, config: dict) -> float:
    return percentile_ratio(_hold_times(_pairs(group, config)), hi=50, lo=10)


def rkdt_cv(group: Group, config: dict) -> float:
    return coefficient_of_variation(_release_to_next_keydown(_pairs(group, config)))


# --------------------------------------------------------------------------
# Revision
# --------------------------------------------------------------------------

def _is_deletion(d: dict, config: dict) -> bool:
    return d.get("key") in config["tier1"]["keys"]["deletion_keys"]


def _is_insertion(d: dict, config: dict) -> bool:
    key = d.get("key")
    if key in config["tier1"]["keys"]["insertion_special_keys"]:
        return True
    return isinstance(key, str) and len(key) == 1


def revision_delete_ratio(group: Group, config: dict) -> float:
    stream = _keydown_stream(group, config)
    deletes = sum(1 for d in stream if _is_deletion(d, config))
    inserts = sum(1 for d in stream if _is_insertion(d, config))
    return safe_ratio(deletes, deletes + inserts)


def revision_deletion_episodes_per_100_keys(group: Group, config: dict) -> float:
    """Count maximal runs of consecutive deletion keystrokes ("episodes"),
    normalized per 100 keys typed in this group.
    """
    stream = _keydown_stream(group, config)
    if not stream:
        return math.nan
    episodes = 0
    in_episode = False
    for d in stream:
        if _is_deletion(d, config):
            if not in_episode:
                episodes += 1
                in_episode = True
        else:
            in_episode = False
    return episodes / len(stream) * 100


def revision_immediate_correction_rate(group: Group, config: dict) -> float:
    """Fraction of insertion keystrokes whose very next keystroke is a
    deletion -- a proxy for "typed something, immediately regretted it".
    """
    stream = _keydown_stream(group, config)
    inserts = 0
    corrected = 0
    for a, b in zip(stream, stream[1:]):
        if _is_insertion(a, config):
            inserts += 1
            if _is_deletion(b, config):
                corrected += 1
    return safe_ratio(corrected, inserts)


# --------------------------------------------------------------------------
# Burst
# --------------------------------------------------------------------------

def _bursts(group: Group, config: dict) -> list[int]:
    """Length (in keystrokes) of every burst, where a burst is a maximal run
    of keydowns with gap < gap_threshold_ms between consecutive keydowns.
    """
    pairs = _pairs(group, config)
    ts = _keydown_timestamps(pairs)
    if not ts:
        return []
    threshold = config["tier1"]["burst"]["gap_threshold_ms"]
    lengths = [1]
    for a, b in zip(ts, ts[1:]):
        if b - a < threshold:
            lengths[-1] += 1
        else:
            lengths.append(1)
    return lengths


def burst_bursts_per_100_keys(group: Group, config: dict) -> float:
    stream = _keydown_stream(group, config)
    if not stream:
        return math.nan
    bursts = _bursts(group, config)
    return len(bursts) / len(stream) * 100


def burst_relative_burst_length(group: Group, config: dict) -> float:
    stream = _keydown_stream(group, config)
    bursts = _bursts(group, config)
    if not stream or not bursts:
        return math.nan
    return (sum(bursts) / len(bursts)) / len(stream)


# --------------------------------------------------------------------------
# Mouse / cursor
# --------------------------------------------------------------------------

def mouse_moves_per_100_keys(group: Group, config: dict) -> float:
    stream = _keydown_stream(group, config)
    if not stream:
        return math.nan
    return len(group.mouse_events) / len(stream) * 100


def mouse_path_length(group: Group, config: dict) -> float:
    """Total Euclidean path length traced by the mouse, normalized to
    "pixels of movement per N keystrokes" (N = tier1.mouse.path_length_per_keys
    in config.yaml). This controls for response length: a longer answer
    naturally has more keystrokes AND more time for mouse movement, so raw
    path length would mostly measure response length rather than mousing
    behavior.
    """
    events = sorted(group.mouse_events, key=lambda e: e.timestamp)
    if len(events) < 2:
        return math.nan
    total = 0.0
    for a, b in zip(events, events[1:]):
        dx = b.data.get("x", 0) - a.data.get("x", 0)
        dy = b.data.get("y", 0) - a.data.get("y", 0)
        total += math.hypot(dx, dy)
    stream = _keydown_stream(group, config)
    if not stream:
        return math.nan
    per_keys = config["tier1"]["mouse"]["path_length_per_keys"]
    return total / len(stream) * per_keys


def cursor_jump_rate(group: Group, config: dict) -> float:
    """Fraction of consecutive cursor-event transitions that are NOT linear.

    A transition is linear if it advances by exactly +1 ch on the same line,
    or moves to the start (ch=0) of the very next line. Anything else --
    backward move, jump to a non-adjacent line, multi-char skip on the same
    line -- counts as a jump.
    """
    events = sorted(group.cursor_events, key=lambda e: e.timestamp)
    if len(events) < 2:
        return math.nan
    jumps = 0
    transitions = 0
    for a, b in zip(events, events[1:]):
        a_line, a_ch = a.data.get("line"), a.data.get("ch")
        b_line, b_ch = b.data.get("line"), b.data.get("ch")
        if None in (a_line, a_ch, b_line, b_ch):
            continue
        transitions += 1
        is_linear = (b_line == a_line and b_ch == a_ch + 1) or (b_line == a_line + 1 and b_ch == 0)
        if not is_linear:
            jumps += 1
    return safe_ratio(jumps, transitions)


FEATURES: dict[str, callable] = {
    "kit_cv": kit_cv,
    "kit_p90_p50": kit_p90_p50,
    "kit_p50_p10": kit_p50_p10,
    "kit_long_pause_fraction": kit_long_pause_fraction,
    "kit_pause_time_fraction": kit_pause_time_fraction,
    "kit_pause_entropy": kit_pause_entropy,
    "kht_cv": kht_cv,
    "kht_p90_p50": kht_p90_p50,
    "kht_p50_p10": kht_p50_p10,
    "rkdt_cv": rkdt_cv,
    "revision_delete_ratio": revision_delete_ratio,
    "revision_deletion_episodes_per_100_keys": revision_deletion_episodes_per_100_keys,
    "revision_immediate_correction_rate": revision_immediate_correction_rate,
    "burst_bursts_per_100_keys": burst_bursts_per_100_keys,
    "burst_relative_burst_length": burst_relative_burst_length,
    "mouse_moves_per_100_keys": mouse_moves_per_100_keys,
    "mouse_path_length": mouse_path_length,
    "cursor_jump_rate": cursor_jump_rate,
}
