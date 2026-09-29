"""Per-key KHT and per-bigram KIT/RUKD, adapted from pipeline_v2/step1_extract.

Pure shared calculations over normalized Event objects; no dataset paths.
RUKD follows the reference's most recent matched release, not necessarily
the release of the immediately preceding keydown. Durations are milliseconds.
"""
from __future__ import annotations

from collections import defaultdict, deque
from itertools import product
import string

import numpy as np
import pandas as pd

LETTERS = tuple(string.ascii_lowercase) + (" ",)
KEYS = LETTERS + ("backspace", "shift")
DEFAULT_BIGRAMS = tuple(f"{a}->{b}" for a, b in product(LETTERS, repeat=2))
STATS = ("mean", "std", "cv", "range", "max", "median")


def validate_bigrams(bigrams):
    bigrams = tuple(bigrams)
    if not bigrams or len(set(bigrams)) != len(bigrams) or set(bigrams) - set(DEFAULT_BIGRAMS):
        raise ValueError("Bigrams must be a nonempty unique list such as ['a->b', ' ->a']")
    return bigrams


def summarize(values):
    """Reference filters: inclusive 50..5000 ms, then Q1/Q3 +/- 2 IQR.

Population std (ddof=0). Unlike the reference, no surviving samples => NaN.
"""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr >= 50.) & (arr <= 5000.)]
    if arr.size:
        q1, q3 = np.percentile(arr, [25, 75])
        arr = arr[(arr >= q1 - 2 * (q3 - q1)) & (arr <= q3 + 2 * (q3 - q1))]
    if not arr.size:
        return dict.fromkeys(STATS, np.nan)
    mean, std = float(arr.mean()), float(arr.std())
    return dict(mean=mean, std=std, cv=std / mean, range=float(np.ptp(arr)),
                max=float(arr.max()), median=float(np.median(arr)))


def feature_name(kind, token, stat):
    return f"{kind}_{token.replace(' ', 'space')}_{stat}"


def timing_catalog(bigrams=DEFAULT_BIGRAMS):
    rows = []
    for kind, tokens in (("kht", KEYS), ("kit", bigrams), ("rukd", bigrams)):
        for token in tokens:
            for stat in STATS:
                rows.append(dict(feature=feature_name(kind, token, stat), family=kind,
                                 key_or_bigram=token, statistic=stat,
                                 unit="dimensionless" if stat == "cv" else "ms"))
    return pd.DataFrame(rows)


def collect_timings(events, bigrams=DEFAULT_BIGRAMS):
    """Match each keyup once, FIFO by physical code; ignore repeated keydowns."""
    allowed = set(bigrams)
    pending = defaultdict(deque)
    values = defaultdict(list)
    previous_down = previous_up = None
    for event in sorted(events, key=lambda e: e.timestamp):
        if event.event_type != "key":
            continue
        data = event.data
        key = str(data.get("key", "")).lower()
        phase = data.get("key_event_phase")
        code = data.get("code") or key
        ts = event.timestamp
        if phase == "keydown":
            if key not in KEYS or data.get("repeat"):
                continue
            pending[code].append((key, ts))
            if previous_down:
                old_key, old_ts = previous_down
                bigram = f"{old_key}->{key}"
                if bigram in allowed:
                    values[("kit", bigram)].append(ts - old_ts)
            if previous_up:
                old_key, old_ts = previous_up
                bigram = f"{old_key}->{key}"
                if bigram in allowed:
                    values[("rukd", bigram)].append(ts - old_ts)
            previous_down = key, ts
        elif phase == "keyup" and pending[code]:
            down_key, down_ts = pending[code].popleft()
            if ts >= down_ts:
                values[("kht", down_key)].append(ts - down_ts)
                if down_key in LETTERS:
                    previous_up = down_key, ts
    return values


def timing_features(events, bigrams=DEFAULT_BIGRAMS):
    """Sparse result; callers reindex to the fixed catalog to retain empty columns."""
    return {feature_name(kind, token, stat): value
            for (kind, token), values in collect_timings(events, bigrams).items()
            for stat, value in summarize(values).items()}
