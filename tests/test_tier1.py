"""Unit tests for src/features/tier1.py against hand-computed expected values.

None of these numbers come from real data -- each is derived by hand from
the synthetic event sequence constructed in the test itself.
"""
from __future__ import annotations

import math

import pytest

from src.features import tier1
from tests.fixtures import cursor_event, key_event, load_test_config, make_group, mouse_event

CONFIG = load_test_config()


# ---------------------------------------------------------------------------
# Inter-key interval timing
# ---------------------------------------------------------------------------
# Keydowns at t = 0, 100, 200, 300, 800 -> intervals = [100, 100, 100, 500]
# median = 100, mean = 200, sample std (ddof=1) = 200 -> cv = 1.0
# p50 (numpy linear interp on [100,100,100,500]) = 100
# p90 = 380 -> p90/p50 = 3.8
# p10 = 100 -> p50/p10 = 1.0
# long_pause_fraction: threshold = 2*median = 200; only 500 exceeds -> 1/4 = 0.25
# pause_time_fraction: pause_time = 500; total_time = 800-0 = 800 -> 0.625

def _iki_group():
    ts_list = [0, 100, 200, 300, 800]
    events = [key_event(ts=t, key="a", code="KeyA", phase="keydown") for t in ts_list]
    return make_group(key_events=events)


def test_kit_cv():
    assert tier1.kit_cv(_iki_group(), CONFIG) == pytest.approx(1.0)


def test_kit_p90_p50():
    assert tier1.kit_p90_p50(_iki_group(), CONFIG) == pytest.approx(3.8)


def test_kit_p50_p10():
    assert tier1.kit_p50_p10(_iki_group(), CONFIG) == pytest.approx(1.0)


def test_kit_long_pause_fraction():
    assert tier1.kit_long_pause_fraction(_iki_group(), CONFIG) == pytest.approx(0.25)


def test_kit_pause_time_fraction():
    assert tier1.kit_pause_time_fraction(_iki_group(), CONFIG) == pytest.approx(0.625)


def test_kit_pause_entropy_zero_when_uniform():
    ts_list = [0, 100, 200, 300, 400]  # all intervals identical -> zero entropy
    events = [key_event(ts=t, key="a", code="KeyA") for t in ts_list]
    assert tier1.kit_pause_entropy(make_group(key_events=events), CONFIG) == pytest.approx(0.0)


def test_kit_functions_nan_on_too_few_events():
    g = make_group(key_events=[key_event(ts=0, key="a", code="KeyA")])
    assert math.isnan(tier1.kit_cv(g, CONFIG))
    assert math.isnan(tier1.kit_p90_p50(g, CONFIG))


# ---------------------------------------------------------------------------
# Key hold time (KHT)
# ---------------------------------------------------------------------------
# holds = [50, 50, 200] (three keydown/keyup pairs, distinct codes)
# mean = 100, sample std (ddof=1) = sqrt(((-50)^2+(-50)^2+100^2)/2) = sqrt(7500) = 86.6025
# cv = 86.6025/100 = 0.866025
# sorted holds = [50, 50, 200]; p50 = 50; p90 = 50+0.8*(200-50) = 170 -> p90/p50 = 3.4
# p10 = 50 -> p50/p10 = 1.0

def _kht_group():
    events = [
        key_event(ts=0, key="a", code="KeyA", phase="keydown"),
        key_event(ts=50, key="a", code="KeyA", phase="keyup"),
        key_event(ts=500, key="b", code="KeyB", phase="keydown"),
        key_event(ts=550, key="b", code="KeyB", phase="keyup"),
        key_event(ts=1000, key="c", code="KeyC", phase="keydown"),
        key_event(ts=1200, key="c", code="KeyC", phase="keyup"),
    ]
    return make_group(key_events=events)


def test_kht_cv():
    assert tier1.kht_cv(_kht_group(), CONFIG) == pytest.approx(0.8660254, rel=1e-5)


def test_kht_p90_p50():
    assert tier1.kht_p90_p50(_kht_group(), CONFIG) == pytest.approx(3.4)


def test_kht_p50_p10():
    assert tier1.kht_p50_p10(_kht_group(), CONFIG) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Release-to-next-keydown time (RKDT)
# ---------------------------------------------------------------------------
# keyupA=50, keydownB=200 -> gap 150; keyupB=250, keydownC=500 -> gap 250
# mean=200, sample std = sqrt(((-50)^2+50^2)/1) = sqrt(5000) = 70.7107
# cv = 70.7107/200 = 0.353553

def test_rkdt_cv():
    events = [
        key_event(ts=0, key="a", code="KeyA", phase="keydown"),
        key_event(ts=50, key="a", code="KeyA", phase="keyup"),
        key_event(ts=200, key="b", code="KeyB", phase="keydown"),
        key_event(ts=250, key="b", code="KeyB", phase="keyup"),
        key_event(ts=500, key="c", code="KeyC", phase="keydown"),
        key_event(ts=550, key="c", code="KeyC", phase="keyup"),
    ]
    g = make_group(key_events=events)
    assert tier1.rkdt_cv(g, CONFIG) == pytest.approx(0.3535534, rel=1e-5)


# ---------------------------------------------------------------------------
# Repeats are excluded (config default exclude_repeats: true)
# ---------------------------------------------------------------------------

def test_repeat_keydowns_excluded_from_iki():
    events = [
        key_event(ts=0, key="a", code="KeyA", phase="keydown"),
        key_event(ts=10, key="a", code="KeyA", phase="keydown", repeat=True),
        key_event(ts=20, key="a", code="KeyA", phase="keydown", repeat=True),
        key_event(ts=200, key="b", code="KeyB", phase="keydown"),
    ]
    g = make_group(key_events=events)
    # only the two non-repeat keydowns (t=0, t=200) should count -> one interval of 200
    intervals = tier1._inter_key_intervals(tier1._pairs(g, CONFIG))
    assert intervals == [200.0]


# ---------------------------------------------------------------------------
# Revision features
# ---------------------------------------------------------------------------
# stream: a(ins), b(ins), Backspace(del), c(ins) -> deletes=1, inserts=3 -> ratio 0.25

def test_revision_delete_ratio():
    events = [
        key_event(ts=0, key="a"),
        key_event(ts=10, key="b"),
        key_event(ts=20, key="Backspace"),
        key_event(ts=30, key="c"),
    ]
    g = make_group(key_events=events)
    assert tier1.revision_delete_ratio(g, CONFIG) == pytest.approx(0.25)


# stream: a, Backspace, Backspace, b, Backspace, c, d (7 keys)
# episodes: [Backspace,Backspace] after a = 1 episode; [Backspace] after b = 1 episode -> 2 episodes
# 2/7*100 = 28.5714...

def test_revision_deletion_episodes_per_100_keys():
    events = [
        key_event(ts=0, key="a"),
        key_event(ts=10, key="Backspace"),
        key_event(ts=20, key="Backspace"),
        key_event(ts=30, key="b"),
        key_event(ts=40, key="Backspace"),
        key_event(ts=50, key="c"),
        key_event(ts=60, key="d"),
    ]
    g = make_group(key_events=events)
    assert tier1.revision_deletion_episodes_per_100_keys(g, CONFIG) == pytest.approx(200 / 7)


# stream: a, Backspace, b, c, Backspace
# inserts = a, b, c (3); corrected = a->Backspace, c->Backspace (2) -> rate 2/3

def test_revision_immediate_correction_rate():
    events = [
        key_event(ts=0, key="a"),
        key_event(ts=10, key="Backspace"),
        key_event(ts=20, key="b"),
        key_event(ts=30, key="c"),
        key_event(ts=40, key="Backspace"),
    ]
    g = make_group(key_events=events)
    assert tier1.revision_immediate_correction_rate(g, CONFIG) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Burst features
# ---------------------------------------------------------------------------
# keydowns at t = 0, 200, 400, 2000, 2100 (gap_threshold_ms = 1000)
# gaps: 200, 200, 1600, 100 -> bursts of length [3, 2]
# bursts_per_100_keys = 2/5*100 = 40
# relative_burst_length = mean([3,2]) / 5 = 2.5/5 = 0.5

def _burst_group():
    ts_list = [0, 200, 400, 2000, 2100]
    events = [key_event(ts=t, key="a", code="KeyA") for t in ts_list]
    return make_group(key_events=events)


def test_burst_bursts_per_100_keys():
    assert tier1.burst_bursts_per_100_keys(_burst_group(), CONFIG) == pytest.approx(40.0)


def test_burst_relative_burst_length():
    assert tier1.burst_relative_burst_length(_burst_group(), CONFIG) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Mouse features
# ---------------------------------------------------------------------------
# 5 keydowns (reuse burst timestamps), 3 mouse events at (0,0), (3,4), (6,8)
# distances: 5, 5 -> total 10; path_length_per_keys=100 -> 10/5*100 = 200
# mouse_moves_per_100_keys = 3/5*100 = 60

def test_mouse_moves_per_100_keys():
    ts_list = [0, 200, 400, 2000, 2100]
    key_events = [key_event(ts=t, key="a", code="KeyA") for t in ts_list]
    mouse_events = [mouse_event(ts=t, x=x, y=y) for t, (x, y) in zip([0, 50, 100], [(0, 0), (3, 4), (6, 8)])]
    g = make_group(key_events=key_events, mouse_events=mouse_events)
    assert tier1.mouse_moves_per_100_keys(g, CONFIG) == pytest.approx(60.0)


def test_mouse_path_length():
    ts_list = [0, 200, 400, 2000, 2100]
    key_events = [key_event(ts=t, key="a", code="KeyA") for t in ts_list]
    mouse_events = [mouse_event(ts=t, x=x, y=y) for t, (x, y) in zip([0, 50, 100], [(0, 0), (3, 4), (6, 8)])]
    g = make_group(key_events=key_events, mouse_events=mouse_events)
    assert tier1.mouse_path_length(g, CONFIG) == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# cursor_jump_rate
# ---------------------------------------------------------------------------
# transitions:
#   (0,0)->(0,1): linear (same line, +1 ch)
#   (0,1)->(1,0): linear (start of next line)
#   (1,0)->(1,5): jump (multi-char skip, same line)
#   (1,5)->(0,2): jump (backward line)
# 2 jumps / 4 transitions = 0.5

def test_cursor_jump_rate_hand_built_sequence():
    events = [
        cursor_event(ts=0, line=0, ch=0),
        cursor_event(ts=10, line=0, ch=1),
        cursor_event(ts=20, line=1, ch=0),
        cursor_event(ts=30, line=1, ch=5),
        cursor_event(ts=40, line=0, ch=2),
    ]
    g = make_group(cursor_events=events)
    assert tier1.cursor_jump_rate(g, CONFIG) == pytest.approx(0.5)


def test_cursor_jump_rate_all_linear_is_zero():
    events = [
        cursor_event(ts=0, line=0, ch=0),
        cursor_event(ts=10, line=0, ch=1),
        cursor_event(ts=20, line=0, ch=2),
    ]
    g = make_group(cursor_events=events)
    assert tier1.cursor_jump_rate(g, CONFIG) == pytest.approx(0.0)


def test_cursor_jump_rate_nan_with_fewer_than_two_events():
    g = make_group(cursor_events=[cursor_event(ts=0, line=0, ch=0)])
    assert math.isnan(tier1.cursor_jump_rate(g, CONFIG))
