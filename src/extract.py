"""CLI entry point: walk the dataset, extract every enabled feature for
every (user, session, q_id, r_t, version) group, and write a long-format CSV.

Output is written RAW: no scaling, no NaN imputation. Scaling belongs in the
training script inside each CV fold; XGBoost handles NaN natively while SVM
does not, so each downstream model should decide for itself how to treat it.

Usage:
    python -m src.extract --data-root data/raw --out features.csv
    python -m src.extract --list-features
    python -m src.extract --data-root data/raw --out features.csv \\
        --exclude-features tier1_code_mouse_path_length
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

import yaml

from src.features import (
    TIER1_FEATURES,
    TIER2_CODE_FEATURES,
    TIER2_EXPLANATION_FEATURES,
    TIER3_FEATURES,
    all_feature_names,
)
from src.features.tier1 import keydown_count
from src.loader import DatasetError, Event, discover_sessions, iter_user_sessions
from src.streams import Group, build_groups

logger = logging.getLogger(__name__)

CSV_FIELDS = ["user", "session", "q_id", "r_t", "version", "tier", "feature", "value"]


def _is_human_typed(session: int, version: int) -> bool:
    """Whether a (session, version) group is human-generated text, per
    Ashley (built the collection platform): version 1 is always the ChatGPT
    copy-paste box (LLM output pasted in, no human typing by construction);
    version 2 is the box the human actually typed/transcribed/paraphrased
    into. Session 1 (bona fide) only ever has version 1, and it IS
    human-generated there -- there is no AI box in that condition. See
    RESPONSE_FIELD_MAP's docstring in src/loader.py for the verification
    against real data.
    """
    if session == 1:
        return version == 1
    return version == 2


class ConfigError(RuntimeError):
    """Raised for a missing/unparseable/incomplete config.yaml.

    Meant to be caught at the CLI boundary (main()) and reported as a short
    actionable message, not a stack trace -- same treatment DatasetError
    already gets for a missing dataset.
    """


def load_config(config_path: Path) -> dict:
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
    except OSError as exc:
        raise ConfigError(f"could not read config file '{config_path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file '{config_path}' is not valid YAML: {exc}") from exc

    if not isinstance(config, dict) or not config:
        raise ConfigError(
            f"config file '{config_path}' is empty or not a mapping of settings. "
            f"See config.yaml for the expected structure."
        )
    return config


class _ConfigKeyRecorder(dict):
    """Stand-in for the config dict that records every key path accessed via
    `[...]` instead of returning a real value.

    Used to derive the set of config keys the registered feature functions
    actually require by *running* them against a synthetic group, rather
    than hardcoding a key list here that would silently drift out of sync
    with src/features/*.py. Keys read via `.get(key, default)` are treated
    as optional (the function already tolerates their absence) and are not
    recorded.
    """

    def __init__(self, accessed: set[str], prefix: str = "") -> None:
        super().__init__()
        self._accessed = accessed
        self._prefix = prefix

    def __getitem__(self, key):
        path = f"{self._prefix}.{key}" if self._prefix else str(key)
        self._accessed.add(path)
        return _ConfigKeyRecorder(self._accessed, path)

    def get(self, key, default=None):
        return default


def _probe_group(*, r_t: str, text: str, prompt: str) -> Group:
    """A minimal but non-degenerate synthetic group used only to walk every
    feature function once so `_required_config_keys` can observe which
    config keys it touches. Values are not meaningful, just non-empty."""
    key_events = []
    ts = 0
    for i, ch in enumerate("ab"):
        code = f"Key{i}"
        key_events.append(
            Event(
                user="probe", session=1, q_id=1, r_t=r_t, version=1, event_type="key",
                data={"key_event_phase": "keydown", "code": code, "key": ch, "repeat": False, "line": 0, "ch": i + 1},
                timestamp=ts,
            )
        )
        key_events.append(
            Event(
                user="probe", session=1, q_id=1, r_t=r_t, version=1, event_type="key",
                data={"key_event_phase": "keyup", "code": code, "key": ch}, timestamp=ts + 10,
            )
        )
        ts += 100

    g = Group(user="probe", session=1, q_id=1, r_t=r_t, version=1)
    g.key_events = key_events
    g.mouse_events = [
        Event(user="probe", session=1, q_id=1, r_t=r_t, version=1, event_type="mouse", data={"x": 0, "y": 0}, timestamp=0),
        Event(user="probe", session=1, q_id=1, r_t=r_t, version=1, event_type="mouse", data={"x": 1, "y": 1}, timestamp=10),
    ]
    g.cursor_events = [
        Event(user="probe", session=1, q_id=1, r_t=r_t, version=1, event_type="cursor", data={"line": 0, "ch": 0}, timestamp=0),
        Event(user="probe", session=1, q_id=1, r_t=r_t, version=1, event_type="cursor", data={"line": 0, "ch": 1}, timestamp=10),
    ]
    g.text = text
    g.prompt = prompt
    return g


def _required_config_keys() -> set[str]:
    """Derive the set of dotted config key paths (e.g. "tier1.pause.entropy_bins")
    that the registered feature functions require, by actually calling them
    against synthetic groups and a recording config stand-in."""
    accessed: set[str] = set()
    recorder = _ConfigKeyRecorder(accessed)

    code_group = _probe_group(r_t="code", text="x = 1\n", prompt="write x")
    explanation_group = _probe_group(r_t="explanation", text="This is a test. Another sentence.", prompt="explain it")

    all_funcs = (
        list(TIER1_FEATURES.values())
        + list(TIER2_CODE_FEATURES.values())
        + list(TIER3_FEATURES.values())
        + [keydown_count]  # not a registered feature, but reads config the same way
    )
    for func in all_funcs:
        try:
            func(code_group, recorder)
        except Exception:
            pass

    for func in list(TIER1_FEATURES.values()) + list(TIER2_EXPLANATION_FEATURES.values()) + [keydown_count]:
        try:
            func(explanation_group, recorder)
        except Exception:
            pass

    # Not touched by any probed function above -- extract_all()/main() read
    # it directly as a secondary sanity-check threshold (see extract_all's
    # docstring), so it has to be added explicitly or a config missing this
    # key would crash with a raw KeyError instead of the clean "missing
    # required key(s)" error every other config key gets.
    accessed.add("tier1.filtering.sanity_check_min_keydown_count")

    return accessed


def _missing_config_keys(config: dict) -> list[str]:
    missing = []
    for dotted in sorted(_required_config_keys()):
        node = config
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                missing.append(dotted)
                break
            node = node[part]
    return missing


def _group_rows(
    group: Group,
    config: dict,
    enabled: Optional[set[str]],
    feature_errors: Counter,
    feature_totals: Counter,
) -> list[dict]:
    rows: list[dict] = []

    if group.r_t not in ("code", "explanation"):
        # tier1 feature names are prefixed with r_t (tier1_{r_t}_kit_cv), so
        # an unrecognized r_t would otherwise mint names like
        # "tier1_garbage_kit_cv" that all_feature_names() never produces and
        # that --features/--exclude-features can't reason about consistently.
        # Skipping the whole group (rather than emitting tier1 only) keeps
        # every emitted feature name inside the documented schema.
        logger.warning(
            "user=%s session=%s q_id=%s: unrecognized r_t=%r; skipping this "
            "group entirely (no tier1/tier2/tier3 features computed)",
            group.user, group.session, group.q_id, group.r_t,
        )
        return rows

    def _emit(tier: str, feature_name: str, func) -> None:
        if enabled is not None and feature_name not in enabled:
            return
        feature_totals[feature_name] += 1
        try:
            value = func(group, config)
        except Exception:
            feature_errors[feature_name] += 1
            logger.exception(
                "user=%s session=%s q_id=%s r_t=%s version=%s: feature %s raised an "
                "exception, recording NaN instead of failing the whole run",
                group.user, group.session, group.q_id, group.r_t, group.version, feature_name,
            )
            value = math.nan
        rows.append(
            {
                "user": group.user,
                "session": group.session,
                "q_id": group.q_id,
                "r_t": group.r_t,
                "version": group.version,
                "tier": tier,
                "feature": feature_name,
                "value": value,
            }
        )

    for name, func in TIER1_FEATURES.items():
        _emit("tier1", f"tier1_{group.r_t}_{name}", func)

    if group.r_t == "code":
        for name, func in TIER2_CODE_FEATURES.items():
            _emit("tier2", f"tier2_code_{name}", func)
        for name, func in TIER3_FEATURES.items():
            _emit("tier3", f"tier3_llm_{name}", func)
    else:  # explanation
        for name, func in TIER2_EXPLANATION_FEATURES.items():
            _emit("tier2", f"tier2_explanation_{name}", func)

    return rows


def extract_all(
    data_root: Path,
    config: dict,
    enabled: Optional[set[str]] = None,
    feature_errors: Optional[Counter] = None,
    feature_totals: Optional[Counter] = None,
    dropped_groups: Optional[Counter] = None,
    short_kept_groups: Optional[Counter] = None,
) -> Iterator[dict]:
    """Yield one row dict per (group, feature) across the whole dataset.

    Groups are kept/dropped by an explicit rule, not a heuristic: a group is
    human-generated text (see _is_human_typed) if it's (session=1,
    version=1) or (session in 2-5, version=2); everything else -- version 1
    for sessions 2-5, the ChatGPT paste box -- is dropped, because it
    contains no human typing by construction, not because it happens to be
    short. `dropped_groups`, if given, is incremented per (session, version)
    so the exclusion is visible/auditable rather than silent.

    config.yaml's tier1.filtering.sanity_check_min_keydown_count is NOT an
    exclusion filter (that's the rule above). It's a secondary sanity check:
    a kept (human-typed) group with fewer keydowns than this is still
    included in the output, just flagged in `short_kept_groups` -- a
    human-typed box with very few keydowns is worth a second look, not proof
    it's not real data.
    """
    if feature_errors is None:
        feature_errors = Counter()
    if feature_totals is None:
        feature_totals = Counter()
    if dropped_groups is None:
        dropped_groups = Counter()
    if short_kept_groups is None:
        short_kept_groups = Counter()

    sanity_min_keydown_count = config["tier1"]["filtering"]["sanity_check_min_keydown_count"]

    total_groups = 0
    kept_groups = 0
    for session_data in iter_user_sessions(data_root):
        groups = build_groups(session_data)
        for group in groups.values():
            total_groups += 1
            if not _is_human_typed(group.session, group.version):
                dropped_groups[(group.session, group.version)] += 1
                continue
            kept_groups += 1
            if keydown_count(group, config) < sanity_min_keydown_count:
                short_kept_groups[(group.session, group.version)] += 1
            yield from _group_rows(group, config, enabled, feature_errors, feature_totals)
        logger.info(
            "user=%s session=%s: processed %d groups (%d kept / %d total so far)",
            session_data.user, session_data.session, len(groups), kept_groups, total_groups,
        )


def _resolve_enabled(args: argparse.Namespace) -> Optional[set[str]]:
    if args.features is not None:
        return set(args.features)
    if args.exclude_features is not None:
        return set(all_feature_names()) - set(args.exclude_features)
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/raw"),
        help="Directory containing one subfolder per user (see data/README.md)",
    )
    parser.add_argument("--out", type=Path, default=Path("features.csv"), help="Output CSV path")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    features_group = parser.add_mutually_exclusive_group()
    features_group.add_argument(
        "--features", nargs="+", default=None, help="Only compute these full feature names"
    )
    features_group.add_argument(
        "--exclude-features", nargs="+", default=None, help="Compute all features except these"
    )
    parser.add_argument(
        "--list-features", action="store_true",
        help="Print every feature name this registry can produce (tier{n}_{r_t}_{name}) and exit",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.list_features:
        for name in sorted(all_feature_names()):
            print(name)
        return 0

    try:
        discover_sessions(args.data_root)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    missing_keys = _missing_config_keys(config)
    if missing_keys:
        print(
            f"error: config file '{args.config}' is missing required key(s): "
            f"{', '.join(missing_keys)}",
            file=sys.stderr,
        )
        return 1

    enabled = _resolve_enabled(args)
    if enabled is not None and not enabled:
        print(
            "error: no features selected -- --features/--exclude-features "
            "resolved to an empty set, which would write a header-only CSV",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    feature_errors: Counter = Counter()
    feature_totals: Counter = Counter()
    dropped_groups: Counter = Counter()
    short_kept_groups: Counter = Counter()
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in extract_all(
            args.data_root, config, enabled, feature_errors, feature_totals, dropped_groups, short_kept_groups
        ):
            writer.writerow(row)
            rows_written += 1

    if dropped_groups:
        total_dropped = sum(dropped_groups.values())
        logger.info(
            "dropped %d group(s) that are not human-typed by construction "
            "(version 1 for sessions 2-5 is the ChatGPT paste box) by (session, version):",
            total_dropped,
        )
        for session, version in sorted(dropped_groups):
            logger.info(
                "  session=%s version=%s: %d dropped",
                session, version, dropped_groups[(session, version)],
            )

    if short_kept_groups:
        sanity_min_keydown_count = config["tier1"]["filtering"]["sanity_check_min_keydown_count"]
        total_short = sum(short_kept_groups.values())
        logger.warning(
            "sanity check: %d kept (human-typed) group(s) had fewer than "
            "sanity_check_min_keydown_count=%d keydowns -- still included in "
            "the output, just flagged, by (session, version):",
            total_short, sanity_min_keydown_count,
        )
        for session, version in sorted(short_kept_groups):
            logger.warning(
                "  session=%s version=%s: %d group(s) below sanity threshold",
                session, version, short_kept_groups[(session, version)],
            )

    for name in sorted(feature_totals):
        errors = feature_errors.get(name, 0)
        if not errors:
            continue
        total = feature_totals[name]
        if errors == total:
            logger.warning(
                "feature %s raised on ALL %d/%d groups -- this almost certainly "
                "indicates a bug in the feature function, not missing data",
                name, errors, total,
            )
        else:
            logger.warning("feature %s raised on %d/%d groups", name, errors, total)

    logger.info("wrote %d rows to %s", rows_written, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
