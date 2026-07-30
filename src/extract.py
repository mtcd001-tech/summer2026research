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
from pathlib import Path
from typing import Iterable, Iterator, Optional

import yaml

from src.features import (
    TIER1_FEATURES,
    TIER2_CODE_FEATURES,
    TIER2_EXPLANATION_FEATURES,
    TIER3_FEATURES,
    all_feature_names,
)
from src.loader import DatasetError, discover_sessions, iter_user_sessions
from src.streams import Group, build_groups

logger = logging.getLogger(__name__)

CSV_FIELDS = ["user", "session", "q_id", "r_t", "version", "tier", "feature", "value"]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _group_rows(group: Group, config: dict, enabled: Optional[set[str]]) -> list[dict]:
    rows: list[dict] = []

    def _emit(tier: str, feature_name: str, func) -> None:
        if enabled is not None and feature_name not in enabled:
            return
        try:
            value = func(group, config)
        except Exception:
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
    elif group.r_t == "explanation":
        for name, func in TIER2_EXPLANATION_FEATURES.items():
            _emit("tier2", f"tier2_explanation_{name}", func)
    else:
        logger.warning(
            "user=%s session=%s q_id=%s: unrecognized r_t=%r; only tier1 features "
            "computed for this group (tier2/tier3 require a known r_t)",
            group.user, group.session, group.q_id, group.r_t,
        )

    return rows


def extract_all(data_root: Path, config: dict, enabled: Optional[set[str]] = None) -> Iterator[dict]:
    """Yield one row dict per (group, feature) across the whole dataset."""
    for session_data in iter_user_sessions(data_root):
        groups = build_groups(session_data)
        for group in groups.values():
            yield from _group_rows(group, config, enabled)


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
    parser.add_argument("--features", nargs="*", default=None, help="Only compute these full feature names")
    parser.add_argument("--exclude-features", nargs="*", default=None, help="Compute all features except these")
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

    config = load_config(args.config)
    enabled = _resolve_enabled(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in extract_all(args.data_root, config, enabled):
            writer.writerow(row)
            rows_written += 1

    logger.info("wrote %d rows to %s", rows_written, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
