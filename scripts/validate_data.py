"""Run AFTER the dataset lands in data/raw/.

Verifies folder structure, per-user/per-session file presence, and JSON
schema conformance -- deliberately WITHOUT using src/loader.py's tolerant
fallback parsing, so a problem surfaced here is a real data issue rather
than something the loader quietly worked around.

Usage:
    python scripts/validate_data.py --data-root data/raw
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loader import RESPONSE_FIELD_MAP, SESSIONS  # noqa: E402

REQUIRED_EVENT_KEYS = {"s_n", "r_t", "q_id", "version", "event_type", "data", "timestamp"}
REQUIRED_KEY_DATA_KEYS = {"key", "code", "key_event_phase", "repeat", "line", "ch"}
REQUIRED_MOUSE_DATA_KEYS = {"x", "y"}
REQUIRED_CURSOR_DATA_KEYS = {"line", "ch"}


def _check_keystrokes_file(path: Path) -> list[str]:
    problems = []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: could not parse JSON ({exc})"]

    if "keystrokes" not in raw:
        problems.append(f"{path}: missing top-level 'keystrokes' key")
    if "questions" not in raw:
        problems.append(f"{path}: missing top-level 'questions' key")

    for i, ev in enumerate(raw.get("keystrokes", [])):
        missing = REQUIRED_EVENT_KEYS - set(ev)
        if missing:
            problems.append(f"{path}: event {i} missing keys {sorted(missing)}")
            continue
        data = ev.get("data") or {}
        et = ev.get("event_type")
        if et == "key":
            missing_data = REQUIRED_KEY_DATA_KEYS - set(data)
        elif et == "mouse":
            missing_data = REQUIRED_MOUSE_DATA_KEYS - set(data)
        elif et == "cursor":
            missing_data = REQUIRED_CURSOR_DATA_KEYS - set(data)
        else:
            problems.append(f"{path}: event {i} has unrecognized event_type={et!r}")
            continue
        if missing_data:
            problems.append(f"{path}: event {i} (event_type={et}) missing data keys {sorted(missing_data)}")
    return problems


def _check_responses_file(path: Path) -> list[str]:
    problems = []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: could not parse JSON ({exc})"]

    if "responses" not in raw:
        problems.append(f"{path}: missing top-level 'responses' key")
    if "questions" not in raw:
        problems.append(f"{path}: missing top-level 'questions' key")

    for i, entry in enumerate(raw.get("responses", [])):
        known = set(entry) & set(RESPONSE_FIELD_MAP)
        if not known:
            q_id = entry.get("q_id", entry.get("question"))
            problems.append(
                f"{path}: response entry {i} (q_id={q_id}) has no field matching "
                f"RESPONSE_FIELD_MAP; fields present: {sorted(entry)}"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    args = parser.parse_args(argv)

    data_root = args.data_root
    if not data_root.exists():
        print(f"error: '{data_root}' does not exist. See data/README.md.")
        return 1

    user_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    if not user_dirs:
        print(f"error: '{data_root}' has no user subfolders. See data/README.md.")
        return 1

    coverage: dict[str, dict[int, str]] = defaultdict(dict)
    all_problems: list[str] = []

    for user_dir in user_dirs:
        for session in SESSIONS:
            k_path = user_dir / f"s{session}_keystrokes.json"
            r_path = user_dir / f"s{session}_responses.json"
            k_exists, r_exists = k_path.exists(), r_path.exists()
            if not k_exists and not r_exists:
                coverage[user_dir.name][session] = "absent"
                continue
            if not k_exists:
                coverage[user_dir.name][session] = "missing keystrokes"
                continue
            if not r_exists:
                coverage[user_dir.name][session] = "missing responses"
                continue

            problems = _check_keystrokes_file(k_path) + _check_responses_file(r_path)
            if problems:
                coverage[user_dir.name][session] = f"{len(problems)} issue(s)"
                all_problems.extend(problems)
            else:
                coverage[user_dir.name][session] = "ok"

    print(f"Users found: {len(user_dirs)}")
    print()
    header = "user".ljust(20) + "".join(f"s{s}".rjust(18) for s in SESSIONS)
    print(header)
    for user, sessions in coverage.items():
        row = user.ljust(20)
        for s in SESSIONS:
            row += sessions.get(s, "absent").rjust(18)
        print(row)

    total_expected = len(user_dirs) * len(SESSIONS)
    total_ok = sum(1 for sessions in coverage.values() for v in sessions.values() if v == "ok")
    print()
    print(f"Sessions OK: {total_ok}/{total_expected}")

    if all_problems:
        print()
        print(f"Schema issues ({len(all_problems)}):")
        for p in all_problems[:200]:
            print(f"  - {p}")
        if len(all_problems) > 200:
            print(f"  ... and {len(all_problems) - 200} more")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
