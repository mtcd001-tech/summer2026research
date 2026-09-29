"""Normalize both cohorts in place, preserving file bytes and recording moves.

Default is a dry run; use --apply to rename. Refuses collisions before any move.
Only recognized scenario directories and JSON filenames are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scenarios import SCENARIOS, FOLDER_ALIASES


def plan_moves(root: Path) -> list[tuple[Path, Path]]:
    root = root.resolve(strict=True)
    moves = []
    destinations = set()
    for user in sorted(root.iterdir()):
        if not user.is_dir():
            continue
        for session, aliases in FOLDER_ALIASES.items():
            for alias in aliases:
                directory = user / alias
                if not directory.is_dir():
                    continue
                for source in sorted(directory.iterdir()):
                    if not source.is_file():
                        raise ValueError(f"Unexpected nested directory: {source}")
                    name = source.name
                    match = re.fullmatch(r"s([1-5])_(keystrokes|responses|user_info)(?: \(\d+\))?\.json", name)
                    if match:
                        if int(match[1]) != session:
                            raise ValueError(f"Session/folder mismatch: {source}")
                        name = f"s{session}_{match[2]}.json"
                    target = user / SCENARIOS[session] / name
                    if not source.resolve().is_relative_to(root) or not target.resolve().is_relative_to(root):
                        raise ValueError(f"Path escapes dataset: {source}")
                    if target in destinations or (target.exists() and source != target):
                        raise ValueError(f"Destination collision: {target}")
                    destinations.add(target)
                    if source != target:
                        moves.append((source, target))
    return moves


def normalize(root: Path, apply: bool = False) -> int:
    root = root.resolve(strict=True)
    moves = plan_moves(root)
    print(f"{root}: {len(moves)} file renames {'(apply)' if apply else '(dry run)'}")
    if not apply or not moves:
        return len(moves)
    manifest = root.parent / "normalization_manifest.json"
    if manifest.exists():
        raise ValueError(f"Preserve existing migration history before another migration: {manifest}")
    records = [{"source": str(a.relative_to(root)), "target": str(b.relative_to(root)),
                "sha256": hashlib.sha256(a.read_bytes()).hexdigest()} for a, b in moves]
    manifest.write_text(json.dumps({"root": str(root), "moves": records}, indent=2), encoding="utf-8")
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    for directory in {source.parent for source, _ in moves}:
        if not any(directory.iterdir()):
            directory.rmdir()
    return len(moves)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, default=[ROOT / "dataCs1/raw", ROOT / "dataCs2/raw"])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    # Preflight every cohort before changing any of them.
    for root in args.roots:
        plan_moves(root)
    for root in args.roots:
        normalize(root, args.apply)


if __name__ == "__main__":
    main()
