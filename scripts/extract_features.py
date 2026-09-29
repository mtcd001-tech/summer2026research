"""Extract shared registered features for one or more cohorts in a single command."""
from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset_features import extract_dataset, feature_catalog, extract_timing_dataset, combine_feature_sets
from src.features.vietnamese_timing import DEFAULT_BIGRAMS, validate_bigrams, timing_catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", type=Path, default=[ROOT / "dataCs1", ROOT / "dataCs2"])
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--features", nargs="+")
    parser.add_argument("--list-features", action="store_true")
    parser.add_argument("--feature-set", choices=["both", "existing", "timing"], default="both")
    parser.add_argument("--bigrams", type=Path, help="Optional JSON list of fixed bigrams, e.g. ['a->b']; reuse the same file for all cohorts")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.list_features:
        if args.feature_set in ("both", "existing"):
            print(feature_catalog()[["feature", "tier", "r_t"]].to_string(index=False))
        if args.feature_set in ("both", "timing"):
            print(timing_catalog(validate_bigrams(json.loads(args.bigrams.read_text())) if args.bigrams else DEFAULT_BIGRAMS).to_string(index=False))
        return 0
    if args.features and args.feature_set != "existing":
        parser.error("--features selects registered features; use --feature-set existing")
    bigrams = validate_bigrams(json.loads(args.bigrams.read_text(encoding="utf-8"))) if args.bigrams else DEFAULT_BIGRAMS
    if len({path.name for path in args.datasets}) != len(args.datasets):
        parser.error("Dataset names must be unique")
    for dataset in args.datasets:
        print(f"Extracting {dataset} ...", flush=True)
        if args.feature_set in ("both", "existing"):
            result = extract_dataset(dataset, args.config, set(args.features) if args.features else None)
            print(f"  {result['groups']} response groups, {result['questions']} questions, "
                  f"{result['feature_count']} registered features", flush=True)
        if args.feature_set in ("both", "timing"):
            result = extract_timing_dataset(dataset, bigrams)
            print(f"  {result['groups']} response groups, {result['features']} timing features", flush=True)
        if args.feature_set == "both":
            combine_feature_sets(dataset)
        print(f"  Saved to {dataset / 'processed'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
