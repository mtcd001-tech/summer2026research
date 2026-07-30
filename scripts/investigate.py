"""Run AFTER the dataset lands in data/raw/. Answers, empirically, the three
open questions the extraction pipeline deliberately does NOT guess at:

  1. What does keystroke `version` (1 vs 2) correspond to among the response
     fields (e.g. chatgptAnswer/retype for sessions 2/5)?
  2. Does replaying keydown events (src.streams.reconstruct_text) reproduce
     the stored response text, and how often?
  3. What fraction of code tokens are boilerplate (identical prototype code
     every participant is given for a question), and should tier2_code_*
     features strip it before computing identifier_diversity etc.?

This script makes NO assumptions the extraction pipeline depends on -- it is
purely diagnostic. Nothing here should be used to hardcode a mapping in
src/loader.py until its output has actually been reviewed against real data.

Usage:
    python scripts/investigate.py --data-root data/raw
"""
from __future__ import annotations

import argparse
import difflib
import statistics
import sys
import tokenize
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codetext import tokenize_code  # noqa: E402
from src.loader import iter_user_sessions  # noqa: E402
from src.streams import build_groups, reconstruct_text  # noqa: E402


# ---------------------------------------------------------------------------
# Q1: version <-> response-field mapping
# ---------------------------------------------------------------------------
# Rationale: `chatgptAnswer` is plausibly text shown to the participant
# rather than typed by them, whereas `retype` is plausibly typed keystroke by
# keystroke. If that's right, the keystroke `version` whose keydown COUNT is
# near zero should line up with whichever field is the "shown, not typed"
# one, and the version with substantial keydown volume, closely tracking the
# stored text's length, should line up with the typed one. This computes
# that correlation directly rather than assuming which is which.

def investigate_version_mapping(data_root: Path) -> None:
    print("=" * 78)
    print("Q1: keystroke `version` <-> response-field mapping")
    print("=" * 78)

    # (session) -> version -> list of (keydown_count, text_len_per_field)
    per_session_points: dict[int, dict[int, list[tuple[int, dict[str, int]]]]] = defaultdict(lambda: defaultdict(list))

    for session_data in iter_user_sessions(data_root):
        groups = build_groups(session_data)
        # collect all response text lengths for this user/session/q_id/r_t,
        # regardless of which version the loader assigned them to, keyed by
        # the raw source_field name so we can test field/version pairings
        # independent of the (possibly wrong) RESPONSE_FIELD_MAP guess.
        text_by_qid_rt: dict[tuple[int, str], dict[str, int]] = defaultdict(dict)
        for resp in session_data.responses:
            text_by_qid_rt[(resp.q_id, resp.r_t)][resp.source_field] = len(resp.text)

        for key, group in groups.items():
            _, session, q_id, r_t, version = key
            keydown_count = sum(
                1 for ev in group.key_events if ev.data.get("key_event_phase") == "keydown" and not ev.data.get("repeat")
            )
            field_lens = text_by_qid_rt.get((q_id, r_t), {})
            per_session_points[session][version].append((keydown_count, field_lens))

    for session in sorted(per_session_points):
        print(f"\n--- session {session} ---")
        for version in sorted(per_session_points[session]):
            points = per_session_points[session][version]
            keydown_counts = [p[0] for p in points]
            all_fields = set()
            for _, field_lens in points:
                all_fields |= set(field_lens)
            print(f"  version={version}: n={len(points)}, mean keydown_count={statistics.mean(keydown_counts):.1f}")
            for field in sorted(all_fields):
                pairs = [(kc, fl[field]) for kc, fl in points if field in fl]
                if len(pairs) < 2:
                    continue
                xs, ys = zip(*pairs)
                try:
                    r = statistics.correlation(xs, ys)
                except statistics.StatisticsError:
                    r = float("nan")
                print(f"    corr(keydown_count, len({field})) = {r:.3f}  (n={len(pairs)})")

    print(
        "\nInterpretation guide: for each session, the field whose length "
        "correlates most strongly with a given version's keydown_count is "
        "the field that version's keystrokes most likely produced. A "
        "version with near-zero keydown_count but a field with substantial "
        "length suggests that field was shown/pasted, not typed."
    )


# ---------------------------------------------------------------------------
# Q2: keystroke-replay reconstruction vs stored text
# ---------------------------------------------------------------------------

def investigate_reconstruction_agreement(data_root: Path) -> None:
    print("\n" + "=" * 78)
    print("Q2: keystroke-replay reconstruction vs stored response text")
    print("=" * 78)

    exact_matches = 0
    total = 0
    ratios: list[float] = []

    for session_data in iter_user_sessions(data_root):
        groups = build_groups(session_data)
        for group in groups.values():
            if group.text is None or not group.key_events:
                continue
            reconstructed = reconstruct_text(group.key_events)
            total += 1
            if reconstructed == group.text:
                exact_matches += 1
            ratio = difflib.SequenceMatcher(a=reconstructed, b=group.text).ratio()
            ratios.append(ratio)

    if total == 0:
        print("No groups had both key events and stored text; nothing to compare.")
        return

    print(f"Groups compared: {total}")
    print(f"Exact match: {exact_matches}/{total} ({exact_matches / total:.1%})")
    print(f"Mean similarity ratio (difflib): {statistics.mean(ratios):.3f}")
    print(f"Median similarity ratio: {statistics.median(ratios):.3f}")
    print(
        "\nIf exact-match rate is high, the reconstruction heuristic in "
        "src/streams.py is validated and stored text remains authoritative "
        "as designed. If it's low, inspect a few low-ratio examples directly "
        "-- the likely culprit is the Enter-key split-point limitation "
        "documented in reconstruct_text()'s docstring."
    )


# ---------------------------------------------------------------------------
# Q3: boilerplate fraction in provided prototypes
# ---------------------------------------------------------------------------
# Heuristic: for a given (session, q_id), an identifier that appears in
# EVERY (or nearly every) participant's code for that question is very
# likely boilerplate from the provided prototype, not something each
# participant independently chose to name. This can't distinguish "everyone
# happened to use the same common name" (e.g. `i`, `result`) from genuine
# prototype boilerplate with certainty, so this reports the fraction under a
# couple of thresholds rather than a single number.

def investigate_boilerplate(data_root: Path, thresholds=(0.9, 1.0)) -> None:
    print("\n" + "=" * 78)
    print("Q3: boilerplate fraction in provided code prototypes")
    print("=" * 78)

    # (session, q_id) -> user -> Counter[identifier]
    idents_by_question: dict[tuple[int, int], dict[str, Counter]] = defaultdict(dict)

    for session_data in iter_user_sessions(data_root):
        groups = build_groups(session_data)
        for key, group in groups.items():
            _, session, q_id, r_t, _version = key
            if r_t != "code" or not group.text:
                continue
            tokens = tokenize_code(group.text)
            if tokens is None:
                continue
            names = Counter(t.string for t in tokens if t.type == tokenize.NAME)
            idents_by_question[(session, q_id)].setdefault(group.user, Counter())
            idents_by_question[(session, q_id)][group.user].update(names)

    for (session, q_id), by_user in sorted(idents_by_question.items()):
        n_users = len(by_user)
        if n_users < 2:
            continue
        presence = Counter()
        total_occurrences = 0
        occurrences_by_ident = Counter()
        for counter in by_user.values():
            for ident, count in counter.items():
                presence[ident] += 1
                occurrences_by_ident[ident] += count
                total_occurrences += count

        print(f"\n--- session {session}, q_id {q_id} ({n_users} users) ---")
        for threshold in thresholds:
            boilerplate_idents = {ident for ident, n in presence.items() if n / n_users >= threshold}
            boilerplate_occurrences = sum(occurrences_by_ident[i] for i in boilerplate_idents)
            frac = boilerplate_occurrences / total_occurrences if total_occurrences else float("nan")
            print(
                f"  present in >= {threshold:.0%} of users: {len(boilerplate_idents)} identifiers, "
                f"{boilerplate_occurrences}/{total_occurrences} occurrences ({frac:.1%})"
            )

    print(
        "\nIf the >=90-100% band captures a large, consistent fraction of "
        "occurrences across questions, tier2_code_* features (especially "
        "identifier_diversity) should strip those identifiers -- or the "
        "prototype's exact token span, if it can be recovered from the "
        "question text -- before computing stylometric features, otherwise "
        "diversity partly measures shared boilerplate rather than authored code."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    args = parser.parse_args(argv)

    investigate_version_mapping(args.data_root)
    investigate_reconstruction_agreement(args.data_root)
    investigate_boilerplate(args.data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
