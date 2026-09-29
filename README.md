# Keystroke feature extraction (CS1 and CS2)

## Shared five-scenario analysis

Extract **both feature families** (existing behavioral/text features and
Vietnamese-style per-key/bigram timings) for both cohorts:

```powershell
python scripts/extract_features.py
```

See [feature extraction commands and definitions](scripts/FEATURE_EXTRACTION.md).

CS1 and CS2 now share `raw/<user>/<scenario>/sN_keystrokes.json` and
`sN_responses.json`, with extracted tables in each cohort's
`processed/features.csv`. Scenarios are **Bona fide, Transcription, Mimicry,
Paraphrasing, Deception**. Legacy folder aliases remain readable.

```powershell
pip install -r requirements.txt -r requirements-modeling.txt
python scripts/analysis/scenario_importance.py --extract
```

This generates MI and signed effect-size heatmaps (PNG/PDF) and full plotting
tables for both datasets under `analysis/scenarios/`. See the
[shared analysis guide](scripts/analysis/README.md) for methods, commands,
data normalization, and limitations. Saved-model analysis is separately
available in `scripts/analysis/feature_importance.py`.

The original extraction design notes below describe the initial CS1 setup;
their pre-data status and `data/raw` paths are historical. Use `dataCs1/raw`
or `dataCs2/raw` with the extraction and validation commands.

Extracts a tabular feature set (keystroke-timing, code/text stylometry,
LLM-signature) from keystroke and response logs, for downstream MLP/SVM/
XGBoost classification (not part of this project). This project only builds
and validates the feature table -- no model training/evaluation code lives
here.

## Status

The dataset has **not landed yet**. Everything here is built against the
documented schema and validated with synthetic fixtures (see `tests/`). Once
real data arrives, follow "After the data lands" below.

## Install

```
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('stopwords')"
```

## Run the tests (works right now, no data needed)

```
python -m pytest
```

## Drop the data in

See [data/README.md](data/README.md) for the exact expected folder layout
(`data/raw/<user>/s{n}_keystrokes.json` + `s{n}_responses.json`).

## After the data lands

```
python scripts/validate_data.py --data-root data/raw
python scripts/investigate.py --data-root data/raw
python -m src.extract --data-root data/raw --out features.csv
```

- `validate_data.py` checks folder/file/schema conformance and prints a
  per-user/per-session coverage table.
- `investigate.py` empirically answers three open questions the extraction
  code deliberately does not guess at (see below) -- it makes no assumptions
  the pipeline depends on.
- `src/extract.py` walks the dataset and writes a long-format CSV:
  `user, session, q_id, r_t, version, tier, feature, value`. Values are
  written raw (no scaling, no NaN imputation) -- that's a training-time
  decision, not an extraction-time one.

Run `python -m src.extract --list-features` to see every feature name the
current registry produces, or `--features NAME [NAME ...]` /
`--exclude-features NAME [NAME ...]` to toggle a subset.

## Project layout

```
config.yaml            thresholds (pause multiplier, burst gap, entropy bins, ...)
data/
  README.md            expected on-disk layout for the dataset
src/
  loader.py            dataset discovery, JSON parsing, schema normalization
  streams.py           groups events by (user, session, q_id, r_t, version);
                        includes a validation-only keystroke-replay reconstruction
  codetext.py           shared line classification + indentation normalization
                        + tolerant tokenize wrapper (used by tier2 and tier3)
  statutils.py          shared stats helpers (cv, percentile ratio, entropy)
  extract.py            CLI entry point
  features/
    __init__.py          registry: feature name -> callable
    tier1.py             keystroke timing / revision / burst / mouse / cursor
    tier2.py             code + explanation stylometry
    tier3.py             LLM-signature features
scripts/
  validate_data.py      run after data lands: structure + schema report
  investigate.py        run after data lands: answers the 3 open questions
tests/
  fixtures.py            synthetic event/response/group builders
  test_*.py              unit + integration tests, all synthetic
```

## Feature registry pattern

Every feature function has the signature `(group, config) -> float`, lives
in `src/features/tier{1,2,3}.py`, and is registered in one dict in
`src/features/__init__.py`. Adding a feature never touches the extraction
loop in `src/extract.py`. `group` is a `src.streams.Group`: one
(user, session, q_id, r_t, version) cell with its key/mouse/cursor events,
stored response text, and matching question prompt.

## Design decisions already made (see task history, not re-litigated here)

- **Output is raw.** No scaling, no NaN imputation. XGBoost handles NaN
  natively; SVM doesn't -- so each downstream model decides for itself, and
  scaling must happen inside each CV fold, not before.
- **Indentation is normalized before any character-count feature** (one
  canonical unit char per indent LEVEL, tab vs N-space agnostic).
- **Line classification** (used identically for `comment_density`,
  `blank_line_ratio`, `comment_to_code_ratio`): BLANK if whitespace-only,
  COMMENT if first non-whitespace char is `#`, else CODE. The three buckets
  sum to 100% of lines.
- **Response text is authoritative** for tier2/tier3 text features.
  Keystroke-replay reconstruction (`src.streams.reconstruct_text`) exists
  only as a validation/sanity-check tool for `scripts/investigate.py`.
- **Response-field extraction is schema-flexible.** `RESPONSE_FIELD_MAP` in
  `src/loader.py` is a single, editable table from field name to
  `(r_t, version)`; an unrecognized field logs a loud warning ("response
  text LOST for this entry") instead of silently dropping data.

## Open questions (do not guess at -- run `scripts/investigate.py` on real data)

1. **What does keystroke `version` (1 vs 2) correspond to**, e.g. for
   session 2's `chatgptAnswer` / `retype` fields? `RESPONSE_FIELD_MAP`
   currently guesses `chatgptAnswer -> version 1`, `retype -> version 2`,
   flagged inline as tentative. `investigate_version_mapping()` correlates
   each version's keydown volume against each response field's text length
   to check this empirically.
2. **Does keystroke replay reproduce the stored text?**
   `investigate_reconstruction_agreement()` reports exact-match rate and a
   mean similarity ratio. The reconstruction heuristic has a known
   limitation around mid-line `Enter` (documented in
   `reconstruct_text`'s docstring) that this check is meant to surface.
3. **What fraction of code tokens are boilerplate** (identifiers/prototype
   code shared verbatim across participants for the same question)?
   `investigate_boilerplate()` flags identifiers present in >=90%/100% of
   users' code per question and reports what share of total identifier
   occurrences they account for -- to decide whether tier2_code_* features
   need boilerplate stripped first.

## NLP tooling choice

Explanation-text features (`tier2.py`, `EXPLANATION_FEATURES`) use **NLTK**
(word/sentence tokenization, POS tagging, English stopwords), not spaCy --
no separate language-model download, small corpora only. If NLTK's data
files are missing, functions raise `LookupError` with the exact
`nltk.download(...)` call needed, rather than silently returning NaN: a
missing corpus is a setup problem, not a property of the text.

Text is lowercased and stripped of surrounding punctuation before computing
`ttr` / `hapax_ratio` / `word_entropy` (config.yaml `tier2.text`) -- this is
a deliberate, documented choice since it materially changes those values.
