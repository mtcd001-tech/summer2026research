# Shared feature extraction for CS1 and CS2

Run from the repository root:

```powershell
python scripts/extract_features.py
python scripts/extract_features.py --datasets dataCs2
python scripts/extract_features.py --feature-set existing --list-features
python scripts/extract_features.py --feature-set timing
```

Default `--feature-set both` extracts both families for both cohorts. Install
`requirements.txt` and the NLTK resources documented in the project README
for the existing text features. All paths can be supplied on the command line;
there is no runtime dependency on the Vietnamese repository or a D: drive.

## Features checked against the reference project

`Vietnamese_keystrokes/src/pipeline_v2/new_study_pipeline.py` covers the five
behavioral scenarios and tiered timing/revision/text features. Our existing
registry contains 56 features: 18 behavioral features for each response type,
8 code stylometry features, 9 explanation features, and 3 code-format features.
`processed/feature_catalog.csv` lists the actual names and implementations.
We reuse the tested local implementations: the reference's prototype computes
some KIT statistics using hold times, whereas local KIT correctly uses
successive keydowns. Stored responses remain authoritative for text features.

`Vietnamese_keystrokes/src/pipeline_v2/step1_extract.py` provides the additional
per-key and per-bigram timing definitions:

| Family | Measurement |
|---|---|
| KHT | Matched keyup minus keydown, per key |
| KIT | Consecutive eligible keydown interval, per bigram |
| RUKD | Current keydown minus most recent matched letter/space release, per bigram |

Each produces mean, population standard deviation, CV, range, maximum and
median. Filtering matches the reference: 50 through 5000 milliseconds inclusive,
followed by a two-IQR fence. CV is dimensionless; other statistics are in ms.
This reference RUKD definition differs from the existing `rkdt_cv`, which uses
the release of the preceding keydown pair and can include overlapping presses.

The default fixed vocabulary contains 29 KHT keys (a-z, space, backspace, shift)
and all 729 a-z/space bigrams for KIT and RUKD: **8,922 timing columns**.
This gives both cohorts identical columns without learning a vocabulary from
the evaluation set. Punctuation and digits are outside this reference key set.
Case is normalized; repeated keydowns are excluded; keyups are consumed once
using FIFO pairing by physical code. Events are sorted within each group.
Missing/filtered-out timings remain NaN rather than the reference's zero fill.
All-empty feature columns are retained to preserve the shared schema.

For a smaller fixed vocabulary, pass `--bigrams vocabulary.json`, containing
a JSON list such as `["a->b", "t->h", " ->a"]`. Reuse it across cohorts. If
chosen by frequency for a predictive experiment, learn that list on training
participants only. This extractor does not perform feature selection.

## Outputs in each cohort's processed directory

| File | Contents |
|---|---|
| `features.csv` | Existing 56-feature family in long format, compatible with importance analysis |
| `features_wide.csv` | Existing features; one row per question/response type |
| `features_questions.csv` | Existing code and explanation features aligned into one row per question |
| `features_timing.csv` | Additional Vietnamese-style timings; one row per question/response type |
| `features_combined.csv` | Both families joined by full group identity |
| `feature_catalog.csv`, `timing_feature_catalog.csv` | Names, implementations or timing definitions |
| `feature_quality.csv`, `timing_feature_quality.csv` | Observed and missing counts |
| `coverage.csv` | Complete participant/file-pair counts for all five scenarios |
| `extraction_metadata.json`, `timing_metadata.json` | Configuration, pairing/filtering rules and extraction summaries |

Features are computed independently per `(user, session, q_id, r_t, version)`;
no interval crosses question boundaries. Keep `(dataset, user)` together when
splitting participants across training and evaluation sets. Metadata columns
(`dataset`, `user`, `session`, `q_id`, `r_t`, `version`, `scenario`, `label`) are
not predictors. Labels are 0..4 in the shared scenario order.

Only human-typed versions are included: v1 for Bona fide, v2 for sessions 2-5.
Missing session pairs are reported and skipped; short groups remain included.
Scaling and imputation belong inside training folds. Code/explanation features
that do not apply to a response type remain NaN. `features_combined.csv` retains
this response-level layout; `features_questions.csv` is the joint layout for
the original family only. Do not merge users across cohorts solely by user ID.

`src/dataset_features.py` exposes `extract_dataset`, `extract_timing_dataset`,
`combine_feature_sets` and `feature_catalog` for Python callers. The pure timing
functions live in `src/features/vietnamese_timing.py`. Both cohorts invoke the
same code. The earlier importance heatmaps cover the existing family; rerunning
extraction does not add the new timing columns to those earlier heatmaps.
