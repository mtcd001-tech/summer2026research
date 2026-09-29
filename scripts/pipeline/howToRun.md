# Running the Pipeline

Everything below assumes you are in the project root with the venv active:

```powershell
cd "C:\Users\cao minh\keystroke-feature-extraction"
.\.venv\Scripts\Activate.ps1
```

---

## Quick reference

| I want to... | Command |
|---|---|
| Sanity-check the wiring (~1 min) | `python scripts/pipeline/step2_train.py --model xgb --task binary --quick` |
| Run one model, one task | `python scripts/pipeline/step2_train.py --model xgb --task multiclass` |
| Run just one fold | `python scripts/pipeline/step2_train.py --model xgb --task binary --fold 3` |
| Run a few specific folds | `python scripts/pipeline/step2_train.py --model xgb --task binary --fold 1 3 5` |
| See results so far | `python scripts/pipeline/step3_collect.py` |
| Run everything | `python scripts/pipeline/run_all.py` |
| Rebuild the fold CSVs | `python scripts/pipeline/step1_prepare.py --force` |

---

## Running one model at a time

This is the normal workflow. Each combination is independent, so you can run one,
look at the result, and decide whether to continue.

```powershell
python scripts/pipeline/step2_train.py --model MODEL --task TASK
```

- `--model` is one of `mlp`, `svm`, `xgb`
- `--task` is one of `binary`, `multiclass`

This runs all 5 folds for that one combination and writes results to
`runs/{task}/{model}/fold{n}/results.json`.

**Suggested order.** Start with the most informative and fastest, end with the
slowest:

```powershell
python scripts/pipeline/step2_train.py --model xgb --task multiclass
python scripts/pipeline/step2_train.py --model xgb --task binary
python scripts/pipeline/step2_train.py --model svm --task multiclass
python scripts/pipeline/step2_train.py --model svm --task binary
python scripts/pipeline/step2_train.py --model mlp --task multiclass
python scripts/pipeline/step2_train.py --model mlp --task binary
```

XGBoost multiclass first because the confusion matrix is the most interesting
output — it shows which conditions get mistaken for which. MLP last because it is
the slowest and least likely to win on a dataset this size.

---

## Running specific folds

By default `step2_train.py` runs all 5 folds. To run a subset, pass `--fold` with
one or more fold numbers (1-5):

```powershell
python scripts/pipeline/step2_train.py --model xgb --task binary --fold 3
python scripts/pipeline/step2_train.py --model xgb --task binary --fold 1 3 5
```

Useful when one fold's result looks off and you want to redo just that one
(combine with `--force`, otherwise it's a no-op since the fold already has a
`results.json`):

```powershell
python scripts/pipeline/step2_train.py --model xgb --task binary --fold 3 --force
```

The skip-if-`results.json`-exists behavior is unchanged and applies per fold
regardless of `--fold` — it only changes which fold numbers are considered.
`step3_collect.py` doesn't assume folds 1..N are contiguous, so partial results
from any subset (e.g. only folds 1 and 3 done) are reported correctly, still
flagged `[PARTIAL: n/5 folds]`.

---

## Checking results

```powershell
python scripts/pipeline/step3_collect.py
```

Works at any point, including mid-run. Reports mean ± std accuracy and macro-F1
per model per task, plus the summed confusion matrix for multiclass, and flags any
combination that is only partially complete.

---

## Re-running something

By default, any fold that already has a `results.json` is skipped. To force a
re-run:

```powershell
python scripts/pipeline/step2_train.py --model xgb --task binary --force
```

Or delete the results directory for that combination:

```powershell
Remove-Item -Recurse runs\binary\xgb
```

---

## Quick mode

```powershell
python scripts/pipeline/step2_train.py --model xgb --task binary --quick
```

Cuts the genetic-algorithm budget to population=4, 1 generation, 2 CV splits
(and `max_iter=50` for MLP). Finishes in about a minute.

**Use this only to verify the wiring works** — that the script runs end to end and
writes results for all 5 folds. The hyperparameters it finds are effectively
random, so the accuracy is not a real result. Do not report quick-mode numbers.

---

## Rebuilding the datasets

Only needed if `features_clean.csv` changes or you change the label mapping or the
feature drop list.

```powershell
python scripts/pipeline/step1_prepare.py --force
```

Writes `datasets/{binary,multiclass}/{train,test}_fold{1..5}.csv`. Without
`--force` it skips if all folds already exist.

---

## Reading the numbers

**Binary task is imbalanced** — roughly 60 bona fide vs 240 assisted in each test
fold. A model that predicts "assisted" for everything scores about 80% accuracy
while being useless. **Report macro-F1, not accuracy**, and if you do quote
accuracy, state the 80% majority baseline next to it.

**Multiclass is balanced** — about 60 samples per class per fold. Chance is 20%.

**Both tasks use user-independent splits.** 20 users train, 5 users test, no
participant on both sides. Verified: zero user overlap across all folds. This is
a strict evaluation, so expect lower numbers than the effect sizes suggest —
that is the honest result and the one worth reporting.

**The confusion matrix matters more than the headline accuracy.** The specific
question is whether paraphrase (class 3) gets confused with bona fide (class 0).
That has been the persistent weak spot across both studies, and it is what the
code-structure features were added to address.

**FAR/FRR, and specifically per-condition FAR, are the direct answer to that
question** — look at the FAR-by-condition line for `s4` (paraphrase) before
anything else. `step3_collect.py` prints it separately from the pooled FAR
precisely because the pooled number can look fine while paraphrase alone is
leaking heavily into bona fide; the per-condition breakdown is where that
would actually show up. Pooled FAR/FRR match the binary task's definitions
(bona fide = positive class), so they're directly comparable across both
tasks and to the ICTAI/IJCB convention from the prior studies.

---

## Pipeline structure

```
scripts/pipeline/
  _common.py          shared paths, constants, GA budget presets
  step1_prepare.py    features_clean.csv -> fold train/test CSVs
  step2_train.py      one model + one task -> runs/{task}/{model}/fold{n}/
  step3_collect.py    all results.json -> summary table
  run_all.py          all 6 combinations, resumable
```

Data flow:

```
features_clean.csv
      |
      v  step1_prepare
datasets/{task}/{train,test}_fold{1..5}.csv
      |
      v  step2_train  (calls modeling/{mlp,svm,xgb}.py)
runs/{task}/{model}/fold{n}/results.json
      |
      v  step3_collect
summary table + confusion matrix
```

---

## Troubleshooting

**Nothing happens / a fold is skipped.** It already has a `results.json`. Use
`--force` or delete the directory.

**Want to know how far along a run is.** Check what has been written:

```powershell
Get-ChildItem -Recurse -Filter "results.json" | Select-Object LastWriteTime, FullName
```

**A run seems stuck.** The GA fits population x generations x CV-splits models per
fold, so a full run genuinely takes a while — MLP especially. Verify with `--quick`
first if you are unsure whether it is working at all.

**Verify the splits are still clean** after any change to step1:

```powershell
python -c "import pandas as pd; a=pd.read_csv('datasets/binary/train_fold1.csv'); b=pd.read_csv('datasets/binary/test_fold1.csv'); c=[x for x in a.columns if 'user' in x.lower()][0]; print('overlap:', set(a[c]) & set(b[c]))"
```

Must print `overlap: set()`.
