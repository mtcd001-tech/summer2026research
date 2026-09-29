# Five-scenario feature relevance

Both cohorts use the same layout and commands:

```text
dataCs1/ (and dataCs2/)
  raw/<user>/Bona fide/s1_{keystrokes,responses}.json
  raw/<user>/Transcription/s2_{keystrokes,responses}.json
  raw/<user>/Mimicry/s3_{keystrokes,responses}.json
  raw/<user>/Paraphrasing/s4_{keystrokes,responses}.json
  raw/<user>/Deception/s5_{keystrokes,responses}.json
  processed/features.csv
  normalization_manifest.json
```

Optional `sN_user_info.json` files are retained. Missing sessions are not
fabricated. Dataset identity remains separate, so identical user IDs in
different cohorts are never merged.

```powershell
.venv/Scripts/python.exe scripts/normalize_data.py
.venv/Scripts/python.exe scripts/normalize_data.py --apply
.venv/Scripts/python.exe scripts/analysis/scenario_importance.py --extract
# Replot/recompute from the extracted tables:
.venv/Scripts/python.exe scripts/analysis/scenario_importance.py --top-n 20
```

Use `--datasets dataCs1` for one cohort or pass multiple dataset directories.
`--top-n 0` plots every feature. `src/scenarios.py` owns the shared naming.
The normalizer preflights path collisions, preserves bytes, and records old
and new relative paths plus SHA-256 hashes before renaming. To reverse a
migration, use the manifest's target-to-source mapping after checking that
each target still has its recorded hash and the original path is vacant.
Already-normalized trees require no changes.

Outputs under `analysis/scenarios/`:

- `scenario_importance.csv`: all features and five scenarios, with cohort,
  response type, MI, signed Cohen's d, rank, observed/missing counts and status.
- `<cohort>/<response_type>/*_matrix.csv`: complete feature-by-scenario tables.
- `*_plotted.csv`: exact rows and values plotted, ordered by maximum absolute score.
- `*_heatmap.png` and `*_heatmap.pdf`: separate MI and effect-size heatmaps.
- `metadata.json`: inputs, seed and method.

Each scenario is the positive class versus observations from the other four.
MI measures univariate dependence in nats. Cohen's d uses sample-size-weighted
pooled variance; positive means the scenario's feature mean is higher than
the rest. Code and explanation are scored separately. Missing/nonfinite values
are excluded feature by feature; fewer than two observations in either class
produce NA, displayed gray. Constant features have zero MI and undefined d.
The rest group follows observed sample frequencies, not equal scenario weights.

These are descriptive scores on all available question-level observations.
They are not model coefficients, held-out permutation importance, causal
effects, or significance tests. Repeated observations from users and unequal
cohort/scenario coverage affect interpretation. Colors scale independently per
heatmap; compare CSV values when comparing plots. The existing
`feature_importance.py` remains available for saved-model permutation scores
and selection stability; it uses the existing runs and split files.

Reference context: `D:/New folder (2)/Vietnamese_keystrokes/src/pipeline_v2/plot_importance.py`
aggregates XGBoost importances from saved folds. This report shares its reusable
table-to-plot approach but explicitly uses the five behavioral scenarios and
does not reuse unrelated Vietnamese models or their M2-M5 scenario labels.
