# Shared cohort layout

CS1 and CS2 both use this structure:

```text
raw/<user>/Bona fide/s1_{keystrokes,responses}.json
raw/<user>/Transcription/s2_{keystrokes,responses}.json
raw/<user>/Mimicry/s3_{keystrokes,responses}.json
raw/<user>/Paraphrasing/s4_{keystrokes,responses}.json
raw/<user>/Deception/s5_{keystrokes,responses}.json
processed/features.csv
```

Each brace expression denotes two files. Optional `sN_user_info.json` files
are retained. Missing sessions are allowed. Participant identity is local
to each cohort. `normalization_manifest.json` records path changes and hashes.

See [shared commands and analysis methods](../scripts/analysis/README.md).
