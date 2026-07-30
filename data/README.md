# Expected data layout

This folder is intentionally empty except for this file (dataset contents are
gitignored). Unzip/copy the CS1 dataset here so that `src/loader.py` can discover
it automatically.

## Directory layout

One subfolder per participant, named with a stable user id (any string works,
but must be consistent across sessions for the same person). Inside each user
folder, one pair of JSON files per session (sessions 1-5):

```
data/
  raw/
    user_001/
      s1_keystrokes.json
      s1_responses.json
      s2_keystrokes.json
      s2_responses.json
      s3_keystrokes.json
      s3_responses.json
      s4_keystrokes.json
      s4_responses.json
      s5_keystrokes.json
      s5_responses.json
    user_002/
      s1_keystrokes.json
      s1_responses.json
      ...
    ...
```

- `src/loader.py` recursively looks for `data/raw/<user>/s{n}_keystrokes.json`
  and its matching `s{n}_responses.json`, for `n` in 1-5. The user id is taken
  from the immediate parent directory name.
- A user is not required to have all 5 sessions or all 6 questions present;
  missing files/questions are logged and skipped rather than treated as a
  fatal error (participants can drop out or a file can fail to upload).
- If your unzip produces a different top-level folder name than `raw/`, either
  rename it to `raw/` or point `--data-root` (see `src/extract.py`) at the
  folder that directly contains the per-user subfolders.

## Session -> condition mapping

| session | condition   |
|---------|-------------|
| 1       | Bona fide   |
| 2       | Transcribe  |
| 3       | Mimicry     |
| 4       | Paraphrase  |
| 5       | Deception   |

## After the data lands

Run, in order:

```
python scripts/validate_data.py --data-root data/raw
python scripts/investigate.py --data-root data/raw
python -m src.extract --data-root data/raw --out features.csv
```

`validate_data.py` checks structure/schema conformance and prints a
per-user/per-session coverage report before you trust anything downstream.
`investigate.py` answers the three open empirical questions documented in the
project README (version <-> response-field mapping, keystroke-reconstruction
agreement with stored text, boilerplate token fraction) - it makes no
assumptions the extraction code depends on, it's purely diagnostic.
