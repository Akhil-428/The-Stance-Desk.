# Data

The datasets are **not redistributed** in this repository. Scripts fetch or verify them by SHA-256.

## FNC-1 (Fake News Challenge, Stage 1)

Source: <https://github.com/FakeNewsChallenge/fnc-1>, pinned to commit `29d473af2d15278f0464d5e41e4cbe7eb58231f2`. Please read that repository for the dataset's terms of use and citation. Task description: <http://www.fakenewschallenge.org>.

```powershell
python tools/fetch_fnc1.py            # downloads the four CSVs to data/raw/fnc1/ and verifies SHA-256
python tools/fetch_fnc1.py --verify   # check only, no network
```

Only the four data CSVs are fetched. No code from the upstream repository is downloaded or run.

| File | Rows | Content |
|---|---|---|
| `train_stances.csv` | 49,972 | `Headline`, `Body ID`, `Stance` (1,648 distinct headlines) |
| `train_bodies.csv` | 1,683 | `Body ID`, `articleBody` |
| `competition_test_stances.csv` | 25,413 | official test pairs (894 distinct headlines) |
| `competition_test_bodies.csv` | 904 | test article bodies |

Class distribution (computed from the files):

| Split | unrelated | discuss | agree | disagree |
|---|---|---|---|---|
| Train (49,972) | 36,545 (73.1 %) | 8,909 (17.8 %) | 3,678 (7.4 %) | 840 (1.7 %) |
| Test (25,413) | 18,349 (72.2 %) | 4,464 (17.6 %) | 1,903 (7.5 %) | 697 (2.7 %) |

A model that always answers *unrelated* therefore scores 72.2 % accuracy on the test set, which is why this project reports macro-F1, per-class F1 and the FNC-1 score alongside accuracy.

## Train / validation split

`data/splits/train_val_split.json` stores the row indices of the split used for all reported results: `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`, fold 0, **grouped by `Body ID`** so no article body appears on both sides. Result: 44,973 training pairs and 4,999 validation pairs. The official test set is used once, after model selection. The file is tracked so that results can be reproduced on the exact same split.

## Historical predictions (optional)

`data/historical/*.csv` are the prediction files committed in [zeyongj/Fake-News-Stance-Detection](https://github.com/zeyongj/Fake-News-Stance-Detection) (Apache-2.0, commit `7c93f795bb9d7445103cf1e936c8b72dbc3d6908`). They are git-ignored. The metrics recomputed from them are tracked in `outputs/metrics/historical/*.json`; to regenerate them, clone that repository and run:

```powershell
$env:FNC_ORIGINAL_REPO = "path\to\Fake-News-Stance-Detection"
python evaluate_historical.py
```
