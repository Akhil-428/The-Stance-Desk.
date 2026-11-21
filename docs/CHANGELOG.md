# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/). Audit IDs (K1–K18, H-1, …) refer to [AUDIT_SUMMARY.md](AUDIT_SUMMARY.md).

## [1.0.0] - 2026-11-21

First public release. A re-implementation of [zeyongj/Fake-News-Stance-Detection](https://github.com/zeyongj/Fake-News-Stance-Detection) (commit `7c93f795`), which was only read, never modified.

### Added
- **Models.** Fine-tuned `bert-base-uncased` (4-class stance, sentence-pair input, 512 tokens) trained by `train_bert.py`; a CPU Logistic Regression engine on headline/body TF-IDF plus relation features (`features.py`); bug-fixed re-implementations of the original Logistic Regression and Bernoulli Naive Bayes baselines.
- **Evaluation.** `evaluation.py`: accuracy, macro/weighted precision-recall-F1, per-class metrics, confusion matrix and the official FNC-1 score (raw and relative), with strict input validation. A regression test reproduces the independently verified figures for the original `BERT.csv` (10,068.25 / 11,651.25, accuracy 0.903750, macro-F1 0.691).
- **Data pipeline.** `preprocessing.py`: UTF-8 loading, key merge on `Body ID`, and a seeded, stratified train/validation split **grouped by article body**, persisted in `data/splits/`. `tools/fetch_fnc1.py` downloads the dataset from a pinned upstream commit and verifies SHA-256.
- **Application.** Streamlit app "The Stance Desk": a PREDICT view (one story box: first line headline, the rest article; verdict stamp, class probabilities, report) and a PROJECT & RESULTS view (overview, performance, comparison, methodology, dataset, limitations).
- **Reproducibility.** `evaluate_historical.py` recomputes the original repository's results (labelled historical, not reproduced). `docs/training_log.jsonl` records the training run behind the reported model.
- **Repository.** Apache-2.0 license, NOTICE, README with architecture diagrams, model card, security policy, contributing guide, issue/PR templates, CI workflow, Dependabot, citation file.

### Changed (relative to the original project)
| Audit | Change | Where |
|---|---|---|
| K2 | Seeded split grouped by Body ID and stratified; indices persisted | `preprocessing.make_split` |
| K3 | Count / TF-IDF names derive from the same definition as the features; historical files are relabelled in the app | `train_baselines.py`, `evaluate_historical.py` |
| K4 | `argmax(predict_proba)` instead of argmax over one-vs-rest 0/1 output | `train_baselines.py` |
| K5 | Bernoulli NB reported once (the two original files are byte-identical) | `evaluate_historical.py` |
| K7 | Vectorisers are fitted inside a `Pipeline` on the training fold only | `train_baselines.py` |
| K8 | Sentence-pair encoding with segment ids (decision D1) | `preprocessing.bert_encode` |
| K9 | Model selection on validation macro-F1; macro metrics and FNC score reported | `train_bert.py`, `evaluation.py` |
| K10 | UTF-8 and `newline=""` for all CSV I/O; strict alignment, no silent truncation | `evaluation.py` |
| K11 | CUDA if available, else CPU; `FNC_DEVICE` override | `model.choose_device` |
| K12 | Per-batch dynamic padding; compact int32 arrays instead of Python lists | `train_bert.py` |
| K13 | Two validations per epoch; only improving checkpoints are saved; fp16 | `train_bert.py` |
| K14 | No removed or deprecated APIs (`plot_roc_curve`, `multi_class`, `file_utils`, `evaluation_strategy`, `distutils`) | all |
| K16 | Outputs only under `outputs/`, `runs/`, `models/` | all |
| K18 | Warnings are no longer globally suppressed | all |
| H-1 | `safetensors` only; pickle checkpoints refused; offline loading | `model.py` |

The TPU-only dependencies (`torch_xla`, `cloud-tpu-client`) and the unavailable `sklearn==1.0.1` pin were dropped; see decision D0.

### Interface
- Replaced an earlier five-page layout with a single masthead and two views (PREDICT, PROJECT & RESULTS). Inputs, engine choice and the last verdict persist when switching views; deep links such as `/?section=performance` work.
- The verdict is drawn on the story surface (CSS stamp, about one second, respecting `prefers-reduced-motion`); model scores are shown as `>99.9%` / `<0.1%` rather than rounded to 100 % / 0 %; all user text is HTML-escaped.
- Desktop PREDICT fits one viewport (heights from `100dvh` tokens in `ui/static/stance_desk.css`).
