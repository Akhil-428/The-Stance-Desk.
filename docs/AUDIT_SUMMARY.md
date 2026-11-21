# Audit of the original project: summary

This project re-implements [zeyongj/Fake-News-Stance-Detection](https://github.com/zeyongj/Fake-News-Stance-Detection), a 2021 course project (SFU CMPT 413/713, Group AWSL), at commit `7c93f795bb9d7445103cf1e936c8b72dbc3d6908`. Before writing any code, that repository was audited so that the re-implementation could keep what works and fix what does not. This page summarises the audit and defines the IDs (K1–K18, H-1, …) used in code comments, `DECISIONS.md` and `CHANGELOG.md`.

**Scope and method.** Static, read-only review of the notebook (all 66 cells), `output/scorer.py`, the committed CSVs, the PDFs and the git history. No code from that repository was executed, and its committed prediction CSVs were treated as data. The audit judges *reproducibility and methodology*, not the authors' intent: the work is a reasonable student implementation of BERT fine-tuning for FNC-1, and nothing malicious was found.

## What was confirmed

- Recomputing accuracy from the committed `output/BERT.csv` against the gold labels gives **0.903750 (22,967 / 25,413)**, exactly the reported 90.37 %. The same file gives a **relative FNC-1 score of 86.41 %** and **macro-F1 0.691**, with `disagree` at F1 0.36.
- This confirms the arithmetic, not the provenance: the fine-tuned checkpoint was loaded from a private Google Drive, the training cell is commented out, and the training data is not in the repository, so the predictions cannot be regenerated from it.

## Main findings

| Topic | Finding |
|---|---|
| Reproducibility | The repository cannot produce a prediction as shipped: no training data, no checkpoint, TPU-only code, installs that cannot work on current Python (torch 1.9 / transformers 4.12.5 pins). |
| Validation figure | The reported 99.2 % validation accuracy cannot be established as held-out: the split is unseeded, unstratified and row-level, and the 49,972 FNC-1 training pairs are built from only 1,683 article bodies (1,648 distinct headlines), so bodies appear on both sides of a random row split (G-P1, G-P2, G-L2). |
| Baselines | The Logistic Regression "CountVectorizer" and "TfidfVectorizer" results are **swapped** by the code (F.5, proven from the loop order). The two Naive Bayes files are byte-identical. A one-vs-rest `argmax` over 0/1 outputs forces uncertain rows to `unrelated`. Vectorisers were fitted before the split (G-P3). As a result the baselines barely beat "always unrelated" and the BERT-vs-baseline comparison is not meaningful. |
| Metric choice | 72.2 % of test pairs are `unrelated`, so accuracy flatters every model. Relative FNC score, macro-F1 and per-class F1 are the informative figures. |
| Method | The title says "prompt-based", but there is no prompt template or verbalizer; the input is `headline [SEP] body` into a standard classification head (K17). |
| Scorer | `output/scorer.py` crashes on Windows (cp1252 default encoding), silently truncates mismatched files, fails on unknown labels and prints only weighted metrics (H). |
| Security | No malicious code found. The risk is the dependency stack: loading a pickle-based `pytorch_model.bin` with torch 1.9 / transformers 4.12.5 is a code-execution path, and OSV.dev listed dozens of advisories for those versions at audit time (H-1). The notebook also installs unpinned and direct-URL packages (M-1) and clones and runs a third-party scorer (M-2). |

## Fix list (K-IDs) and where this project addresses each

| ID | Severity | Issue | Fix in this project |
|---|---|---|---|
| K1 | High | Model weights unavailable; training never runs | Retrained from official `bert-base-uncased` (safetensors, pinned revision): `train_bert.py` |
| K2 | High | Validation leakage and an invalid 0.992 | Seeded split grouped by Body ID, stratified, indices persisted |
| K3 | High | Count / TF-IDF names swapped | Names derive from the same definition as the features; historical files relabelled |
| K4 | High | One-vs-rest `argmax` forces rows to `unrelated` | `argmax(predict_proba(...))` |
| K5 | Medium | Bernoulli NB Count and TF-IDF identical | Reported once |
| K6 | Medium | Bag of words on the concatenation cannot model headline–body relation | New relation-feature engine (decision D10); the original-style baselines are kept for comparison |
| K7 | Medium | Vectorisers fitted before the split | `sklearn.Pipeline` fitted on the training fold |
| K8 | Medium | Literal `[SEP]` in a single string; `token_type_ids` unused | Sentence-pair encoding with segment ids (D1) |
| K9 | Medium | Accuracy-only selection; imbalance ignored | Selection on validation macro-F1; macro-F1 and FNC score reported (class weighting not applied, see limitations) |
| K10 | Medium | Scorer crashes on Windows (cp1252) | Own `evaluation.py`, explicit UTF-8, strict validation |
| K11 | Medium | TPU-only device code | CUDA if available, else CPU |
| K12 | Medium | Padding everything to 512 as Python lists uses a lot of RAM | Per-batch dynamic padding, compact arrays |
| K13 | Medium | Evaluates every 100 steps, no checkpoint limit | Two evaluations per epoch, save only improvements, fp16 |
| K14 | Low | Removed or deprecated APIs | Current equivalents |
| K15 | Low | `pd.merge` with both `left_on` and `left_index` | Plain key merge |
| K16 | Low | Outputs overwrite committed filenames | Outputs under `outputs/`, `runs/`, `models/` |
| K17 | Low | "Prompt-based" title without a prompt method; "512 characters" should be tokens; only weighted metrics | Described as standard BERT fine-tuning; macro metrics and per-class F1 reported |
| K18 | Low | All warnings suppressed | Warnings visible |
| H-1 | High | Pickle-based checkpoint loading with vulnerable library versions | safetensors only, pickle refused, modern pinned stack |

## How the app labels these results

The app and the README show the original project's numbers separately, tagged **"historical, self-reported, not reproduced"**, next to the numbers from models trained here. Our BERT is a **new training run on a stricter split**, not a reproduction of 90.37 %; the two are not an apples-to-apples comparison because the input encoding, the training subset and the checkpoint-selection rule all differ (see `DECISIONS.md`).
