"""Central configuration: paths, labels, seeds and hyperparameters.

All paths are relative to the project folder. The original repository is only
read (never written) by evaluate_historical.py, via the optional ORIGINAL_REPO.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# ---- data -----------------------------------------------------------------
DATA_DIR = PROJECT_DIR / "data"
FNC1_DIR = DATA_DIR / "raw" / "fnc1"
SPLIT_FILE = DATA_DIR / "splits" / "train_val_split.json"
HISTORICAL_DIR = DATA_DIR / "historical"

# Optional: a local clone of zeyongj/Fake-News-Stance-Detection (read-only
# reference). Set the FNC_ORIGINAL_REPO environment variable to its path to
# re-create the "historical" results with evaluate_historical.py. Unset by
# default; the committed outputs/metrics/historical/*.json already hold them.
_orig = os.environ.get("FNC_ORIGINAL_REPO")
ORIGINAL_REPO: Path | None = Path(_orig) if _orig else None
ORIGINAL_REPO_COMMIT = "7c93f795bb9d7445103cf1e936c8b72dbc3d6908"

# Source of the dataset, recorded for provenance.
FNC1_SOURCE = "https://github.com/FakeNewsChallenge/fnc-1"
FNC1_COMMIT = "29d473af2d15278f0464d5e41e4cbe7eb58231f2"

# ---- models ---------------------------------------------------------------
MODELS_DIR = PROJECT_DIR / "models"
PRETRAINED_BERT_DIR = MODELS_DIR / "pretrained" / "bert-base-uncased"
PRETRAINED_BERT_SOURCE = "google-bert/bert-base-uncased"
PRETRAINED_BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
BERT_MODEL_DIR = MODELS_DIR / "bert-fnc1"           # our fine-tuned model
LR_MODEL_DIR = MODELS_DIR / "lr-relational"          # our LR engine

# ---- outputs --------------------------------------------------------------
OUTPUTS_DIR = PROJECT_DIR / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# ---- task -----------------------------------------------------------------
# Integer ids follow the original notebook (cells 11-12) so numbers line up.
LABELS = ["unrelated", "discuss", "agree", "disagree"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
# Display / FNC-1 scorer order.
DISPLAY_ORDER = ["agree", "disagree", "discuss", "unrelated"]

LABEL_EXPLANATIONS = {
    "agree": "The article body supports or confirms the claim made in the headline.",
    "disagree": "The article body refutes, denies or contradicts the claim in the headline.",
    "discuss": "The article body talks about the same claim but does not take a clear position on whether it is true.",
    "unrelated": "The article body is about a different topic from the headline.",
}

SEED = 42
VAL_FRACTION = 0.10  # 1 of 10 StratifiedGroupKFold folds, grouped by Body ID

# ---- BERT fine-tuning (original cell 55 values unless noted) --------------
BERT = {
    "max_length": 512,
    "micro_batch_size": 8,
    "grad_accum_steps": 2,          # effective batch 16 (cell 55)
    "learning_rate": 5e-5,          # Trainer default used by the original
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "epochs": 3,
    "eval_batch_size": 32,
    "fp16": True,
}

# Input limits for the app.
MAX_INPUT_CHARS = 100_000
