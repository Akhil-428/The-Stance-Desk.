"""Data loading, input validation, train/validation split and model input building.

Fixes relative to the original notebook (see docs/CHANGELOG.md):
* UTF-8 reads on Windows (audit K10).
* Plain key merge on Body ID (K15).
* Seeded, stratified split *grouped by Body ID*, persisted to disk (K2).
* One input-building function shared by training, evaluation and the app.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config


class InputValidationError(ValueError):
    """Raised for user input that cannot be classified."""


# ---------------------------------------------------------------------------
# Text handling
# ---------------------------------------------------------------------------
_WS = re.compile(r"[ \t\f\v]+")


def normalize_text(text: str) -> str:
    """Light, lossless-in-meaning normalisation: unify newlines, squeeze spaces."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    return text.strip()


def validate_inputs(headline, body) -> tuple[str, str]:
    """Return normalised (headline, body) or raise InputValidationError."""
    if headline is None or body is None:
        raise InputValidationError("Both a headline and an article body are required.")
    if not isinstance(headline, str) or not isinstance(body, str):
        raise InputValidationError("Headline and article body must be text.")
    headline, body = normalize_text(headline), normalize_text(body)
    if not headline:
        raise InputValidationError("Please enter a headline.")
    if not body:
        raise InputValidationError("Please paste the article body.")
    for name, value in (("Headline", headline), ("Article body", body)):
        if len(value) > config.MAX_INPUT_CHARS:
            raise InputValidationError(
                f"{name} is too long ({len(value):,} characters; limit {config.MAX_INPUT_CHARS:,})."
            )
    if not re.search(r"\w", headline) or not re.search(r"\w", body):
        raise InputValidationError("Inputs must contain words, not only symbols or punctuation.")
    return headline, body


# ---------------------------------------------------------------------------
# FNC-1 loading
# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {path}. See README 'Dataset' for how to obtain FNC-1."
        )
    return pd.read_csv(path, encoding="utf-8", dtype={"Body ID": int})


def load_fnc1(split: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load 'train' or 'test' (official competition test set) as one frame.

    Columns: Headline, Body ID, articleBody, Stance, label. Row order of the
    stances file is preserved (required by the FNC-1 scorer).
    """
    d = Path(data_dir or config.FNC1_DIR)
    prefix = {"train": "train", "test": "competition_test"}[split]
    stances = _read_csv(d / f"{prefix}_stances.csv")
    bodies = _read_csv(d / f"{prefix}_bodies.csv")
    if bodies["Body ID"].duplicated().any():
        raise ValueError(f"Duplicate Body IDs in {prefix}_bodies.csv")
    df = stances.merge(bodies, on="Body ID", how="left", validate="many_to_one")
    if df["articleBody"].isna().any():
        raise ValueError(f"{int(df['articleBody'].isna().sum())} stances have no matching body")
    unknown = set(df["Stance"]) - set(config.LABELS)
    if unknown:
        raise ValueError(f"Unknown stance labels: {unknown}")
    df["Headline"] = df["Headline"].map(normalize_text)
    df["articleBody"] = df["articleBody"].map(normalize_text)
    df["label"] = df["Stance"].map(config.LABEL2ID).astype(int)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Train / validation split (grouped by Body ID, stratified, seeded)
# ---------------------------------------------------------------------------
@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray


def make_split(train_df: pd.DataFrame, path: Path | None = None, seed: int = config.SEED) -> Split:
    """Create (or load) the persisted split. No Body ID appears in both parts."""
    from sklearn.model_selection import StratifiedGroupKFold

    path = Path(path or config.SPLIT_FILE)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["n_rows"] != len(train_df):
            raise ValueError("Saved split does not match the training data size")
        return Split(np.array(saved["train_idx"]), np.array(saved["val_idx"]))

    n_splits = round(1 / config.VAL_FRACTION)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(train_df, train_df["label"], groups=train_df["Body ID"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "method": f"StratifiedGroupKFold(n_splits={n_splits}, shuffle=True), fold 0, groups=Body ID",
                "n_rows": len(train_df),
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
            }
        ),
        encoding="utf-8",
    )
    return Split(train_idx, val_idx)


# ---------------------------------------------------------------------------
# Model inputs
# ---------------------------------------------------------------------------
def original_concat_text(headline: str, body: str) -> str:
    """Input string used by the original notebook (cell 13): Headline + "[SEP]" + body."""
    return f"{headline}[SEP]{body}"


def bert_encode(tokenizer, headlines, bodies, max_length: int = config.BERT["max_length"]):
    """Sentence-pair encoding: [CLS] headline [SEP] body [SEP] with segment ids.

    Decision D1 (docs/DECISIONS.md): same tokens as the original "[SEP]" join,
    but with token_type_ids set so BERT knows which part is the headline, and
    'longest_first' truncation so the (short) headline is kept intact.
    """
    if isinstance(headlines, str):
        headlines, bodies = [headlines], [bodies]
    return tokenizer(
        list(headlines),
        list(bodies),
        truncation="longest_first",
        max_length=max_length,
        padding=False,
    )
