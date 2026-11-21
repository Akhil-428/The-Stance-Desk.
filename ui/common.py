"""Shared data access for the UI: cached model/metrics/dataset loaders.

Nothing here computes or invents metrics; it only reads what the training and
evaluation scripts saved (outputs/metrics/*.json, models/*/model_info.json)
and the FNC-1 CSVs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from evaluation import load_metrics_dir
from model import available_engines, load_predictor

ENGINE_SHORT = {"bert": "BERT", "lr": "Logistic Regression"}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _metrics_mtime_key(directory) -> tuple:
    return tuple(sorted((p.name, p.stat().st_mtime) for p in directory.glob("*.json"))) if directory.exists() else ()


@st.cache_data(show_spinner=False)
def _load(directory_str: str, freshness_key: tuple) -> dict:
    # NB: Streamlit excludes "_"-prefixed args from the cache key, so this one must not start with "_".
    return load_metrics_dir(Path(directory_str))


def ours_metrics() -> dict:
    d = config.METRICS_DIR / "ours"
    return _load(str(d), _metrics_mtime_key(d))


def historical_metrics() -> dict:
    d = config.METRICS_DIR / "historical"
    return _load(str(d), _metrics_mtime_key(d))


def model_info(kind: str) -> dict:
    d = {"bert": config.BERT_MODEL_DIR, "lr": config.LR_MODEL_DIR}[kind]
    p = d / "model_info.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def per_class_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for lab, v in metrics["per_class"].items():
        rows.append({"Class": lab, "Precision": v["precision"], "Recall": v["recall"], "F1": v["f1"],
                     "Support": v["support"], "Predicted": v["predicted"]})
    return pd.DataFrame(rows)


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def ordered_keys(data: dict, first: tuple[str, ...] = ("bert_fnc1", "lr_relational", "BERT")) -> list[str]:
    return sorted(data, key=lambda k: (k not in first, first.index(k) if k in first else 0, k))


# ---------------------------------------------------------------------------
# engines
# ---------------------------------------------------------------------------
def engine_status() -> dict[str, str]:
    return available_engines()


def _info_mtime(kind: str) -> float:
    d = {"bert": config.BERT_MODEL_DIR, "lr": config.LR_MODEL_DIR}[kind]
    p = d / "model_info.json"
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner=False, max_entries=2)
def _get_predictor(kind: str, info_mtime: float):
    # info_mtime is part of the cache key, so a re-trained model is reloaded automatically.
    return load_predictor(kind)


def get_predictor(kind: str):
    return _get_predictor(kind, _info_mtime(kind))


@st.cache_data(show_spinner=False)
def validation_examples() -> pd.DataFrame:
    """Real labelled examples from OUR validation split (never the test set)."""
    from preprocessing import load_fnc1, make_split

    df = load_fnc1("train")
    va = df.iloc[make_split(df).val_idx]
    return va[["Headline", "articleBody", "Stance"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# dataset statistics (computed from the FNC-1 CSVs, never typed in)
# ---------------------------------------------------------------------------
def _csv_key() -> tuple:
    d = config.FNC1_DIR
    return tuple(sorted((p.name, p.stat().st_mtime) for p in d.glob("*.csv"))) if d.exists() else ()


@st.cache_data(show_spinner=False)
def _dataset_stats(fnc1_dir: str, freshness_key: tuple) -> dict:
    d = Path(fnc1_dir)
    out = {}
    for split, prefix in (("train", "train"), ("test", "competition_test")):
        stances = pd.read_csv(d / f"{prefix}_stances.csv", encoding="utf-8")
        bodies = pd.read_csv(d / f"{prefix}_bodies.csv", encoding="utf-8")
        words = bodies["articleBody"].astype(str).str.split().str.len()
        out[split] = {
            "pairs": int(len(stances)),
            "bodies": int(bodies["Body ID"].nunique()),
            "headlines": int(stances["Headline"].nunique()),
            "counts": {l: int((stances["Stance"] == l).sum()) for l in config.DISPLAY_ORDER},
            "median_words": int(words.median()),
            "share_over_400_words": float((words > 400).mean()),
        }
    split_file = config.SPLIT_FILE
    if split_file.exists():
        try:
            s = json.loads(split_file.read_text(encoding="utf-8"))
            out["split"] = {k: (len(v) if isinstance(v, list) else v) for k, v in s.items()
                            if k in ("train_idx", "val_idx", "seed")}
        except (OSError, json.JSONDecodeError):
            pass
    return out


def dataset_stats() -> dict | None:
    try:
        return _dataset_stats(str(config.FNC1_DIR), _csv_key())
    except (OSError, KeyError, ValueError):
        return None
