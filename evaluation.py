"""Evaluation: standard metrics plus the official FNC-1 score.

The FNC-1 score is re-implemented from the published rule (the original
repository's scorer.py is neither copied nor run):
  +0.25 if gold and prediction agree on related vs unrelated;
  +0.75 more if gold is related and the exact stance is correct.
MAX = score of a perfect submission, NULL = score of "always unrelated".

Unlike the original scorer, this module never truncates silently: row counts,
row order and labels are validated and errors are raised clearly.
"""
from __future__ import annotations

import csv
import json
import platform
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import config

RELATED = {"agree", "disagree", "discuss"}


class EvaluationError(ValueError):
    pass


def _check_labels(labels: Sequence[str], what: str) -> None:
    unknown = sorted(set(labels) - set(config.LABELS))
    if unknown:
        raise EvaluationError(f"{what} contains unknown labels: {unknown}")


def fnc_score(gold: Sequence[str], pred: Sequence[str]) -> dict:
    if len(gold) != len(pred):
        raise EvaluationError(f"Length mismatch: {len(gold)} gold vs {len(pred)} predictions")
    _check_labels(gold, "gold")
    _check_labels(pred, "predictions")
    score = 0.0
    for g, p in zip(gold, pred):
        if g == p:
            score += 0.25
            if g != "unrelated":
                score += 0.50
        if g in RELATED and p in RELATED:
            score += 0.25
    n_unrel = sum(1 for g in gold if g == "unrelated")
    max_score = 0.25 * n_unrel + 1.0 * (len(gold) - n_unrel)
    null_score = 0.25 * n_unrel
    return {
        "score": score,
        "max": max_score,
        "null": null_score,
        "relative": score / max_score if max_score else 0.0,
        "null_relative": null_score / max_score if max_score else 0.0,
    }


def compute_metrics(gold: Sequence[str], pred: Sequence[str]) -> dict:
    """All metrics used by the dashboard. Labels are strings."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    gold, pred = list(gold), list(pred)
    fnc = fnc_score(gold, pred)  # validates lengths and labels
    order = config.DISPLAY_ORDER
    cm = confusion_matrix(gold, pred, labels=order)
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=order, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(gold, pred, labels=order, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(gold, pred, labels=order, average="weighted", zero_division=0)
    g_rel = np.array([g != "unrelated" for g in gold])
    p_rel = np.array([q != "unrelated" for q in pred])
    return {
        "n": len(gold),
        "accuracy": float(np.mean(np.array(gold) == np.array(pred))),
        "macro": {"precision": float(mp), "recall": float(mr), "f1": float(mf)},
        "weighted": {"precision": float(wp), "recall": float(wr), "f1": float(wf)},
        "per_class": {
            lab: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
                "predicted": int(cm[:, i].sum()),
            }
            for i, lab in enumerate(order)
        },
        "confusion_matrix": {"labels": order, "rows_gold_cols_pred": cm.tolist()},
        "fnc": fnc,
        "related_vs_unrelated_accuracy": float(np.mean(g_rel == p_rel)),
        "majority_baseline_accuracy": float(np.mean(np.array(gold) == "unrelated")),
    }


def read_stance_csv(path: Path) -> list[dict]:
    """Read a Headline/Body ID/Stance CSV as UTF-8 (fixes the Windows cp1252 crash)."""
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or [c.strip() for c in reader.fieldnames][:3] != ["Headline", "Body ID", "Stance"]:
            raise EvaluationError(f"{path.name}: expected header Headline,Body ID,Stance; got {reader.fieldnames}")
        return list(reader)


def score_prediction_files(gold_csv: Path, pred_csv: Path) -> dict:
    """Score a prediction CSV against gold, requiring identical row alignment."""
    gold_rows, pred_rows = read_stance_csv(Path(gold_csv)), read_stance_csv(Path(pred_csv))
    if len(gold_rows) != len(pred_rows):
        raise EvaluationError(f"Row count mismatch: gold {len(gold_rows)} vs predictions {len(pred_rows)}")
    from preprocessing import normalize_text as norm  # CRLF / whitespace-insensitive comparison

    for i, (g, p) in enumerate(zip(gold_rows, pred_rows)):
        if norm(g["Headline"]) != norm(p["Headline"]) or g["Body ID"] != p["Body ID"]:
            raise EvaluationError(f"Row {i} is misaligned (Headline/Body ID differ)")
    return compute_metrics([r["Stance"] for r in gold_rows], [r["Stance"] for r in pred_rows])


def save_metrics(metrics: dict, path: Path, **meta) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            **meta,
        },
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_metrics_dir(directory: Path) -> dict[str, dict]:
    """Load every metrics JSON in a directory, keyed by file stem."""
    directory = Path(directory)
    out = {}
    if directory.exists():
        for p in sorted(directory.glob("*.json")):
            try:
                out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    return out
