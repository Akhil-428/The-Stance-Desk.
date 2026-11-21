"""Model loading and inference.

Two engines, both trained by this project on FNC-1 (never fabricated output):
* BertStancePredictor - fine-tuned bert-base-uncased, loaded from safetensors only.
* LRStancePredictor   - Logistic Regression on headline/body TF-IDF + relation
  features (fallback engine; also used for comparison).

`load_predictor()` returns the best available engine or raises
ModelUnavailableError with a human-readable reason.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config
from preprocessing import bert_encode, validate_inputs


class ModelUnavailableError(RuntimeError):
    """No usable trained model (missing files, bad config, integrity failure)."""


class InferenceError(RuntimeError):
    """The model was loaded but inference failed."""


@dataclass
class PredictionResult:
    label: str
    probabilities: dict[str, float]          # all four labels, sums to 1
    model_name: str
    device: str
    truncated: bool = False                  # body cut to fit max_length
    n_tokens: int | None = None
    elapsed_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return self.probabilities[self.label]


def softmax(logits) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def probs_to_result_fields(probs, id2label: dict[int, str]) -> tuple[str, dict[str, float]]:
    """Map a probability vector (model id order) to (label, {label: prob})."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1 or len(probs) != len(id2label):
        raise InferenceError(f"Expected {len(id2label)} class scores, got shape {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise InferenceError("Model produced non-finite scores")
    probs = probs / probs.sum()
    pmap = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    return id2label[int(np.argmax(probs))], pmap


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_info(model_dir: Path) -> dict:
    p = model_dir / "model_info.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ModelUnavailableError(f"Invalid model_info.json in {model_dir}: {e}") from e


# ---------------------------------------------------------------------------
# BERT
# ---------------------------------------------------------------------------
def choose_device(preference: str | None = None) -> str:
    """'auto' -> cuda if available else cpu. Env FNC_DEVICE overrides."""
    import torch

    pref = (preference or os.environ.get("FNC_DEVICE", "auto")).lower()
    if pref == "cpu":
        return "cpu"
    if pref in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise ModelUnavailableError("GPU requested (FNC_DEVICE=cuda) but CUDA is not available.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


class BertStancePredictor:
    kind = "bert"

    def __init__(self, model_dir: Path = config.BERT_MODEL_DIR, device: str | None = None):
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise ModelUnavailableError(f"BERT model folder not found: {model_dir}")
        if not (model_dir / "model.safetensors").exists():
            if any(model_dir.glob("*.bin")) or any(model_dir.glob("*.pt")):
                raise ModelUnavailableError(
                    "Refusing to load a pickle-based checkpoint (.bin/.pt). Only safetensors is allowed."
                )
            raise ModelUnavailableError(f"model.safetensors missing in {model_dir}")
        if not (model_dir / "config.json").exists():
            raise ModelUnavailableError(f"config.json missing in {model_dir}")
        try:
            import torch
            from transformers import AutoConfig, AutoTokenizer, BertForSequenceClassification
        except ImportError as e:  # pragma: no cover
            raise ModelUnavailableError(f"Deep-learning libraries not installed: {e}") from e

        cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        id2label = {int(k): v for k, v in cfg.id2label.items()}
        if cfg.num_labels != 4 or sorted(id2label.values()) != sorted(config.LABELS):
            raise ModelUnavailableError(
                f"Invalid model configuration: expected 4 labels {config.LABELS}, found {id2label}"
            )
        self.id2label = id2label
        self.info = _read_info(model_dir)
        self.max_length = int(self.info.get("max_length", config.BERT["max_length"]))
        self.device = choose_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = BertForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, use_safetensors=True, trust_remote_code=False
        )
        self.model.eval()
        try:
            self.model.to(self.device)
        except RuntimeError as e:  # e.g. CUDA OOM while moving weights
            if self.device == "cuda":
                torch.cuda.empty_cache()
                self.device = "cpu"
                self.model.to("cpu")
            else:
                raise ModelUnavailableError(f"Could not load model: {e}") from e
        self.name = self.info.get("display_name", "BERT (bert-base-uncased fine-tuned on FNC-1)")

    def _forward(self, enc, device):
        import torch

        batch = {k: torch.tensor([v]).to(device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**batch).logits[0].float().cpu().numpy()
        return logits

    def predict(self, headline: str, body: str) -> PredictionResult:
        import torch

        headline, body = validate_inputs(headline, body)
        t0 = time.perf_counter()
        enc = bert_encode(self.tokenizer, headline, body, self.max_length)
        enc = {k: v[0] for k, v in enc.items()}
        # Length count only (to warn about truncation); the model input above is already <= max_length.
        full_len = len(self.tokenizer(headline, body, truncation=False, verbose=False)["input_ids"])
        notes = []
        device = self.device
        try:
            logits = self._forward(enc, device)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.model.to("cpu")
            self.device = device = "cpu"
            notes.append("GPU ran out of memory; this prediction was computed on the CPU.")
            logits = self._forward(enc, device)
        except Exception as e:  # noqa: BLE001
            raise InferenceError(f"Inference failed: {e}") from e
        label, probs = probs_to_result_fields(softmax(logits), self.id2label)
        return PredictionResult(
            label=label, probabilities=probs, model_name=self.name, device=device,
            truncated=full_len > self.max_length, n_tokens=full_len,
            elapsed_ms=(time.perf_counter() - t0) * 1000, notes=notes,
        )


# ---------------------------------------------------------------------------
# Logistic Regression (relational features)
# ---------------------------------------------------------------------------
class LRStancePredictor:
    kind = "lr"

    def __init__(self, model_dir: Path = config.LR_MODEL_DIR):
        model_dir = Path(model_dir)
        path = model_dir / "model.joblib"
        info = _read_info(model_dir)
        if not path.exists() or not info:
            raise ModelUnavailableError(f"Logistic Regression model not found in {model_dir}")
        # joblib is pickle-based. We only load the file our own training script
        # wrote, and verify its SHA-256 against the hash recorded at save time.
        if sha256_file(path) != info.get("sha256"):
            raise ModelUnavailableError("Integrity check failed: model.joblib hash does not match model_info.json")
        import joblib

        self.pipeline = joblib.load(path)
        classes = list(self.pipeline.classes_)
        if sorted(classes) != sorted(config.LABELS):
            raise ModelUnavailableError(f"Invalid LR model classes: {classes}")
        self.id2label = {i: c for i, c in enumerate(classes)}
        self.info = info
        self.device = "cpu"
        self.name = info.get("display_name", "Logistic Regression (TF-IDF + relation features)")

    def predict_proba_frame(self, headlines, bodies) -> np.ndarray:
        import pandas as pd

        return self.pipeline.predict_proba(pd.DataFrame({"Headline": headlines, "articleBody": bodies}))

    def predict(self, headline: str, body: str) -> PredictionResult:
        headline, body = validate_inputs(headline, body)
        t0 = time.perf_counter()
        try:
            probs = self.predict_proba_frame([headline], [body])[0]
        except Exception as e:  # noqa: BLE001
            raise InferenceError(f"Inference failed: {e}") from e
        label, pmap = probs_to_result_fields(probs, self.id2label)
        return PredictionResult(label=label, probabilities=pmap, model_name=self.name, device="cpu",
                                elapsed_ms=(time.perf_counter() - t0) * 1000)


# ---------------------------------------------------------------------------
ENGINES = {"bert": BertStancePredictor, "lr": LRStancePredictor}


def available_engines() -> dict[str, str]:
    """Engines whose files exist (cheap check, no loading)."""
    out = {}
    if (config.BERT_MODEL_DIR / "model.safetensors").exists():
        out["bert"] = "BERT (fine-tuned bert-base-uncased)"
    if (config.LR_MODEL_DIR / "model.joblib").exists():
        out["lr"] = "Logistic Regression (TF-IDF + relation features)"
    return out


def load_predictor(kind: str | None = None, **kwargs):
    """Load a specific engine, or the best available one (BERT, then LR)."""
    kinds = [kind] if kind else ["bert", "lr"]
    reasons = []
    for k in kinds:
        if k not in ENGINES:
            raise ModelUnavailableError(f"Unknown engine '{k}'")
        try:
            return ENGINES[k](**kwargs) if k == "bert" else ENGINES[k]()
        except ModelUnavailableError as e:
            reasons.append(f"{k}: {e}")
    raise ModelUnavailableError("No trained model is available. " + " | ".join(reasons))
