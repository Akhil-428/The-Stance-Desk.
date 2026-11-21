"""Skip markers for tests that need files git does not hold (datasets, model weights).

On a fresh clone (and in CI) the FNC-1 CSVs, the historical prediction CSVs and the
model weights are absent, so those tests are skipped with a clear reason instead of
failing. After `python tools/fetch_fnc1.py` and training, they run.
"""
import pytest

import config
from model import available_engines

_FNC1 = ("train_bodies.csv", "train_stances.csv", "competition_test_bodies.csv", "competition_test_stances.csv")

HAVE_FNC1 = all((config.FNC1_DIR / f).exists() for f in _FNC1)
HAVE_HISTORICAL = (config.HISTORICAL_DIR / "BERT.csv").exists() and (
    config.HISTORICAL_DIR / "competition_test_stances.csv").exists()
HAVE_PRETRAINED = (config.PRETRAINED_BERT_DIR / "model.safetensors").exists()
HAVE_ENGINE = bool(available_engines())

requires_fnc1 = pytest.mark.skipif(not HAVE_FNC1, reason="FNC-1 CSVs missing: run python tools/fetch_fnc1.py")
requires_historical = pytest.mark.skipif(
    not HAVE_HISTORICAL, reason="historical CSVs missing: see data/README.md (evaluate_historical.py)")
requires_pretrained = pytest.mark.skipif(
    not HAVE_PRETRAINED, reason="pretrained bert-base-uncased missing: see models/README.md")
requires_engine = pytest.mark.skipif(
    not HAVE_ENGINE, reason="no trained model in models/: see models/README.md")
