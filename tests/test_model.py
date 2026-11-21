"""3, 5, 6, 7, 8, 9: model loading, prediction output, class coverage, scores, errors, CPU/GPU."""
import json
import shutil

import numpy as np
import pytest
from _markers import requires_pretrained

import config
from model import (
    BertStancePredictor,
    InferenceError,
    LRStancePredictor,
    ModelUnavailableError,
    available_engines,
    load_predictor,
    probs_to_result_fields,
    softmax,
)
from preprocessing import InputValidationError

HEADLINE = "Apple to unveil new iPhone next month, sources say"
BODY = ("Apple is expected to unveil its next iPhone at an event next month, according to people familiar "
        "with the plans. The company declined to comment on the report.")


# ---------------------------------------------------------------- post-processing
@pytest.mark.parametrize("cls_id", range(4))
def test_each_of_four_classes_maps_correctly(cls_id):
    """Fixed logits peaking at each class go through the real post-processing."""
    logits = np.full(4, -2.0)
    logits[cls_id] = 3.0
    label, probs = probs_to_result_fields(softmax(logits), config.ID2LABEL)
    assert label == config.ID2LABEL[cls_id]
    assert set(probs) == set(config.LABELS)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)
    assert max(probs, key=probs.get) == label


def test_softmax_properties():
    p = softmax(np.array([[1000.0, 0, -1000, 5], [0.1, 0.2, 0.3, 0.4]]))
    assert np.allclose(p.sum(1), 1.0) and np.all(p >= 0) and np.all(np.isfinite(p))


@pytest.mark.parametrize("bad", [np.ones(3), np.array([np.nan, 1, 1, 1]), np.ones((2, 4))])
def test_bad_scores_raise(bad):
    with pytest.raises(InferenceError):
        probs_to_result_fields(bad, config.ID2LABEL)


# ---------------------------------------------------------------- missing / invalid models
def test_missing_bert_folder(tmp_path):
    with pytest.raises(ModelUnavailableError, match="not found"):
        BertStancePredictor(tmp_path / "nope")


def test_pickle_checkpoint_refused(tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"not loaded")
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ModelUnavailableError, match="pickle"):
        BertStancePredictor(tmp_path)


@requires_pretrained
def test_invalid_label_config_rejected(tmp_path):
    shutil.copy(config.PRETRAINED_BERT_DIR / "config.json", tmp_path / "config.json")
    (tmp_path / "model.safetensors").write_bytes(b"")
    cfg = json.loads((tmp_path / "config.json").read_text())
    cfg.update(num_labels=3, id2label={"0": "a", "1": "b", "2": "c"}, label2id={"a": 0, "b": 1, "c": 2})
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ModelUnavailableError, match="Invalid model configuration"):
        BertStancePredictor(tmp_path)


def test_lr_missing_and_tampered(tmp_path):
    with pytest.raises(ModelUnavailableError):
        LRStancePredictor(tmp_path)
    if not (config.LR_MODEL_DIR / "model.joblib").exists():
        pytest.skip("LR model not trained")
    shutil.copy(config.LR_MODEL_DIR / "model.joblib", tmp_path / "model.joblib")
    info = json.loads((config.LR_MODEL_DIR / "model_info.json").read_text())
    info["sha256"] = "0" * 64
    (tmp_path / "model_info.json").write_text(json.dumps(info))
    with pytest.raises(ModelUnavailableError, match="Integrity"):
        LRStancePredictor(tmp_path)


def test_no_models_gives_understandable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "BERT_MODEL_DIR", tmp_path / "bert")
    monkeypatch.setattr(config, "LR_MODEL_DIR", tmp_path / "lr")
    import model as M

    monkeypatch.setattr(M.BertStancePredictor.__init__, "__defaults__", (tmp_path / "bert", None))
    monkeypatch.setattr(M.LRStancePredictor.__init__, "__defaults__", (tmp_path / "lr",))
    assert available_engines() == {}
    with pytest.raises(ModelUnavailableError, match="No trained model is available"):
        load_predictor()


# ---------------------------------------------------------------- real LR engine
@pytest.fixture(scope="module")
def lr():
    if not (config.LR_MODEL_DIR / "model.joblib").exists():
        pytest.skip("LR model not trained")
    return LRStancePredictor()


def _check_result(r):
    assert r.label in config.LABELS
    assert set(r.probabilities) == set(config.LABELS)
    assert sum(r.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= v <= 1.0 for v in r.probabilities.values())
    assert r.label == max(r.probabilities, key=r.probabilities.get)


def test_lr_prediction_valid(lr):
    _check_result(lr.predict(HEADLINE, BODY))
    _check_result(lr.predict("Café ünïcode “test”", BODY + " Ünïcode — text."))
    with pytest.raises(InputValidationError):
        lr.predict("", BODY)


def test_lr_unrelated_pair_is_unrelated(lr):
    r = lr.predict("Scientists discover new species of frog in Amazon rainforest",
                   "The central bank raised interest rates by a quarter point on Tuesday, citing persistent inflation "
                   "in the housing market and strong employment figures.")
    assert r.label == "unrelated"


def test_lr_validation_predictions_cover_all_classes():
    m = json.loads((config.METRICS_DIR / "ours" / "lr_relational.json").read_text())
    assert all(v["predicted"] > 0 for v in m["metrics"]["per_class"].values())


# ---------------------------------------------------------------- real BERT engine
def _bert_ready():
    return (config.BERT_MODEL_DIR / "model.safetensors").exists()


@pytest.mark.slow
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_bert_prediction_cpu_gpu(device):
    import os

    import torch

    if not _bert_ready():
        pytest.skip("BERT model not trained yet")
    if device == "cuda" and (not torch.cuda.is_available() or os.environ.get("FNC_SKIP_GPU_TESTS")):
        pytest.skip("CUDA unavailable or GPU busy")
    p = BertStancePredictor(device=device)
    assert p.device == device
    r = p.predict(HEADLINE, BODY)
    _check_result(r)
    long = p.predict(HEADLINE, BODY + " filler" * 1500)
    assert long.truncated and long.n_tokens > 512
    _check_result(long)


@pytest.mark.slow
def test_bert_cpu_gpu_agree():
    import os

    import torch

    if not _bert_ready() or not torch.cuda.is_available() or os.environ.get("FNC_SKIP_GPU_TESTS"):
        pytest.skip("needs trained BERT and a free GPU")
    a = BertStancePredictor(device="cpu").predict(HEADLINE, BODY)
    b = BertStancePredictor(device="cuda").predict(HEADLINE, BODY)
    assert a.label == b.label
    for k in config.LABELS:
        assert a.probabilities[k] == pytest.approx(b.probabilities[k], abs=2e-3)


@pytest.mark.slow
def test_bert_oom_falls_back_to_cpu(monkeypatch):
    import torch

    if not _bert_ready():
        pytest.skip("BERT model not trained yet")
    p = BertStancePredictor(device="cpu")
    real = p._forward
    calls = {"n": 0}

    def flaky(enc, device):
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.OutOfMemoryError("simulated CUDA OOM")
        return real(enc, device)

    monkeypatch.setattr(p, "_forward", flaky)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    r = p.predict(HEADLINE, BODY)
    assert r.device == "cpu" and any("out of memory" in n for n in r.notes)


@pytest.mark.slow
def test_bert_inference_failure_wrapped(monkeypatch):
    if not _bert_ready():
        pytest.skip("BERT model not trained yet")
    p = BertStancePredictor(device="cpu")
    monkeypatch.setattr(p, "_forward", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(InferenceError, match="boom"):
        p.predict(HEADLINE, BODY)


def test_gpu_requested_but_unavailable(monkeypatch):
    import torch

    from model import choose_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert choose_device("auto") == "cpu"
    with pytest.raises(ModelUnavailableError, match="CUDA is not available"):
        choose_device("cuda")
