"""1, 6, 9, 10: every view renders; prediction UI end to end; 'not ready' state; navigation state.

The UI was redesigned (one PREDICT view + one PROJECT & RESULTS view instead of
five pages). The original twelve tests keep their names and intent; they now
drive the new widgets (see docs/CHANGELOG.md). New tests cover navigation and
session-state behaviour introduced by the redesign.
"""
import json
import os

import pytest
from _markers import requires_engine
from streamlit.testing.v1 import AppTest

import config

PROJECT = str(config.PROJECT_DIR)
SECTIONS = ["overview", "performance", "comparison", "methodology", "dataset", "limitations"]
APPLE = ("Apple to unveil new iPhone next month",
         "Apple is expected to unveil its next iPhone next month, according to people familiar with the plans.")


def page(name: str) -> AppTest:
    script = (
        "import sys, os\n"
        f"sys.path.insert(0, {PROJECT!r})\n"
        "os.environ.setdefault('FNC_DEVICE', 'cpu')\n"
        "from ui.navigation import run_app\n"
        f"run_app({name!r})\n"
    )
    return AppTest.from_string(script, default_timeout=300)


def html(at) -> str:
    return " ".join(m.value for m in at.markdown)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("FNC_DEVICE", os.environ.get("FNC_APP_TEST_DEVICE", "cpu"))


# --- original tests, adapted to the redesigned UI ---------------------------
@pytest.mark.parametrize("name", ["home", "predict", "performance", "comparison", "about"])
def test_page_renders_without_exceptions(name):
    """Former page names are still accepted by run_app() and map onto the two views."""
    at = page(name).run()
    assert not at.exception, [e.message for e in at.exception]


def test_home_mentions_not_fact_checking():
    at = page("home").run()
    assert "not fact-checking" in html(at)


def _metric_json(source: str, key: str) -> dict:
    return json.loads((config.METRICS_DIR / source / f"{key}.json").read_text(encoding="utf-8"))["metrics"]


def test_performance_shows_historical_and_ours():
    at = page("performance").run()
    assert not at.exception
    text = html(at)
    assert "Trained here" in text
    for label in ("Accuracy", "Macro-F1", "FNC-1 score"):
        assert label in text
    ours = _metric_json("ours", "bert_fnc1")
    assert f"{ours['accuracy'] * 100:.2f}%" in text
    at.segmented_control(key="perf_source").set_value("hist")
    at = at.run()
    text = html(at)
    assert "Historical, self-reported and not reproduced" in text
    # historical BERT accuracy as audited (and as read from the JSON)
    assert "90.38%" in text
    assert f"{_metric_json('historical', 'BERT')['accuracy'] * 100:.2f}%" in text


def _submit(at, headline, body, engine=None):
    """One story box: first line headline, the rest the article (see ui/story.py)."""
    if engine:
        at.segmented_control(key="engine").set_value(engine)
    at.text_area(key="story").input(f"{headline}\n{body}" if body else headline)
    at.button(key="analyze").click()
    return at.run()


@requires_engine
def test_predict_empty_input_warns():
    at = page("predict").run()
    at = _submit(at, "", "")
    assert "Please enter a headline" in html(at)
    assert at.session_state["verdict"] is None
    assert not at.exception


def test_predict_real_lr_result():
    if not (config.LR_MODEL_DIR / "model.joblib").exists():
        pytest.skip("LR model missing")
    at = page("predict").run()
    at = _submit(at, *APPLE, engine="lr")
    assert not at.exception, [e.message for e in at.exception]
    text = html(at)
    assert "Predicted stance" in text
    v = at.session_state["verdict"]
    assert v["label"] in config.LABELS and f">{v['label'].upper()}<" in text
    assert abs(sum(v["probabilities"].values()) - 1) < 1e-6
    assert "Logistic Regression" in v["model_name"] and v["model_name"] in text


def test_predict_real_bert_result():
    if not (config.BERT_MODEL_DIR / "model.safetensors").exists():
        pytest.skip("BERT model missing")
    at = page("predict").run()
    at = _submit(at, *APPLE, engine="bert")
    assert not at.exception, [e.message for e in at.exception]
    assert "Predicted stance" in html(at)
    v = at.session_state["verdict"]
    assert v["model_name"].startswith("BERT") and v["device"] == "cpu"


def test_not_ready_when_no_models(tmp_path):
    script = (
        "import sys\n"
        f"sys.path.insert(0, {PROJECT!r})\n"
        "import config\n"
        f"config.BERT_MODEL_DIR = __import__('pathlib').Path({str(tmp_path / 'b')!r})\n"
        f"config.LR_MODEL_DIR = __import__('pathlib').Path({str(tmp_path / 'l')!r})\n"
        "from ui.navigation import run_app\n"
        "run_app('predict')\n"
    )
    at = AppTest.from_string(script, default_timeout=120).run()
    try:
        assert not at.exception
        assert "Prediction engine not ready" in html(at)
        assert "Predicted stance" not in html(at)
        assert at.button(key="analyze").disabled
    finally:
        import importlib

        importlib.reload(config)  # restore real paths for other tests


def test_dashboard_cache_refreshes_when_metrics_change(tmp_path):
    """Regression: '_'-prefixed args are not part of Streamlit's cache key; new results must appear."""
    from ui.common import _load, _metrics_mtime_key

    assert _load(str(tmp_path), _metrics_mtime_key(tmp_path)) == {}
    (tmp_path / "new_model.json").write_text(json.dumps({"meta": {}, "metrics": {}}), encoding="utf-8")
    assert "new_model" in _load(str(tmp_path), _metrics_mtime_key(tmp_path))


# --- new: navigation and state introduced by the redesign -------------------
def test_predict_is_default_view_and_no_sidebar():
    at = AppTest.from_string(
        f"import sys; sys.path.insert(0, {PROJECT!r})\nfrom ui.navigation import run_app\nrun_app()\n",
        default_timeout=120).run()
    assert not at.exception
    assert at.segmented_control(key="view").value == "predict"
    assert at.text_area(key="story") is not None and at.button(key="analyze") is not None
    assert len(at.sidebar) == 0


@pytest.mark.parametrize("section", SECTIONS)
def test_each_results_section_renders(section):
    at = page("results").run()
    at.button(key=f"sec_{section}_off" if section != "overview" else "sec_overview_on").click()
    at = at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.session_state["section"] == section
    assert f'class="sd-sec-kicker">{SECTIONS.index(section) + 1:02d} /' in html(at)


def test_comparison_separates_ours_historical_and_reference():
    at = page("comparison").run()
    text = html(at)
    for marker in ("Trained in this project", "Historical · original repository (not reproduced)",
                   "Always predict &#x27;unrelated&#x27;"):
        assert marker in text
    ours = _metric_json("ours", "lr_relational")
    assert f"{ours['macro']['f1']:.3f}" in text


def test_switching_views_keeps_inputs_and_verdict():
    if not (config.LR_MODEL_DIR / "model.joblib").exists():
        pytest.skip("LR model missing")
    at = page("predict").run()
    at = _submit(at, *APPLE, engine="lr")
    label = at.session_state["verdict"]["label"]
    at.segmented_control(key="view").set_value("results")
    at = at.run()
    assert "Model performance" in " ".join(b.label for b in at.button)
    at.segmented_control(key="view").set_value("predict")
    at = at.run()
    assert at.text_area(key="story").value == f"{APPLE[0]}\n{APPLE[1]}"
    assert at.segmented_control(key="engine").value == "lr"
    assert at.session_state["verdict"]["label"] == label
    assert 'sd-fresh"' not in html(at)  # the stamp animation only plays on the run that produced it


def test_edit_story_keeps_text_and_new_analysis_clears_it():
    if not (config.LR_MODEL_DIR / "model.joblib").exists():
        pytest.skip("LR model missing")
    at = page("predict").run()
    at = _submit(at, *APPLE, engine="lr")
    assert 'sd-fresh"' in html(at)
    assert at.text_area(key="story").disabled  # the stamped story is read-only until EDIT STORY
    at.button(key="edit_story").click()
    at = at.run()
    assert at.session_state["verdict"] is None and not at.text_area(key="story").disabled
    assert at.text_area(key="story").value.startswith(APPLE[0])
    at = _submit(at, *APPLE)
    at.button(key="new_analysis").click()
    at = at.run()
    assert at.session_state["verdict"] is None
    assert at.text_area(key="story").value == ""


def test_single_story_box_feeds_both_model_inputs():
    """No second input field; the verdict comes from the parsed headline and body."""
    if not (config.LR_MODEL_DIR / "model.joblib").exists():
        pytest.skip("LR model missing")
    at = page("predict").run()
    assert len(at.text_input) == 0 and len(at.text_area) == 1
    at = _submit(at, APPLE[0], APPLE[1] + "\n\nA second paragraph.", engine="lr")
    v = at.session_state["verdict"]
    assert v["headline"] == APPLE[0]
    assert v["body"] == APPLE[1] + "\n\nA second paragraph."
    assert 'class="sd-ov sd-ov--verdict' in html(at)  # the stamp is drawn on the story surface


@requires_engine
def test_headline_without_article_is_rejected():
    at = page("predict").run()
    at = _submit(at, "Only a headline", "")
    assert "Please paste the article body" in html(at)
    assert at.session_state["verdict"] is None and not at.exception


def test_verdict_never_uses_truth_words():
    from ui import theme

    verdict = {"label": "disagree", "probabilities": {"agree": .1, "disagree": .7, "discuss": .1, "unrelated": .1},
               "model_name": "m", "device": "cpu", "elapsed_ms": 1.0, "truncated": False, "n_tokens": None,
               "max_length": 512, "notes": [], "headline": "h <b>x</b>", "body": "b", "engine": "lr",
               "engine_short": "LR", "gold": None, "filed": "00:00:00"}
    stamps = " ".join(theme.stamp(lab, 0.9, "BERT") for lab in config.LABELS).upper()
    for word in ("TRUE", "FAKE", "VERIFIED", "DEBUNKED"):
        assert word not in stamps
    out = (theme.overlay_verdict(verdict, True) + theme.foot_verdict(verdict) + theme.verdict_report(verdict, True)
           + theme.foot_idle(verdict["headline"], verdict["body"]))  # foot_idle echoes the user's headline
    assert "<b>x</b>" not in out and "&lt;b&gt;x&lt;/b&gt;" in out  # user text is escaped
    assert theme.score(0.99996) == ">99.9%" and theme.score(0.00001) == "<0.1%"  # no misleading 100.0%


def test_query_param_deep_link_opens_section():
    at = page("predict")
    at.query_params["section"] = "dataset"
    at = at.run()
    assert not at.exception
    assert at.session_state["view"] == "results" and at.session_state["section"] == "dataset"


def test_predict_layout_is_budgeted_to_one_viewport():
    """The desktop PREDICT layout is sized from 100dvh (also verified in a real browser at 1920x1080, 1366x768 and 1100x700)."""
    css = (config.PROJECT_DIR / "ui" / "static" / "stance_desk.css").read_text(encoding="utf-8")
    assert "--mon-h: calc(100dvh" in css
    assert ".st-key-monitor { height: var(--mon-h) !important; }" in css
    assert "sd-strip-bar" not in css  # the bottom broadcast banner was removed
    assert "overflow: hidden" not in css.split("/* ---------- desktop PREDICT")[1].split("}")[0]
