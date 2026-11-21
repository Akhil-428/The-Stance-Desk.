"""PREDICT: one newsroom monitor with ONE story box.

The user types a single story: first line = headline, the rest = article body
(see ui/story.py). That same box is the analysis surface: on submit a scan
overlay runs across it while the real model call happens, and the verdict
stamp lands over it. There is no second input and no separate result panel.

Flow of one run
---------------
1. Restore persisted inputs (widget state is dropped while PROJECT & RESULTS
   is shown, so values are mirrored into plain session keys).
2. ANALYZE STORY (button or Enter) only sets a flag in its on_click callback,
   so the whole monitor can be drawn first: status bar, controls, the story
   box (read-only while a story is being analysed or a verdict is shown),
   an overlay placeholder on top of it, and the bottom band.
3. If a submission is pending, the story is split into (headline, body) and
   validated, the overlay switches to the scan, and the unchanged backend is
   called: get_predictor(kind).predict(headline, body). No artificial delay.
4. The verdict is stored in session_state. The stamp animation is emitted only
   on the run that produced it (`sd-fresh`); later reruns show it at rest.
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

import config
from model import InferenceError, ModelUnavailableError
from preprocessing import InputValidationError
from ui import theme as T
from ui.common import ENGINE_SHORT, engine_status, get_predictor, validation_examples
from ui.story import join_story, parse_story, split_story

PERSIST = {"story": "_saved_story", "engine": "_saved_engine"}
PLACEHOLDER = (
    "Scientists confirm water found on Mars\n"
    "NASA scientists have reported new findings from the latest Mars exploration mission...\n\n"
    "Paste the full article here. Paragraphs and line breaks are kept."
)

# Enter submits, Shift+Enter inserts a line break. Installed once per page load.
# The textarea is blurred first so Streamlit commits the typed text, then the
# ANALYZE STORY button is clicked (the same path as clicking it with the mouse).
ENTER_HOOK = """<script>
(function () {
  if (window.__sdEnterHook) return;
  window.__sdEnterHook = true;
  // Soft keyboards have no Shift+Enter: on touch devices Enter keeps inserting a line break
  // and the ANALYZE STORY button submits.
  if (window.matchMedia && window.matchMedia('(pointer: coarse)').matches) return;
  document.addEventListener('keydown', function (e) {
    if (e.key !== 'Enter' || e.shiftKey || e.isComposing || e.ctrlKey || e.altKey || e.metaKey) return;
    var ta = e.target;
    if (!ta || ta.tagName !== 'TEXTAREA' || !ta.closest('.st-key-story')) return;
    e.preventDefault();
    e.stopPropagation();
    ta.blur();
    setTimeout(function () {
      var b = document.querySelector('.st-key-analyze button');
      if (b && !b.disabled) b.click();
    }, 160);
  }, true);
})();
</script>"""


# ---------------------------------------------------------------------------
# state helpers
# ---------------------------------------------------------------------------
def _restore_inputs(engines: dict) -> None:
    ss = st.session_state
    for key, saved in PERSIST.items():
        if key not in ss and saved in ss:
            ss[key] = ss[saved]
    ss.setdefault("story", "")
    if ss.get("engine") not in engines:  # first run, or engine removed
        ss["engine"] = "bert" if "bert" in engines else (next(iter(engines)) if engines else None)


def _persist_inputs() -> None:
    ss = st.session_state
    for key, saved in PERSIST.items():
        if key in ss:
            ss[saved] = ss[key]


def _on_submit() -> None:
    st.session_state["_submit"] = True


def _on_edit() -> None:
    """Return to editing the same story (verdict removed, text kept)."""
    st.session_state["verdict"] = None


def _on_new_analysis() -> None:
    """Clear the result and the story box for a new story."""
    ss = st.session_state
    ss["story"] = ""
    ss["verdict"] = None
    ss["sample"] = None


_on_clear = _on_new_analysis


def _on_load_sample() -> None:
    ss = st.session_state
    ex = validation_examples()
    want = ss.get("sample_class", "any")
    pool = ex if want == "any" else ex[ex["Stance"] == want]
    row = pool.sample(1).iloc[0]
    ss["story"] = join_story(row["Headline"], row["articleBody"])
    ss["sample"] = {"story": ss["story"].strip(), "gold": row["Stance"]}
    ss["verdict"] = None


def _gold_for(story: str) -> str | None:
    s = st.session_state.get("sample")
    return s["gold"] if s and s["story"] == (story or "").strip() else None


# ---------------------------------------------------------------------------
# view
# ---------------------------------------------------------------------------
def render() -> None:
    ss = st.session_state
    engines = engine_status()
    _restore_inputs(engines)
    ss.setdefault("verdict", None)
    ready = bool(engines)
    pending = ss.pop("_submit", False) and ready

    # Validate a pending submission before drawing, so the box is only locked for a real analysis.
    alert = None
    parsed = None
    if pending:
        try:
            parsed = split_story(ss.get("story", ""))
        except InputValidationError as e:
            alert = {"kicker": "Input required", "title": "The story is incomplete", "msg": str(e), "error": False}
        if parsed:
            ss["verdict"] = None
    locked = parsed is not None or ss.get("verdict") is not None
    if locked:
        # A disabled widget ignores a value arriving from the browser in the same run, so pin the
        # text that is being analysed (typed and submitted in one go) before the box is drawn read-only.
        ss["story"] = ss.get("story", "")
        ss["engine"] = ss.get("engine")

    kind = ss.get("engine")
    engine_label = f"Engine · {ENGINE_SHORT.get(kind, '—')}" if kind else "No engine"

    with st.container(key="monitor"):
        bar = st.empty()
        with st.container(key="monitor_body"):
            with st.container(key="desk"):
                _controls(engines, disabled=locked)
                with st.container(key="surface"):
                    st.text_area("Story: first line headline, then the article", key="story",
                                 placeholder=PLACEHOLDER, disabled=locked, label_visibility="collapsed",
                                 max_chars=2 * config.MAX_INPUT_CHARS)
                    overlay = st.empty()
                with st.container(key="band"):
                    info_col, act_col = st.columns([1.75, 1], gap="medium", vertical_alignment="center")
                    with info_col:
                        foot = st.empty()
                    actions = act_col.container(key="actions")
    st.html(ENTER_HOOK, unsafe_allow_javascript=True)
    _persist_inputs()

    if not ready:
        bar.markdown(T.monitor_bar("alert", engine_label), unsafe_allow_html=True)
        overlay.markdown(T.overlay_alert(
            "System notice", "Prediction engine not ready",
            "No trained model was found in <code>models/</code>. Train one with "
            "<code>python train_baselines.py lr_relational</code> or <code>python train_bert.py</code> "
            "(see README). No prediction is shown rather than a fabricated one.", error=True),
            unsafe_allow_html=True)
        foot.markdown(T.foot_idle(*parse_story(ss.get("story"))), unsafe_allow_html=True)
        _actions(actions, None, ready=False)
        return

    if parsed:
        foot.markdown(T.foot_idle(*parsed), unsafe_allow_html=True)
        alert = _run_analysis(kind, engine_label, parsed, bar, overlay)

    v = ss.get("verdict")
    if alert:
        bar.markdown(T.monitor_bar("alert", engine_label), unsafe_allow_html=True)
        overlay.markdown(T.overlay_alert(alert["kicker"], alert["title"], T.esc(alert["msg"]),
                                         error=alert["error"]), unsafe_allow_html=True)
        foot.markdown(T.foot_idle(*parse_story(ss.get("story"))), unsafe_allow_html=True)
        _actions(actions, None, ready=True)
    elif v:
        fresh = ss.pop("_fresh", False)
        bar.markdown(T.monitor_bar("verdict", f"{v['engine_short']} · {v['device']}"), unsafe_allow_html=True)
        overlay.markdown(T.overlay_verdict(v, fresh), unsafe_allow_html=True)
        foot.markdown(T.foot_verdict(v), unsafe_allow_html=True)
        _actions(actions, v, ready=True)
    else:
        bar.markdown(T.monitor_bar("idle", engine_label), unsafe_allow_html=True)
        overlay.empty()
        foot.markdown(T.foot_idle(*parse_story(ss.get("story"))), unsafe_allow_html=True)
        _actions(actions, None, ready=True)


def _controls(engines: dict, disabled: bool) -> None:
    with st.container(key="controls_row"):
        c1, c2, c3 = st.columns([1.25, 1.5, 0.7], gap="small", vertical_alignment="center")
        with c1:
            st.segmented_control(
                "Engine", options=list(engines), key="engine", required=True, disabled=disabled,
                format_func=lambda k: ENGINE_SHORT.get(k, k),
                help="BERT (fine-tuned here) is the default. Logistic Regression is the CPU fallback engine, "
                     "also trained here. Switching keeps your text.",
            )
        with c2:
            T.md(T.format_hint())
        with c3:
            with st.popover("Sample story", use_container_width=True, key="sample_pop", disabled=disabled,
                            help="Load a real labelled pair from this project's validation split."):
                st.caption("Real headline–article pairs from the FNC-1 validation split used in this "
                           "project (never the test set). The gold label is shown next to the verdict.")
                st.selectbox("Gold stance", ["any"] + config.DISPLAY_ORDER, key="sample_class")
                st.button("Load random example", key="load_sample", on_click=_on_load_sample,
                          use_container_width=True)


def _actions(actions, v: dict | None, ready: bool) -> None:
    with actions:
        if v is None:
            c1, c2 = st.columns([1, 2.3], gap="small")
            c1.button("Clear", key="clear_inputs", on_click=_on_clear, use_container_width=True,
                      help="Empty the story box.")
            c2.button("Analyze story", key="analyze", on_click=_on_submit, disabled=not ready,
                      use_container_width=True,
                      help="Classify how the article relates to its headline (or press Enter in the story box).")
        else:
            c1, c2, c3 = st.columns([1.15, 1, 1.15], gap="small")
            with c1:
                with st.popover("Report", use_container_width=True, key="report_pop",
                                help="Class scores, what the stance means, caveats and filing details."):
                    T.md(T.verdict_report(v))
                    st.button("Clear & start over", key="clear_all", on_click=_on_new_analysis,
                              use_container_width=True, help="Empty the story box and remove this verdict.")
            c2.button("Edit story", key="edit_story", on_click=_on_edit, use_container_width=True,
                      help="Remove the verdict and edit the same story.")
            c3.button("New analysis", key="new_analysis", on_click=_on_new_analysis, use_container_width=True,
                      help="Clear the result and the story box to analyse another story.")


def _run_analysis(kind: str, engine_label: str, parsed: tuple[str, str], bar, overlay) -> dict | None:
    """Run the real model on (headline, body). Returns an alert dict on failure, else None."""
    ss = st.session_state
    headline, body = parsed

    # The monitor goes live while the real model call runs; the scan runs over the story box itself.
    bar.markdown(T.monitor_bar("analysing", engine_label), unsafe_allow_html=True)
    overlay.markdown(T.overlay_analysing(), unsafe_allow_html=True)

    def fail(title: str, msg: str) -> dict:
        return {"kicker": "Analysis failed", "title": title, "msg": msg, "error": True}

    try:
        predictor = get_predictor(kind)
        r = predictor.predict(headline, body)
    except InputValidationError as e:
        return {"kicker": "Input required", "title": "The story is incomplete", "msg": str(e), "error": False}
    except ModelUnavailableError as e:
        return fail("Prediction engine not ready", str(e))
    except MemoryError:
        return fail("Out of memory", "The computer ran out of memory while running the model. "
                                     "Close other applications and try again.")
    except InferenceError as e:
        return fail("Inference failed", str(e))

    ss["verdict"] = {
        "label": r.label,
        "probabilities": dict(r.probabilities),
        "model_name": r.model_name,
        "device": r.device,
        "elapsed_ms": float(r.elapsed_ms),
        "truncated": bool(r.truncated),
        "n_tokens": r.n_tokens,
        "max_length": int(getattr(predictor, "max_length", config.BERT["max_length"])),
        "notes": list(r.notes),
        "headline": headline,
        "body": body,
        "engine": kind,
        "engine_short": ENGINE_SHORT.get(kind, kind),
        "gold": _gold_for(ss.get("story", "")),
        "filed": datetime.now().strftime("%H:%M:%S"),
    }
    ss["_fresh"] = True
    return None
