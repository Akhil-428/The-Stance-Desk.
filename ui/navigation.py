"""App shell: masthead + two-way navigation (PREDICT | PROJECT & RESULTS).

There is no sidebar and no multipage router. The active view lives in
st.session_state["view"]; PREDICT is the default whenever the app opens.

`run_app(default)` also accepts the former page names ("home", "predict",
"performance", "comparison", "about") so older scripts and tests keep working,
and the URL query `?view=results&section=performance` deep-links a section
(read once per session; the URL is never rewritten, so a plain reload of
http://localhost:8501 always opens PREDICT).
"""
from __future__ import annotations

import streamlit as st

from ui import predict_view, results_view
from ui.theme import APP_NAME, apply_theme, masthead

VIEWS = {"predict": "Predict", "results": "Project & results"}
LEGACY = {
    "home": ("predict", None),
    "predict": ("predict", None),
    "results": ("results", None),
    "performance": ("results", "performance"),
    "comparison": ("results", "comparison"),
    "about": ("results", "overview"),
}


def _initial_state(default: str) -> None:
    ss = st.session_state
    if "_nav_init" in ss:
        return
    ss["_nav_init"] = True
    view, section = LEGACY.get(default, ("predict", None))
    qp = st.query_params
    if qp.get("view") in VIEWS:
        view = qp.get("view")
    if qp.get("section") in results_view.SECTION_KEYS:
        view, section = "results", qp.get("section")
    ss["view"] = view
    if section:
        ss["section"] = section


def run_app(default: str = "predict") -> None:
    st.set_page_config(page_title=APP_NAME, page_icon=":material/newspaper:", layout="wide",
                       initial_sidebar_state="collapsed")
    apply_theme()
    _initial_state(default)
    ss = st.session_state
    if ss.get("view") not in VIEWS:  # clicking the active segment deselects it; keep the last view
        ss["view"] = ss.get("_last_view", "predict")

    masthead()
    with st.container(key="nav_wrap"):
        st.segmented_control("Section", options=list(VIEWS), key="view", format_func=VIEWS.get, required=True,
                             label_visibility="collapsed")
    view = ss.get("view") or ss.get("_last_view", "predict")
    ss["_last_view"] = view

    if view == "results":
        with st.container(key="view_results"):
            results_view.render()
    else:
        with st.container(key="view_predict"):
            predict_view.render()
