"""Visual system for The Stance Desk: stylesheet injection and HTML builders.

Every builder returns *compact* HTML (no blank lines, no 4-space indented
lines) because st.markdown treats those as Markdown paragraphs / code blocks.
All user-supplied text goes through `esc()` before it reaches the page.
"""
from __future__ import annotations

import html
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import streamlit as st

import config

STATIC = Path(__file__).resolve().parent / "static"
CSS_PATH = STATIC / "stance_desk.css"
MAP_PATH = STATIC / "world_map.svg"

APP_NAME = "The Stance Desk"
TAGLINE = "Newsroom intelligence · Headline–article analysis"

CLASS_COLORS = {  # mirrors the --c-* tokens in the stylesheet
    "agree": "#3fa08c",
    "disagree": "#e0473d",
    "discuss": "#e0a630",
    "unrelated": "#7d9cc2",
}
# Shown on the stamp around the stance word. Deliberately descriptive, never
# truth claims (this is stance detection, not fact-checking).
STAMP_LINES = {
    "agree": "Article supports headline",
    "disagree": "Article contradicts headline",
    "discuss": "Article reports, no position",
    "unrelated": "Article off-topic",
}


def esc(text) -> str:
    return html.escape("" if text is None else str(text), quote=True)


def md(markup: str) -> None:
    """Render one compact HTML fragment."""
    st.markdown(markup, unsafe_allow_html=True)


@lru_cache(maxsize=2)
def _assets(css_mtime: float, map_mtime: float) -> str:  # mtimes bust the cache when files change
    css = CSS_PATH.read_text(encoding="utf-8")
    world = MAP_PATH.read_text(encoding="utf-8") if MAP_PATH.exists() else ""
    var = f':root{{--world-map:url("data:image/svg+xml;utf8,{quote(world)}");}}' if world else ""
    return var + css


def apply_theme() -> None:
    if CSS_PATH.exists():
        mm = MAP_PATH.stat().st_mtime if MAP_PATH.exists() else 0.0
        md(f"<style>{_assets(CSS_PATH.stat().st_mtime, mm)}</style>")


def excerpt(text: str, limit: int = 420) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def score(x: float) -> str:
    """Model score for display. Never rounds up to a misleading 100.0% or down to 0.0%."""
    if x >= 0.999:
        return ">99.9%"
    if x < 0.001:
        return "<0.1%"
    return f"{x * 100:.1f}%"


# ---------------------------------------------------------------------------
# broadcast header + bottom strip
# ---------------------------------------------------------------------------
def masthead() -> None:
    md(
        '<header class="sd-mast">'
        '<div class="sd-bug" aria-hidden="true"><b>Breaking</b><em>News</em></div>'
        '<div class="sd-brand">'
        '<div class="sd-title" role="heading" aria-level="1">The Stance Desk<span>.</span></div>'
        f'<div class="sd-sub">{esc(TAGLINE)}</div>'
        "</div>"
        '<div class="sd-onair" role="img" aria-label="Live"><span>Live</span><i></i></div>'
        "</header>"
    )


# ---------------------------------------------------------------------------
# monitor chrome
# ---------------------------------------------------------------------------
def monitor_bar(state: str, engine_label: str) -> str:
    """state: idle | analysing | verdict | alert"""
    live = {
        "idle": '<span class="sd-live"><i></i>Standby</span>',
        "analysing": '<span class="sd-live sd-live--on" role="status"><i></i>On air · Analysis</span>',
        "verdict": '<span class="sd-live sd-live--verdict"><i></i>Verdict issued</span>',
        "alert": '<span class="sd-live sd-live--alert"><i></i>Held</span>',
    }[state]
    return (
        '<div class="sd-mbar">'
        f"{live}"
        '<span class="sd-ch">Monitor <b>01</b> · Stance analysis desk · <b>not fact-checking</b></span>'
        '<span class="sd-spacer"></span>'
        f'<span class="sd-eng">{esc(engine_label)}</span>'
        "</div>"
    )


ANALYSIS_STEPS = ("Analyzing headline", "Comparing article context", "Classifying stance")


def format_hint() -> str:
    return ('<div class="sd-format"><span class="sd-num">01</span>First line — <b>headline</b>'
            '<i>|</i><span class="sd-num">02</span>Below — <b>article content</b></div>')


def stamp(label: str, confidence: float, engine_short: str) -> str:
    return (
        f'<div class="sd-stamp sd-stamp--{label}" role="img" aria-label="Predicted stance: {label.upper()}">'
        f'<span class="sd-st-top">{esc(STAMP_LINES[label])}</span>'
        f'<span class="sd-st-word">{label.upper()}</span>'
        f'<span class="sd-st-bot">{esc(engine_short)} · score {score(confidence)}</span>'
        "</div>"
    )


def _mini_scores(probs: dict[str, float], top: str) -> str:
    cells = "".join(
        f'<div class="{"sd-ms sd-ms--top" if l == top else "sd-ms"}" style="--c:{CLASS_COLORS[l]}">'
        f'<span>{l}</span><i><b style="width:{probs[l] * 100:.2f}%"></b></i><em>{score(probs[l])}</em></div>'
        for l in config.DISPLAY_ORDER)
    return f'<div class="sd-mini-scores" aria-label="Class scores">{cells}</div>'


def verdict_flags(v: dict) -> list[tuple[str, str]]:
    """(short chip text, full sentence) for every caveat that applies to this verdict."""
    out = []
    if v["probabilities"][v["label"]] < 0.5:
        out.append(("Low confidence", "Low confidence: the model is unsure between several classes. "
                                      "Treat this prediction with caution."))
    if v.get("truncated"):
        out.append((f"Truncated · {v['max_length']} of {v['n_tokens']:,} tokens",
                    f"The input is {v['n_tokens']:,} tokens long; only the first {v['max_length']} tokens "
                    "(headline + start of the article) were read by BERT. Stance expressed later in the "
                    "article can be missed."))
    for n in v.get("notes") or []:
        out.append(("CPU fallback" if "CPU" in n else "Note", n))
    return out


# ---------------------------------------------------------------------------
# the story surface: one text box; scan + stamp are overlaid on it
# ---------------------------------------------------------------------------
def overlay_analysing() -> str:
    steps = '<i>›</i>'.join(f"<span>{s}</span>" for s in ANALYSIS_STEPS)
    return (
        '<div class="sd-ov sd-ov--scan" aria-busy="true">'
        '<div class="sd-scan"></div>'
        f'<div class="sd-ov-status" role="status" aria-live="polite"><b>●</b> Live analysis <em>{steps}</em></div>'
        "</div>"
    )


def overlay_verdict(v: dict, fresh: bool) -> str:
    label = v["label"]
    gold = ""
    if v.get("gold"):
        match = "match" if v["gold"] == label else "differs"
        gold = f'<span class="sd-gold">FNC-1 gold · <b>{esc(v["gold"])}</b> · {match}</span>'
    cls = f"sd-ov sd-ov--verdict sd-ov--{label}" + (" sd-fresh" if fresh else "")
    return (
        f'<div class="{cls}" style="--c:{CLASS_COLORS[label]}">'
        '<div class="sd-sweep"></div>'
        f'<div class="sd-ov-status sd-ov-status--verdict"><b>●</b> Verdict issued · {esc(v["filed"])}{gold}</div>'
        '<div class="sd-stage">'
        '<div class="sd-ring"></div>'
        + stamp(label, v["probabilities"][label], v["engine_short"])
        + "</div>"
        f'<span class="sd-sr">Predicted stance: {label.upper()}</span>'
        "</div>"
    )


def overlay_alert(kicker: str, title: str, message_html: str, error: bool = False) -> str:
    cls = "sd-ov-alert sd-ov-alert--err" if error else "sd-ov-alert"
    return (
        '<div class="sd-ov sd-ov--alert">'
        f'<div class="{cls}" role="alert"><div class="sd-k">{esc(kicker)}</div>'
        f'<div class="sd-halert" role="heading" aria-level="3">{esc(title)}</div><p>{message_html}</p></div>'
        "</div>"
    )


def foot_idle(headline: str, body: str) -> str:
    if headline:
        hl = f'<span class="sd-detect">Headline detected · <b>{esc(headline)}</b></span>'
    else:
        hl = '<span class="sd-detect sd-detect--none">No headline yet · type it on the first line</span>'
    words = len(body.split())
    return (
        '<div class="sd-foot-info">'
        f'<div class="sd-fi-row">{hl}<span class="sd-count">{len(body):,} characters · {words:,} words in article</span></div>'
        '<div class="sd-fi-row sd-fi-row--keys"><span><kbd>Enter</kbd> analyze story</span>'
        '<span><kbd>Shift</kbd> + <kbd>Enter</kbd> new line</span>'
        '<span class="sd-legend-inline">'
        + "".join(f'<em style="color:{CLASS_COLORS[l]}">{l}</em>' for l in config.DISPLAY_ORDER)
        + "</span></div></div>"
    )


def foot_verdict(v: dict) -> str:
    label = v["label"]
    chips = "".join(f'<span class="sd-chip" title="{esc(full)}">{esc(short)}</span>'
                    for short, full in verdict_flags(v))
    return (
        f'<div class="sd-foot-info sd-foot-info--verdict" style="--c:{CLASS_COLORS[label]}">'
        '<div class="sd-fi-row">'
        f'<span class="sd-vword">{label}</span>'
        f'<span class="sd-vmean">{esc(config.LABEL_EXPLANATIONS[label])}</span>'
        f'<span class="sd-vmeta">{esc(v["engine_short"])} · {esc(v["device"])} · {v["elapsed_ms"]:.0f} ms</span>'
        f"{chips}</div>"
        + _mini_scores(v["probabilities"], label)
        + "</div>"
    )


# ---------------------------------------------------------------------------
# full verdict report (opened from the monitor in a popover)
# ---------------------------------------------------------------------------
def distribution(probs: dict[str, float], top: str) -> str:
    rows = []
    for lab in config.DISPLAY_ORDER:
        p = probs[lab]
        cls = "sd-row sd-row--top" if lab == top else "sd-row"
        rows.append(
            f'<div class="{cls}" style="--c:{CLASS_COLORS[lab]}">'
            f'<span class="sd-lab">{lab}</span>'
            f'<span class="sd-track"><span class="sd-fill" style="width:{p * 100:.2f}%"></span></span>'
            f'<span class="sd-val">{score(p)}</span></div>'
        )
    return '<div class="sd-dist">' + "".join(rows) + "</div>"


def verdict_report(v: dict, fresh: bool = False) -> str:
    label = v["label"]
    c = CLASS_COLORS[label]
    flags = "".join(f'<div class="sd-flag">{esc(full)}</div>' for _, full in verdict_flags(v))
    tokens = ""
    if v.get("n_tokens") is not None:
        tokens = (f"<dt>Tokens</dt><dd>{v['n_tokens']:,} of {v['max_length']} max"
                  f"{' · truncated' if v.get('truncated') else ''}</dd>")
    gold = f"<dt>Gold label</dt><dd>{esc(v['gold'])} (FNC-1 validation split)</dd>" if v.get("gold") else ""
    return (
        f'<div class="sd-report" style="--c:{c}">'
        "<section>"
        '<div class="sd-h"><span>Class scores</span><b>Predicted stance · ' + label.upper() + "</b></div>"
        + distribution(v["probabilities"], label)
        + '<div class="sd-fine"><b>Scores are model outputs, not calibrated probabilities.</b> '
        "They sum to 100% (softmax), but a 99% score does not mean a 99% chance of being right: the BERT "
        "model is often highly confident even on mistakes.</div>"
        "</section>"
        '<section class="sd-meaning">'
        '<div class="sd-h"><span>What this means</span></div>'
        f'<div class="sd-word">{label}</div>'
        f"<p>{esc(config.LABEL_EXPLANATIONS[label])}</p>"
        + flags
        + '<div class="sd-fine"><b>Stance is not truth.</b> This describes how the article relates to the '
        "headline. It does not say whether the claim is true.</div>"
        "</section>"
        "<section>"
        '<div class="sd-h"><span>Filing details</span></div>'
        '<dl class="sd-dl">'
        f"<dt>Model</dt><dd>{esc(v['model_name'])}</dd>"
        f"<dt>Engine</dt><dd>{esc(v['engine_short'])}</dd>"
        f"<dt>Device</dt><dd>{esc(v['device'])}</dd>"
        f"<dt>Inference</dt><dd>{v['elapsed_ms']:.0f} ms</dd>"
        f"{tokens}"
        f"<dt>Input</dt><dd>{len(v['headline']):,} + {len(v['body']):,} characters</dd>"
        f"{gold}"
        f"<dt>Filed</dt><dd>{esc(v['filed'])}</dd>"
        "</dl>"
        "</section>"
        "</div>"
    )


def disclaimer() -> str:
    return (
        '<div class="sd-disclaimer"><b>Not fact-checking</b><span>Stance detection is not fact-checking. '
        "The desk judges how an article relates to a headline (agree, disagree, discuss, unrelated). "
        "It does not decide whether a claim is true or false.</span></div>"
    )


# ---------------------------------------------------------------------------
# editorial helpers for PROJECT & RESULTS
# ---------------------------------------------------------------------------
def section_head(num: int, kicker: str, title: str, standfirst: str = "") -> str:
    sf = f'<p class="sd-standfirst">{standfirst}</p>' if standfirst else ""
    return (f'<div class="sd-sec-kicker">{num:02d} / {esc(kicker)}</div>'
            f'<div class="sd-sec-title" role="heading" aria-level="2">{esc(title)}</div>{sf}')


def h3(title: str, note: str = "") -> str:
    small = f"<small>{note}</small>" if note else ""
    return f'<div class="sd-h3" role="heading" aria-level="3"><span>{esc(title)}</span>{small}</div>'


def badge(kind: str) -> str:
    text = {"ours": "Trained here", "hist": "Historical · not reproduced", "ref": "Reference"}[kind]
    return f'<span class="sd-badge sd-badge--{kind}">{text}</span>'
